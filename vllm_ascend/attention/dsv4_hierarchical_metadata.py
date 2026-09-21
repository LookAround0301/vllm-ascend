# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Per-cache DS V4 address translation, without changing logical lifetimes.

Builders may share metadata between several caches. Only phase objects and
address tensors are copied here; lengths, RoPE and tiling metadata stay shared.
The worker must authorize each positive logical block for this cache before
publishing ``index_map``. An entry is a backend address ID, never a token slot.
"""

from __future__ import annotations

from collections.abc import Sequence
from copy import copy
from typing import TYPE_CHECKING, Any

import torch
from vllm.triton_utils import tl, triton

if TYPE_CHECKING:
    from vllm_ascend.core.dsv4_cache_pages import DSV4CachePage


@triton.jit
def _canonicalize_dsv4_slots_kernel(
    slots_ptr, output_ptr, rows, row_stride, column_stride, TILE: tl.constexpr,
):
    index = tl.program_id(0) * TILE + tl.arange(0, TILE)
    mask = index < rows
    block = tl.load(slots_ptr + index * row_stride, mask, 0).to(tl.int64)
    row = tl.load(slots_ptr + index * row_stride + column_stride, mask, 0).to(tl.int64)
    valid = (block > 0) & (row >= 0)
    tl.store(output_ptr + index * 2, tl.where(valid, block, -1), mask)
    tl.store(output_ptr + index * 2 + 1, tl.where(valid, row, 0), mask)


def sanitize_dsv4_scatter_slots(slots: torch.Tensor) -> torch.Tensor:
    """Widen native BLOCK_OFFSET pairs before scatter address arithmetic.

    Compressor rows are already bounded by the native block size. Canonicalize
    invalid pairs to (-1, 0): (-1, positive_row) can address live storage through
    the Indexer's overlapping 64-byte views. Block zero is reserved for reads.
    """
    rows = slots.shape[0]
    # Eight INT64 pairs fill one 128-byte line. Pad the final AIV write tile.
    output = torch.empty((triton.cdiv(rows, 8) * 8, 2), dtype=torch.int64, device=slots.device)
    if rows:
        tile_size = 256
        _canonicalize_dsv4_slots_kernel[(triton.cdiv(rows, tile_size),)](
            slots, output, rows, slots.stride(0), slots.stride(1), tile_size,
        )
    return output[:rows]


# Batch sizes and input slice alignment vary as requests join or finish.
# Keep them dynamic so those changes reuse the same compiled kernel.
@triton.jit(do_not_specialize=[
    "table_ptr", "lengths_ptr", "query_ptr", "slots_ptr", "maps_ptr",
    "num_rows", "num_requests", "num_slots", "num_tiles", "max_query_len",
])
def _remap_dsv4_metadata_kernel(
    table_ptr, lengths_ptr, query_ptr, slots_ptr, maps_ptr,
    physical_tables_ptr, physical_slots_ptr, errors_ptr,
    table_row_stride, table_col_stride, length_stride, query_stride,
    output_table_stride, output_slot_stride,
    slot_row_stride, slot_col_stride, map_row_stride, map_col_stride,
    num_rows, num_cols, num_requests,
    num_slots, HAS_SLOTS: tl.constexpr, map_size,
    BLOCK_SIZE: tl.constexpr, COMPRESS_RATIO: tl.constexpr,
    SLIDING_WINDOW: tl.constexpr, num_tiles, TILE_SIZE: tl.constexpr,
    window_cols, max_query_len, WINDOWED: tl.constexpr,
):
    cache = tl.program_id(0)
    tile = tl.program_id(1)
    index = tile * TILE_SIZE + tl.arange(0, TILE_SIZE)
    map_base = maps_ptr + cache * map_row_stride
    invalid = tl.load(map_base) != 0
    invalid = invalid | (tl.load(query_ptr) != 0)

    # Check requests independently of table entries, including zero-column
    # dummy tables. Padded rows may contain stale lengths and are ignored.
    request_mask = index < num_requests
    length = tl.load(lengths_ptr + index * length_stride, request_mask, 0).to(tl.int64)
    query_begin = tl.load(query_ptr + index * query_stride, request_mask, 0).to(tl.int64)
    query_end = tl.load(query_ptr + (index + 1) * query_stride, request_mask, 0).to(tl.int64)
    query_length = query_end - query_begin
    used_columns = (length // COMPRESS_RATIO + BLOCK_SIZE - 1) // BLOCK_SIZE
    invalid = invalid | tl.max(
        (request_mask & ((length < 0) | (used_columns > num_cols)
                         | (query_length < 0) | (query_length > length))).to(tl.int32), 0
    )

    if WINDOWED:
        invalid = invalid | tl.max((request_mask & (query_length > max_query_len)).to(tl.int32), 0)
    table_width = window_cols if WINDOWED else num_cols
    row = index // tl.maximum(table_width, 1)
    column = index % tl.maximum(table_width, 1)
    table_mask = index < num_rows * table_width
    active_row = table_mask & (row < num_requests)
    length = tl.load(lengths_ptr + row * length_stride, active_row, 0).to(tl.int64)
    used_columns = (length // COMPRESS_RATIO + BLOCK_SIZE - 1) // BLOCK_SIZE
    if SLIDING_WINDOW > 0:
        query_begin = tl.load(query_ptr + row * query_stride, active_row, 0).to(tl.int64)
        query_end = tl.load(query_ptr + (row + 1) * query_stride, active_row, 0).to(tl.int64)
        first_query = length - (query_end - query_begin)
        skipped_columns = tl.maximum(first_query - SLIDING_WINDOW + 1, 0) // BLOCK_SIZE
        if WINDOWED:
            # Aligned starts keep adjacent AIV tiles on separate 128-byte lines.
            column += skipped_columns // 32 * 32
    table_mask = table_mask & (column < num_cols)
    active = table_mask & (row < num_requests) & (column < used_columns)
    if SLIDING_WINDOW > 0:
        active = active & (column >= skipped_columns)
    logical = tl.load(table_ptr + row * table_row_stride + column * table_col_stride, active, 0).to(tl.int64)
    mapped = tl.load(map_base + tl.minimum(tl.maximum(logical, 0), map_size - 1) * map_col_stride)
    invalid = invalid | tl.max(
        ((logical < 0) | (logical >= map_size) | ((logical > 0) & (mapped <= 0))).to(tl.int32), 0
    )
    tl.store(physical_tables_ptr + cache * output_table_stride + row * num_cols + column,
             tl.where(logical > 0, mapped, 0), table_mask)

    if HAS_SLOTS:
        token_end = tl.load(query_ptr + num_requests * query_stride).to(tl.int64)
        invalid = invalid | (token_end > num_slots)
        slot_mask = index < num_slots
        logical = tl.load(slots_ptr + index * slot_row_stride, slot_mask, 0).to(tl.int64)
        offset = tl.load(slots_ptr + index * slot_row_stride + slot_col_stride, slot_mask, 0).to(tl.int64)
        valid = slot_mask & (index < token_end) & (logical > 0) & (offset >= 0)
        mapped = tl.load(map_base + tl.minimum(tl.maximum(logical, 0), map_size - 1) * map_col_stride)
        invalid = invalid | tl.max(
            (valid & ((logical >= map_size) | (offset >= BLOCK_SIZE) | (mapped <= 0))).to(tl.int32), 0
        )
        output = physical_slots_ptr + cache * output_slot_stride + index * 2
        tl.store(output, tl.where(valid, mapped.to(tl.int64), -1), slot_mask)
        tl.store(output + 1, tl.where(valid, offset, 0), slot_mask)

    # Each program owns a full 128-byte line: independent AIV stores must not
    # overwrite neighboring programs' status through cache-line writeback.
    error_words = tl.arange(0, 32)
    tl.store(errors_ptr + (cache * num_tiles + tile) * 32 + error_words, invalid.to(tl.int32))


def _remap_dsv4_phase(
    phase: Any,
    device_maps: torch.Tensor,
    page: DSV4CachePage,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    table, lengths, query_start = phase.block_table, phase.seq_lens, phase.query_start_loc
    rows, cols = table.shape
    slots = phase.slot_mapping
    slot_rows = 0 if slots is None else slots.shape[0]
    caches = device_maps.shape[0]
    # Cache slices begin on separate 128-byte lines even for tiny tables.
    table_elements = triton.cdiv(rows * cols, 32) * 32
    max_query_len = getattr(phase, "max_query_len", getattr(phase, "max_seqlen_q", 0)) or 0
    window_cols = 0
    if page.sliding_window and page.compress_ratio == 1 and max_query_len and cols % 32 == 0:
        # Cover the union of query windows, including a partially used page
        # and up to 31 columns before the aligned start. Other table entries
        # remain zero. Unaligned rows retain the full-table path so separate
        # AIV programs never write different parts of the same cache line.
        window_cols = triton.cdiv(
            triton.cdiv(page.sliding_window + max_query_len - 1, page.block_size) + 32, 32
        ) * 32
        if window_cols >= cols:
            window_cols = 0
    allocate_table = torch.zeros if window_cols else torch.empty
    table_storage = allocate_table((caches, table_elements), dtype=torch.int32, device=table.device)
    physical_tables = table_storage[:, :rows * cols].view(caches, rows, cols)
    physical_slots = None
    if slots is not None:
        slot_storage = torch.empty((caches, triton.cdiv(slot_rows, 8) * 8, 2), dtype=torch.int64, device=table.device)
        physical_slots = slot_storage[:, :slot_rows]
    tile_size = 256
    num_tiles = triton.cdiv(max(rows * (window_cols or cols), slot_rows, rows, 1), tile_size)
    errors = torch.empty((caches, num_tiles, 32), dtype=torch.int32, device=table.device)
    _remap_dsv4_metadata_kernel[(caches, num_tiles)](
        table, lengths, query_start, table if slots is None else slots, device_maps,
        physical_tables, physical_tables if physical_slots is None else physical_slots, errors,
        table.stride(0), table.stride(1), lengths.stride(0), query_start.stride(0),
        physical_tables.stride(0), 0 if physical_slots is None else physical_slots.stride(0),
        0 if slots is None else slots.stride(0), 0 if slots is None else slots.stride(1),
        device_maps.stride(0), device_maps.stride(1),
        rows, cols, phase.num_reqs_actual, slot_rows, slots is not None, device_maps.shape[1],
        page.block_size, page.compress_ratio, page.sliding_window or 0, num_tiles, tile_size,
        window_cols, max_query_len if window_cols else 0, bool(window_cols),
    )
    return physical_tables, physical_slots, errors.flatten()


def remap_dsv4_metadata_group(
    metadata: Any,
    index_maps: torch.Tensor,
    pages: Sequence[DSV4CachePage],
    *,
    validation_errors: list[torch.Tensor] | None = None,
) -> list[Any]:
    """Translate native phase metadata using the worker's per-cache maps.

    Group membership and cache layouts are established during initialization.
    Only phase objects and address tensors are copied; RoPE, lengths and tiling
    stay shared. Active addresses are checked on device, then checked together
    across groups before model execution. Padding and expired SWA prefixes must
    be masked before lookup because native builder buffers can retain old IDs.
    """
    if metadata is None:
        return [None] * len(pages)
    results = [copy(metadata) for _ in pages]
    device_errors: list[torch.Tensor] = []
    for phase_name in ("prefill", "decode"):
        phase = getattr(metadata, phase_name, None)
        if phase is None:
            continue
        physical_tables, physical_slots, errors = _remap_dsv4_phase(phase, index_maps, pages[0])
        device_errors.append(errors)
        for cache_index, (result, cache_page) in enumerate(zip(results, pages)):
            copied_phase = copy(phase)
            copied_phase.block_table = physical_tables[cache_index]
            if physical_slots is not None:
                copied_phase.slot_mapping = physical_slots[cache_index]
            copied_phase.omoe_dsv4_physical = True
            copied_phase.omoe_dsv4_cache_name = cache_page.name
            setattr(result, phase_name, copied_phase)

    if validation_errors is None:
        validate_dsv4_metadata(device_errors)
    else:
        validation_errors.extend(device_errors)
    for result, cache_page in zip(results, pages):
        result.omoe_dsv4_physical = True
        result.omoe_dsv4_cache_name = cache_page.name
    return results


def validate_dsv4_metadata(errors: list[torch.Tensor]) -> None:
    """Finish device address checks before any native attention consumes them."""
    if not errors:
        return
    combined = errors[0] if len(errors) == 1 else torch.cat(errors)
    # Only the error decision crosses to the host, not every padded flag.
    if combined.any().item():
        raise ValueError(
            "Invalid DS V4 hierarchical metadata: query offsets, length, active table block, "
            "live slot block/row is unmapped or out of range, or index_map zero is not null"
        )
