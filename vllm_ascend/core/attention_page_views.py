# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Contiguous native K/V views over interleaved shared arena pages."""

from copy import copy

import torch

_INDEX_DTYPES = (torch.int32, torch.int64)
_CACHE_DTYPES = (torch.float16, torch.bfloat16)
_PAGES_PER_PAIR = 2


def make_paired_kv_views(
    arena: torch.Tensor,
    native_block_size: int,
    heads: int,
    head_size: int,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Expose native contiguous K/V tensors without copying shared storage.

    Physical bytes are [K0, V0, K1, V1, ...]. For N logical pairs, each
    tensor exposes 2*N-1 native pages, with V's base shifted by one page.
    The native block ID for logical pair b is 2*b in both tensors. Odd
    native pages must never be allocated or referenced by the scheduler.
    Whole-pair ownership and reserving null pair zero belong to the allocator.
    """
    if dtype not in _CACHE_DTYPES:
        raise ValueError("Paired native KV views require float16 or bfloat16 cache dtype.")
    if arena.dtype not in (torch.int8, torch.uint8) or arena.ndim != 1 or not arena.is_contiguous():
        raise ValueError("Paired KV arena must be a contiguous one-dimensional byte tensor.")
    element_size = torch.empty((), dtype=dtype).element_size()
    page_bytes = native_block_size * heads * head_size * element_size
    pair_bytes = _PAGES_PER_PAIR * page_bytes
    if arena.numel() < pair_bytes or arena.numel() % pair_bytes:
        raise ValueError("Paired KV arena must contain a positive integral number of complete K/V pairs.")
    if arena.data_ptr() % element_size:
        raise ValueError("Paired KV arena is not aligned to the cache dtype.")
    native_pages = arena.numel() // page_bytes - 1
    shape = (native_pages, native_block_size, heads, head_size)
    key = arena[:-page_bytes].view(dtype).view(shape)
    value = arena[page_bytes:].view(dtype).view(shape)
    return key, value


def remap_paired_kv_metadata(metadata, native_block_size: int):
    """Copy Ascend metadata and convert logical pairs to native page indices.

    Block tables double nonnegative block IDs. Token slots double only the
    block portion, retaining the intra-block token offset. Negative padding
    sentinels and null block zero are preserved. No metadata tensor is copied
    back to the host or mutated, so several layers may share the input object.
    Use the same native block size as the cache spec and metadata builder.
    """
    block_tables = getattr(metadata, "block_tables", None)
    slots = getattr(metadata, "slot_mapping", None)
    for tensor, ndim, name in ((block_tables, 2, "block_tables"), (slots, 1, "slot_mapping")):
        if tensor is not None and (tensor.dtype not in _INDEX_DTYPES or tensor.ndim != ndim):
            raise ValueError(f"Paired KV {name} must be an int32/int64 tensor with {ndim} dimensions.")
    if block_tables is not None and slots is not None and block_tables.device != slots.device:
        raise ValueError("Paired KV block tables and slots must be on the same device.")
    result = copy(metadata)
    result.omoe_paired_cache = True
    if block_tables is not None:
        result.block_tables = torch.where(block_tables < 0, block_tables, block_tables * _PAGES_PER_PAIR)
    if slots is not None:
        block_offsets = torch.div(slots, native_block_size, rounding_mode="floor") * native_block_size
        result.slot_mapping = torch.where(slots < 0, slots, slots + block_offsets)
    return result


def paired_kv_attention(
    query, key, value, metadata, *, num_heads, num_kv_heads, scale, block_size,
    current_key, current_value, attn_mask,
):
    """Use BNSD for historical KV and native TND for uncached suffixes.

    Large shared caches exceed the TND kernel's 32-bit element offsets.
    Requests with no cached prefix can use their current K/V directly,
    retaining native prefill batching without reading the large cache.
    """
    import torch_npu

    cached_requests = 0
    cached_tokens = 0
    for request in range(len(metadata.seq_lens_list) - 1, -1, -1):
        end = metadata.actual_seq_lengths_q[request]
        start = metadata.actual_seq_lengths_q[request - 1] if request else 0
        if metadata.seq_lens_list[request] != end - start:
            cached_requests = request + 1
            cached_tokens = end
            break

    if cached_tokens:
        request_indices = []
        context_lens = []
        single_token_queries = True
        start = 0
        for request in range(cached_requests):
            end = metadata.actual_seq_lengths_q[request]
            kv_len = metadata.seq_lens_list[request]
            count = end - start
            single_token_queries = single_token_queries and count == 1
            request_indices.extend([request] * count)
            context_lens.extend(range(kv_len - count + 1, kv_len + 1))
            start = end
        if single_token_queries:
            # Decode already has one query per block-table row. Reuse the
            # view instead of synchronously copying an identity index to NPU.
            block_table = metadata.block_tables[:cached_requests]
        else:
            indices = torch.tensor(request_indices, dtype=torch.int64, device=query.device)
            block_table = metadata.block_tables.index_select(0, indices)
        cached_query = query if cached_tokens == query.shape[0] else query[:cached_tokens]
        cached_output, lse = torch_npu.npu_fused_infer_attention_score(
            query=cached_query.unsqueeze(2), key=key, value=value,
            num_key_value_heads=num_kv_heads, num_heads=num_heads, scale=scale,
            block_table=block_table, block_size=block_size, input_layout="BNSD",
            actual_seq_lengths=[1] * cached_tokens, actual_seq_lengths_kv=context_lens,
            sparse_mode=0,
        )
        cached_output = cached_output.squeeze(2)
        if cached_tokens == query.shape[0]:
            return cached_output, lse

    prefill_lengths = [end - cached_tokens for end in metadata.actual_seq_lengths_q[cached_requests:]]
    prefill_query = query[cached_tokens:] if cached_tokens else query
    prefill_key = current_key[cached_tokens:] if cached_tokens else current_key
    prefill_value = current_value[cached_tokens:] if cached_tokens else current_value
    prefill_output, lse = torch_npu.npu_fused_infer_attention_score(
        query=prefill_query, key=prefill_key.contiguous(),
        value=prefill_value.contiguous(), atten_mask=attn_mask,
        num_key_value_heads=num_kv_heads, num_heads=num_heads, scale=scale,
        input_layout="TND", block_size=block_size, actual_seq_lengths=prefill_lengths,
        actual_seq_lengths_kv=prefill_lengths, sparse_mode=3,
    )
    if not cached_tokens:
        return prefill_output, lse
    # The caller consumes only the attention output, not a combined LSE.
    return torch.cat((cached_output, prefill_output), dim=0), None
