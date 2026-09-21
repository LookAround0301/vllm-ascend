# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Publish DS V4 logical-cache maps over the existing O-MoE byte arena.

The scheduler owns allocation handles and expert retirement. The worker applies
and validates its ExpertCacheDelta before calling ``install`` in execute_model.
This module publishes the authorized maps; its final stream fence completes
publication before the worker activates prefetch and runs the model.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from types import MappingProxyType
from typing import TYPE_CHECKING, TypedDict

import torch

from vllm_ascend.core.dsv4_cache_pages import BACKEND_INT32_MAX, DSV4CachePage

if TYPE_CHECKING:
    from vllm_ascend.core.hierarchical_cache_manager import ByteSpan, DSV4CacheUpdate
    from vllm_ascend.core.hierarchical_memory import HierarchicalMemoryLayout


class _CacheMapTransaction(TypedDict):
    """Keep submitted device storage alive if publication fails."""

    arena: torch.Tensor
    device_tensors: list[torch.Tensor]


class DSV4CacheMaps:
    """Fail-closed logical-to-backend page publication within one model step.

    Each cache name has a separate int32 device map, but no separate payload
    arena or worker-side ownership pool. The scheduler's allocator owns cache
    and expert pages; retired experts become reusable only after every worker
    completes the preceding model step. This class publishes those grants and
    checks their geometry and zeroing before submitting device operations.

    A device failure can leave partially cleared bytes or partially published
    maps. The object is then permanently poisoned and pins both the submitted
    arena and temporary tensors; callers must abort instead of freeing memory
    or pretending that rolling back CPU metadata rolled back device accesses.
    """

    def __init__(
        self, layout: HierarchicalMemoryLayout, device: torch.device | str,
        *, synchronize: Callable[[], None] | None = None,
    ):
        self._arena_nbytes = layout.nbytes
        self._expert_page_bytes = layout.expert_page_bytes
        self._null_nbytes = layout.huge_page_bytes
        self._num_blocks = layout.num_blocks
        if self._num_blocks > BACKEND_INT32_MAX:
            raise ValueError("DS V4 logical map capacity exceeds the signed INT32 limit")
        if self._null_nbytes >= self._arena_nbytes or self._null_nbytes % self._expert_page_bytes:
            raise ValueError("The reserved null Huge must contain complete L2 pages inside the arena")
        # Layout construction has already checked schema geometry and L2 fit.
        schemas = layout.cache_schemas
        self._schemas_by_name = {page.name: page for page in schemas}
        # One compact metadata allocation, with disjoint rows per cache. This
        # is the planner-budgeted index_map_nbytes, not another KV/state arena.
        self._index_map_storage = torch.zeros(
            (len(schemas), self._num_blocks), dtype=torch.int32, device=device
        )
        self._index_maps = {page.name: self._index_map_storage[row] for row, page in enumerate(schemas)}
        self.device = self._index_map_storage.device
        self._synchronize = synchronize
        self._failed = False
        self._pending_state: _CacheMapTransaction | None = None

    @property
    def schemas_by_name(self) -> Mapping[str, DSV4CachePage]:
        return MappingProxyType(self._schemas_by_name)

    @property
    def index_maps(self) -> Mapping[str, torch.Tensor]:
        self._check_usable()
        return MappingProxyType(self._index_maps)

    def index_map_group(self, names: Sequence[str]) -> torch.Tensor:
        """View adjacent cache rows without copying the persistent index maps."""
        self._check_usable()
        maps = [self._index_maps[name] for name in names]
        first_row = maps[0].storage_offset() // self._num_blocks
        if all(index_map.storage_offset() == (first_row + row) * self._num_blocks
               for row, index_map in enumerate(maps)):
            return self._index_map_storage.narrow(0, first_row, len(maps))
        # A backend may split/reorder a group; retain its requested row order.
        return torch.stack(maps)

    def _check_usable(self) -> None:
        if self._failed:
            raise RuntimeError("DS V4 cache-map publication failed; this worker must abort")
        if self._pending_state is not None:
            raise RuntimeError("DS V4 cache-map publication is already in progress")

    def _prepare(
        self, updates: Sequence[DSV4CacheUpdate], zero_spans: Sequence[ByteSpan],
    ) -> tuple[dict[str, tuple[list[int], list[int]]], set[ByteSpan]] | None:
        """Validate this step's publication before any device writes."""
        if not updates and not zero_spans:
            return None
        seen = set()
        required_zeros = set()
        by_name: dict[str, tuple[list[int], list[int]]] = {}
        for name, logical_id, offset in updates:
            if not 0 < logical_id < self._num_blocks:
                raise ValueError("Cache logical ID must be nonzero and inside the logical map capacity")
            key = (name, logical_id)
            if key in seen:
                raise ValueError("Duplicate cache-map update key")
            seen.add(key)
            if offset < 0:
                raise ValueError("Cache-map offsets must be nonnegative")
            backend_id = 0
            if offset:
                page = self._schemas_by_name[name]
                if offset < self._null_nbytes:
                    raise ValueError("A real cache page cannot use the reserved null Huge")
                # Schema geometry and the native INT32 view bound were checked
                # at construction. Only this grant's extent and L2 packing vary.
                local_offset = offset % self._expert_page_bytes
                if (offset - local_offset + self._expert_page_bytes > self._arena_nbytes
                        or offset + page.allocation_nbytes > self._arena_nbytes):
                    raise ValueError("The page must lie in a complete L2 inside the arena")
                if (local_offset % page.allocation_nbytes
                        or local_offset + page.allocation_nbytes > self._expert_page_bytes):
                    raise ValueError("The offset is not a complete page packed from its L2's offset zero")
                backend_id = offset // page.address_quantum_bytes
                span = (offset, page.allocation_nbytes)
                if span in required_zeros:
                    raise ValueError("Duplicate physical cache page grant")
                required_zeros.add(span)
            # Free+reuse in one step has one final update from the manager.
            logical_ids, backend_ids = by_name.setdefault(name, ([], []))
            logical_ids.append(logical_id)
            backend_ids.append(backend_id)
        if set(zero_spans) != required_zeros:
            raise ValueError("Cache zero grants must exactly match all nonzero cache-map updates")
        # The common allocator already excludes retained cache/expert pages.
        # Rebuilding and sorting all retained mappings here duplicates its
        # ownership state and makes every decode step grow with total KV use.
        return by_name, required_zeros

    def install(
        self, arena: torch.Tensor, updates: Sequence[DSV4CacheUpdate], zero_spans: Sequence[ByteSpan],
    ) -> None:
        """Clear exact grants and publish one globally ordered transaction.

        Geometry and exact zero grants are checked before the first clear.
        Ownership comes from the scheduler's common allocator, after expert
        retirement completes on every rank. Device failures retain the pending
        tensors and poison the publisher; a partial write is never rolled back.
        """
        self._check_usable()
        prepared = self._prepare(updates, zero_spans)
        if prepared is None:
            return
        by_name, zeros = prepared
        pending: _CacheMapTransaction = {
            "arena": arena,
            "device_tensors": [],
        }
        self._pending_state = pending
        try:
            # Merge only adjacent, already validated grants. Gaps and L2
            # tails remain untouched; never clear a virtual Indexer view.
            zero_ranges: list[ByteSpan] = []
            for offset, nbytes in sorted(zeros):
                if zero_ranges and zero_ranges[-1][0] + zero_ranges[-1][1] == offset:
                    start, size = zero_ranges[-1]
                    zero_ranges[-1] = (start, size + nbytes)
                else:
                    zero_ranges.append((offset, nbytes))
            for offset, nbytes in zero_ranges:
                arena.narrow(0, offset, nbytes).zero_()

            # Cache maps are disjoint rows of the same metadata allocation.
            # storage_offset is in int32 elements, matching flattened indices.
            flat_indices: list[int] = []
            flat_values: list[int] = []
            for name, (logical_ids, backend_ids) in by_name.items():
                base = self._index_maps[name].storage_offset()
                flat_indices.extend(base + logical_id for logical_id in logical_ids)
                flat_values.extend(backend_ids)
            if flat_indices:
                indices = torch.tensor(flat_indices, dtype=torch.int64, device=self.device)
                values = torch.tensor(flat_values, dtype=torch.int32, device=self.device)
                pending["device_tensors"].extend((indices, values))
                self._index_map_storage.view(-1).index_copy_(0, indices, values)
            if self._synchronize is not None:
                self._synchronize()
            elif self.device.type == "npu":
                torch.npu.current_stream(self.device).synchronize()
        except BaseException:
            self._failed = True
            raise
        # The stream fence releases temporary publication tensors and arena references.
        self._pending_state = None
