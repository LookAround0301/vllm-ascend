# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Byte-addressed ownership for Huge -> Expert/SSM -> KV/Conv/SSM pages.

Ownership bookkeeping is independent of device state. One instance owns one
physical byte arena (or several identically indexed arenas).
An expert page is one projection shard, not an entire fused expert payload.

Backend indices describe typed physical locations. Local-L2 page indices are
NOT affine byte offsets: consumers must decode them with the same geometry as
``offset_bytes``. Conv indices use ``conv_address_bytes`` as their stride;
registered local types may quantize byte offsets for their backend separately.
Allocation handles stay on the control plane and identify live allocations by
object identity. Pass only their indices to consumers of that type's page codec.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import islice
from math import gcd, lcm


@dataclass(frozen=True)
class BlockHandle:
    """One live typed allocation; ``index`` is its stable typed page index."""

    kind: str
    index: int

    @property
    def block_id(self) -> int:
        return self.index


NULL_BLOCK = BlockHandle("null", -1)
BLOCK_KINDS = ("huge", "expert", "ssm", "kv", "conv")


@dataclass(frozen=True)
class LocalPageGeometry:
    """Complete pages packed from offset zero of each expert L2.

    ``allocation_nbytes`` is the whole physical ownership extent, not a
    backend address unit. ``alignment_bytes`` constrains every page start;
    it neither rounds up the payload nor introduces a leading gap. A parent
    contains floor(L2 bytes / allocation_nbytes) pages and owns the tail
    until its final child is released. These pages do not enlarge Huge LCM.
    """

    allocation_nbytes: int
    alignment_bytes: int = 1

    def __post_init__(self) -> None:
        if type(self.allocation_nbytes) is not int or type(self.alignment_bytes) is not int:
            raise ValueError("Local page size and alignment must be integers")
        if self.allocation_nbytes <= 0 or self.alignment_bytes <= 0:
            raise ValueError("Local page size and alignment must be positive")
        if self.allocation_nbytes % self.alignment_bytes:
            raise ValueError("Local page bytes must preserve alignment at every packed page start")


class AllocationError(MemoryError):
    """A reservation does not fit the current hierarchical ownership."""


@dataclass
class _L2Block:
    huge_id: int
    kind: str
    index: int
    split_kind: str | None = None
    allocated_children: int = 0


@dataclass(frozen=True)
class _CountPlan:
    huge_splits: tuple[tuple[str, int], ...]
    parent_splits: tuple[tuple[str, str, int], ...]


class HierarchicalBlockPool:
    """Hierarchical allocator with greedy typed admission.

    ``huge_page_bytes`` is lcm(expert, SSM), or expert bytes without SSM.
    SSM is L3 precisely when expert bytes are divisible by SSM bytes.
    Otherwise Huge pages split into either expert or SSM L2 pages. An L2
    page is either allocated whole or split into a single child type.
    Greedy count planning does not retry alternative Huge modes, so mixed
    L2 types may be rejected even when a different packing would fit.

    KV must tile every L2 page exactly. Conv pages contain ``conv_rows``
    consecutive rows, packed from each L2's start. Incomplete trailing rows
    remain owned by that L2 until its final child is released. Conv indices
    encode byte offsets in ``conv_address_bytes`` units; the worker view and
    native kernel use the same leading stride.

    Optional ``local_page_geometries`` add typed children of expert L2 pages.
    They pack from each L2's own offset zero, with no leading alignment gap.
    Their page index q decodes as (q // C) * L2_bytes + (q % C) * page_bytes,
    where C = floor(L2_bytes / page_bytes). Existing KV/Conv/SSM geometry is
    unchanged unless callers explicitly register new local page types.

    Budget bytes below a complete Huge page are explicitly unmanaged.
    All mutation is synchronous; callers must serialize access to a pool.
    """

    def __init__(
        self,
        *,
        total_bytes: int,
        expert_page_bytes: int,
        kv_page_bytes: int,
        ssm_page_bytes: int | None = None,
        conv_page_bytes: int | None = None,
        conv_rows: int = 1,
        alignment_bytes: int = 1,
        local_page_geometries: Mapping[str, LocalPageGeometry] | None = None,
    ) -> None:
        if total_bytes < 0:
            raise ValueError("total_bytes must be nonnegative")
        if alignment_bytes <= 0 or conv_rows <= 0:
            raise ValueError("alignment_bytes and conv_rows must be positive")

        self.total_bytes = total_bytes
        self.alignment_bytes = alignment_bytes
        self.expert_page_bytes = expert_page_bytes
        self.kv_page_bytes = kv_page_bytes
        self.ssm_page_bytes = ssm_page_bytes
        self.conv_page_bytes = conv_page_bytes
        self.conv_rows = conv_rows
        for name, size in (
            ("expert_page_bytes", expert_page_bytes),
            ("kv_page_bytes", kv_page_bytes),
            ("ssm_page_bytes", ssm_page_bytes),
            ("conv_page_bytes", conv_page_bytes),
        ):
            if size is not None:
                if size <= 0:
                    raise ValueError(f"{name} must be positive")
                if size % alignment_bytes:
                    raise ValueError(f"{name} must be divisible by alignment_bytes")
        self.huge_page_bytes = lcm(expert_page_bytes, ssm_page_bytes) if ssm_page_bytes else expert_page_bytes

        self.ssm_is_l3 = ssm_page_bytes is not None and expert_page_bytes % ssm_page_bytes == 0
        self.num_huge_blocks = total_bytes // self.huge_page_bytes
        self.managed_bytes = self.num_huge_blocks * self.huge_page_bytes
        self.unmanaged_bytes = total_bytes - self.managed_bytes
        self.num_expert_per_huge = self.huge_page_bytes // expert_page_bytes
        self.num_ssm_per_huge = self.huge_page_bytes // ssm_page_bytes if ssm_page_bytes else 0
        self.conv_row_bytes = 0
        if conv_page_bytes is not None:
            if conv_page_bytes % conv_rows:
                raise ValueError("conv_page_bytes must contain an integral number of conv_rows")
            self.conv_row_bytes = conv_page_bytes // conv_rows
            if self.conv_row_bytes % alignment_bytes:
                raise ValueError("Conv rows must satisfy alignment_bytes")
        self._page_bytes = {
            "huge": self.huge_page_bytes,
            "expert": expert_page_bytes,
            "kv": kv_page_bytes,
            "ssm": ssm_page_bytes or 0,
            "conv": conv_page_bytes or 0,
        }
        
        self._l2_kinds = ("expert", "ssm") if ssm_page_bytes and not self.ssm_is_l3 else ("expert",)
        # Largest address unit shared by native rows and both possible L2 starts.
        # This encodes indices only; allocations still own a full Conv state.
        self.conv_address_bytes = gcd(self.conv_row_bytes, expert_page_bytes, ssm_page_bytes or expert_page_bytes)
        for kind in self._l2_kinds:
            parent_bytes = self._page_bytes[kind]
            if parent_bytes % kv_page_bytes:
                raise ValueError(f"kv_page_bytes must divide {kind} L2 page bytes exactly")
        if conv_page_bytes is not None and conv_page_bytes > max(self._page_bytes[kind] for kind in self._l2_kinds):
            raise ValueError("A Conv page must fit inside an L2 page")
        if local_page_geometries is not None and not isinstance(local_page_geometries, Mapping):
            raise ValueError("local_page_geometries must map new type names to LocalPageGeometry")
        local_pages = dict(local_page_geometries or {})
        for kind, geometry in local_pages.items():
            if not isinstance(kind, str) or not kind or kind in self._page_bytes:
                raise ValueError("Local page types must have nonempty names distinct from existing block kinds")
            if not isinstance(geometry, LocalPageGeometry):
                raise ValueError("Local page types require LocalPageGeometry values")
            if geometry.allocation_nbytes > expert_page_bytes:
                raise ValueError("A complete local page must fit inside an expert L2")
            if geometry.allocation_nbytes % alignment_bytes or expert_page_bytes % geometry.alignment_bytes:
                raise ValueError("Packed local pages and expert L2 starts must preserve their declared alignment")
            self._page_bytes[kind] = geometry.allocation_nbytes
        # Geometry is immutable; only ownership changes during allocation.
        self._local_pages = local_pages
        self._block_kinds = BLOCK_KINDS + tuple(sorted(local_pages))
        self._child_kinds = ("conv", "kv") + (("ssm",) if self.ssm_is_l3 else ()) + tuple(sorted(local_pages))
        self._free: dict[str, set[int]] = {kind: set[int]() for kind in self._block_kinds}
        self._free["huge"] = set(range(self.num_huge_blocks))
        self._huge_modes: dict[int, str] = {}
        # Typed keys are essential: expert index 1 and SSM index 1 may
        # simultaneously describe different physical L2 locations.
        self._l2: dict[tuple[str, int], _L2Block] = {}
        self._allocated_blocks: dict[tuple[str, int], BlockHandle] = {}
        self.null_block = NULL_BLOCK

    def _parent_indices(self, huge_id: int, kind: str) -> range:
        count = self.huge_page_bytes // self._page_bytes[kind]
        return range(huge_id * count, (huge_id + 1) * count)

    def _child_indices(self, parent_kind: str, parent_index: int, child: str) -> range:
        offset = parent_index * self._page_bytes[parent_kind]
        count = self._page_bytes[parent_kind] // self._page_bytes[child]
        if child in self._local_pages:
            if parent_kind != "expert":
                return range(0)
            # q enumerates actual complete local pages, not page-sized units
            # across the whole arena. The skipped tail is not an extra slot.
            return range(parent_index * count, (parent_index + 1) * count)
        if child == "conv":
            start = offset // self.conv_address_bytes
            step = self._page_bytes[child] // self.conv_address_bytes
            return range(start, start + count * step, step)
        start = offset // self._page_bytes[child]
        return range(start, start + count)

    def _parent_capacity(self, parent_kind: str, kind: str) -> int:
        # Capacity of one whole free L2; an unsupported parent contributes zero.
        if kind in self._l2_kinds:
            return int(parent_kind == kind)
        if (kind == "ssm" or kind in self._local_pages) and parent_kind != "expert":
            return 0
        return self._page_bytes[parent_kind] // self._page_bytes[kind]

    def _plan_counts(self, requests: Mapping[str, int], extra_free_expert_blocks: int = 0) -> _CountPlan | None:
        """Plan typed splits by count; choose physical indices only on allocation."""
        from vllm.utils.math_utils import cdiv

        free_huge = len(self._free["huge"]) - requests.get("huge", 0)
        if free_huge < 0:
            return None
        free_counts = {kind: len(self._free[kind]) for kind in self._l2_kinds + self._child_kinds}
        free_counts["expert"] += extra_free_expert_blocks
        free_parents = {
            kind: max(0, free_counts[kind] - requests.get(kind, 0)) for kind in self._l2_kinds
        }
        demands = {
            kind: max(0, requests.get(kind, 0) - free_counts[kind])
            for kind in self._l2_kinds + self._child_kinds
        }
        huge_splits = []
        parent_splits = []
        # Reserve existing typed pages first, then use surplus L2s for L3s.
        for kind in self._child_kinds:
            count = demands[kind]
            if not count:
                continue
            for parent_kind in self._l2_kinds:
                capacity = self._parent_capacity(parent_kind, kind)
                if not capacity:
                    continue
                parent_count = min(free_parents[parent_kind], cdiv(count, capacity))
                if parent_count:
                    parent_splits.append((parent_kind, kind, parent_count))
                    free_parents[parent_kind] -= parent_count
                    count = max(0, count - parent_count * capacity)
            demands[kind] = count

        # New Huges keep one L2 kind; each split L2 keeps one L3 kind.
        for kind in self._child_kinds:
            count = demands[kind]
            while count:
                parent_kind = next(
                    (parent for parent, available in free_parents.items()
                     if available and self._parent_capacity(parent, kind)), None
                )
                if parent_kind is None:
                    parent_kind = max(
                        self._l2_kinds,
                        key=lambda parent: self._parent_capacity(parent, kind)
                        * (self.huge_page_bytes // self._page_bytes[parent]),
                    )
                    capacity = self._parent_capacity(parent_kind, kind)
                    if not capacity:
                        return None
                    pages_per_huge = self.huge_page_bytes // self._page_bytes[parent_kind]
                    huge_count = cdiv(cdiv(count, capacity), pages_per_huge)
                    if huge_count > free_huge:
                        return None
                    free_huge -= huge_count
                    huge_splits.append((parent_kind, huge_count))
                    free_parents[parent_kind] += huge_count * pages_per_huge
                capacity = self._parent_capacity(parent_kind, kind)
                parent_count = min(free_parents[parent_kind], cdiv(count, capacity))
                parent_splits.append((parent_kind, kind, parent_count))
                free_parents[parent_kind] -= parent_count
                count = max(0, count - parent_count * capacity)

        # Whole L2 requests reuse siblings left by the L3 splits above.
        for kind in self._l2_kinds:
            count = demands[kind] - free_parents[kind]
            if count > 0:
                huge_count = cdiv(count, self.huge_page_bytes // self._page_bytes[kind])
                if huge_count > free_huge:
                    return None
                free_huge -= huge_count
                huge_splits.append((kind, huge_count))
        return _CountPlan(tuple(huge_splits), tuple(parent_splits))

    def get_required_expert_blocks_for_allocation(self, requests: Mapping[str, int]) -> int:
        """Find the extra expert L2 count needed, as in O-MoE's block pool."""
        if self.can_reserve(requests):
            return 0
        low, high = 1, self.managed_bytes // self.expert_page_bytes
        required = 0
        while low <= high:
            count = (low + high) // 2
            if self._plan_counts(requests, extra_free_expert_blocks=count) is not None:
                required = count
                high = count - 1
            else:
                low = count + 1
        return required

    def can_reserve(self, requests: Mapping[str, int]) -> bool:
        """Check typed demand without changing ownership or issuing handles."""
        if all(count <= len(self._free[kind]) for kind, count in requests.items()):
            return True
        return self._plan_counts(requests) is not None

    def reserve(self, requests: Mapping[str, int]) -> dict[str, list[BlockHandle]]:
        """Check the whole request, then allocate its typed blocks in place.

        Like BlockPool.get_new_blocks, capacity is checked before mutation.
        Multiple page types require a split plan instead of one free count.
        No ownership changes if the plan does not fit.
        """
        result = {kind: [] for kind in requests}
        if all(count <= len(self._free[kind]) for kind, count in requests.items()):
            for kind, count in requests.items():
                if not count:
                    continue
                for index in list(islice(self._free[kind], count)):
                    result[kind].append(self._claim(kind, index))
            return result
        plan = self._plan_counts(requests)
        if plan is None:
            raise AllocationError(f"Cannot reserve {requests}")
        # Claim existing pages before splits, including whole L2 requests.
        for kind, count in requests.items():
            for index in list(islice(self._free[kind], min(count, len(self._free[kind])))):
                result[kind].append(self._claim(kind, index))
        for kind, count in plan.huge_splits:
            for huge_id in list(islice(self._free["huge"], count)):
                self._split_huge(huge_id, kind)
        for parent_kind, child, count in plan.parent_splits:
            for index in list(islice(self._free[parent_kind], count)):
                self._split_l2(parent_kind, index, child)
                remaining = requests[child] - len(result[child])
                for child_index in islice(self._child_indices(parent_kind, index, child), remaining):
                    result[child].append(self._claim(child, child_index))
        for kind, count in requests.items():
            for index in list(islice(self._free[kind], count - len(result[kind]))):
                result[kind].append(self._claim(kind, index))
        return result

    def reserve_at(self, kind: str, indices: Sequence[int]) -> list[BlockHandle]:
        """Claim already-free locations, used to protect null Huge0 at startup.

        This does not split parents or search alternative layouts.
        """
        indices = tuple(indices)
        if len(set(indices)) != len(indices):
            raise ValueError("Duplicate allocation index")
        if not all(index in self._free[kind] for index in indices):
            raise AllocationError(f"Requested {kind} indices are not free")
        return [self._claim(kind, index) for index in indices]

    def _split_huge(self, huge_id: int, kind: str) -> None:
        self._free["huge"].remove(huge_id)
        self._huge_modes[huge_id] = kind
        for index in self._parent_indices(huge_id, kind):
            self._l2[kind, index] = _L2Block(huge_id, kind, index)
            self._free[kind].add(index)

    def _split_l2(self, kind: str, index: int, child: str) -> None:
        self._free[kind].remove(index)
        parent = self._l2[kind, index]
        parent.split_kind = child
        self._free[child].update(self._child_indices(parent.kind, parent.index, parent.split_kind))

    def _offset(self, kind: str, index: int) -> int:
        if kind == "conv":
            return index * self.conv_address_bytes
        if kind in self._local_pages:
            # The same typed ordinal can name different bytes in different
            # geometries. In particular q * page_bytes would silently point
            # into the previous L2's tail at the first nondivisible boundary.
            page_bytes = self._page_bytes[kind]
            pages_per_l2 = self.expert_page_bytes // page_bytes
            parent_index, local_index = divmod(index, pages_per_l2)
            return parent_index * self.expert_page_bytes + local_index * page_bytes
        return index * self._page_bytes[kind]

    def _parent(self, kind: str, index: int) -> _L2Block:
        offset = self._offset(kind, index)
        huge_id = offset // self.huge_page_bytes
        parent_kind = self._huge_modes[huge_id]
        return self._l2[parent_kind, offset // self._page_bytes[parent_kind]]

    def _claim(self, kind: str, index: int) -> BlockHandle:
        self._free[kind].remove(index)
        handle = BlockHandle(kind, index)
        self._allocated_blocks[kind, index] = handle
        if kind != "huge" and kind not in self._l2_kinds:
            self._parent(kind, index).allocated_children += 1
        return handle

    def _require_allocated(self, handle: BlockHandle) -> None:
        if not isinstance(handle, BlockHandle) or self._allocated_blocks.get((handle.kind, handle.index)) is not handle:
            raise ValueError("Unknown, foreign, null or stale allocation handle")

    def free(self, handles: Iterable[BlockHandle]) -> None:
        """Release handles atomically; reject duplicate, stale and foreign frees.

        Freeing the last L3 child first reclaims its L2. Only when every L2
        is free does its Huge page become available to change type.
        """
        handles = tuple(handles)
        for handle in handles:
            self._require_allocated(handle)
        if len(set(handles)) != len(handles):
            raise ValueError("The same allocation occurs more than once in a free batch")
        for handle in handles:
            del self._allocated_blocks[handle.kind, handle.index]
            self._free[handle.kind].add(handle.index)
            if handle.kind == "huge":
                continue
            if handle.kind in self._l2_kinds:
                self._try_reclaim_huge(self._l2[handle.kind, handle.index].huge_id)
                continue
            parent = self._parent(handle.kind, handle.index)
            parent.allocated_children -= 1
            if parent.allocated_children == 0:
                self._free[handle.kind].difference_update(
                    self._child_indices(parent.kind, parent.index, parent.split_kind)
                )
                parent.split_kind = None
                self._free[parent.kind].add(parent.index)
                self._try_reclaim_huge(parent.huge_id)

    def _try_reclaim_huge(self, huge_id: int) -> None:
        kind = self._huge_modes[huge_id]
        indices = self._parent_indices(huge_id, kind)
        if all(index in self._free[kind] for index in indices):
            for index in indices:
                self._free[kind].remove(index)
                del self._l2[kind, index]
            del self._huge_modes[huge_id]
            self._free["huge"].add(huge_id)

    def offset_bytes(self, handle: BlockHandle) -> int:
        """Resolve a live handle to its physical byte offset; reject the sentinel."""
        self._require_allocated(handle)
        return self._offset(handle.kind, handle.index)

    def allocation_bytes(self, handle: BlockHandle) -> int:
        self._require_allocated(handle)
        return self._page_bytes[handle.kind]

    def available(self, kind: str) -> int:
        """Exact capacity for this type alone, respecting current fragmentation."""
        if not self._page_bytes[kind]:
            return 0
        if kind == "huge":
            return len(self._free["huge"])
        if kind in self._l2_kinds:
            return len(self._free[kind]) + len(self._free["huge"]) * self.huge_page_bytes // self._page_bytes[kind]
        parents = ("expert",) if kind == "ssm" or kind in self._local_pages else self._l2_kinds
        from_parents = sum(
            self._parent_capacity(parent, kind) * len(self._free[parent]) for parent in parents
        )
        from_huges = len(self._free["huge"]) * max(
            self._parent_capacity(parent, kind) * (self.huge_page_bytes // self._page_bytes[parent])
            for parent in parents
        )
        return len(self._free[kind]) + from_parents + from_huges

    def get_num_free_bytes(self) -> int:
        """Count free pages by type without scanning allocated blocks."""
        return sum(len(self._free[kind]) * self._page_bytes[kind] for kind in self._block_kinds)

    def usage(self) -> dict[str, int | dict[str, int]]:
        """Byte conservation: managed = allocated + free + tail bytes.

        ``free_bytes`` includes free children pinned inside partially live
        parents; it does not imply that an equally sized mixed request fits.
        ``reserved_bytes`` counts all Huge pages committed to a live type.
        """
        allocated = {kind: 0 for kind in self._block_kinds}
        for kind, _ in self._allocated_blocks:
            allocated[kind] += 1
        allocated_bytes = sum(allocated[kind] * self._page_bytes[kind] for kind in self._block_kinds)
        free_bytes = self.get_num_free_bytes()
        tail_bytes = 0
        for parent in self._l2.values():
            if parent.split_kind is not None:
                tail_bytes += self._page_bytes[parent.kind] % self._page_bytes[parent.split_kind]
        return {
            "total_bytes": self.total_bytes,
            "managed_bytes": self.managed_bytes,
            "unmanaged_bytes": self.unmanaged_bytes,
            "allocated_bytes": allocated_bytes,
            "free_bytes": free_bytes,
            "head_bytes": 0,
            "tail_bytes": tail_bytes,
            "padding_bytes": tail_bytes,
            "reserved_bytes": self.managed_bytes - len(self._free["huge"]) * self.huge_page_bytes,
            "free_huge_blocks": len(self._free["huge"]),
            "allocated_by_type": allocated,
        }
