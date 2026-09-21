# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Per-engine typed page admission beneath vLLM's request/block tables.

Ordinary full-attention KV IDs index physical K/V pairs. Mamba IDs are logical
blocks, mapped to separate Conv and SSM allocations at the worker boundary.
DS V4 retains its native logical groups: each block maps every cache in that
group to its own typed physical page, including main and Indexer KV. These
namespaces remain distinct; equal integer IDs do not imply equal ownership.
Only synchronous, no-prefix-cache, non-speculative full attention/GDN and
validated DS V4 schemas are supported. Native managers retain their request
lifetime bookkeeping.
"""

from collections import Counter, deque
from heapq import heappop, heappush
from types import MethodType
from typing import TypeAlias

from vllm.v1.core.kv_cache_utils import KVCacheBlock
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec

from vllm_ascend.core.hierarchical_block_pool import AllocationError, HierarchicalBlockPool

ByteSpan: TypeAlias = tuple[int, int]  # (byte offset, byte length)
DSV4CacheUpdate: TypeAlias = tuple[str, int, int]  # (cache name, logical block ID, byte offset)
GDNStateUpdate: TypeAlias = tuple[int, int, int]  # (logical block ID, Conv ID, SSM ID)
CacheMapUpdate: TypeAlias = DSV4CacheUpdate | GDNStateUpdate


class CacheGroupBlockPool:
    def __init__(self, cache_manager, kind, schemas=()):
        self.cache_manager = cache_manager
        self.kind = kind
        self.schemas = tuple(schemas)
        self.null_block = cache_manager.null_block
        self.num_gpu_blocks = cache_manager.num_gpu_blocks
        self.hash_block_size = cache_manager.hash_block_size
        self.pending_blocks = deque()

    def get_new_blocks(self, num_blocks):
        if num_blocks < 0 or num_blocks > len(self.pending_blocks):
            raise RuntimeError("Hierarchical allocation exceeded its atomic admission grant")
        return [self.pending_blocks.popleft() for _ in range(num_blocks)]

    def free_blocks(self, blocks):
        self.cache_manager.free_blocks(blocks)


class HierarchicalCacheManager:
    """Attach once, before requests exist; never replace an OFF engine's pool."""

    def __init__(self, manager, layout):
        if manager.enable_caching or manager.use_eagle or manager.watermark_blocks:
            raise ValueError("Hierarchical pages require no APC, speculative decoding, or KV watermark")
        self.manager = manager
        self.allocator = HierarchicalBlockPool(**layout.allocator_kwargs())
        self.layout = layout
        # All native cache views use byte zero as null; protect the whole Huge.
        self.null_huge_blocks = self.allocator.reserve_at("huge", (0,))
        self.required_expert_shrink_blocks = 0
        self.num_gpu_blocks = layout.num_blocks
        self.hash_block_size = manager.block_pool.hash_block_size
        self.null_block = KVCacheBlock(0, is_null=True)
        self.metrics_collector = None
        self.enable_caching = False
        self.allocated_blocks = {}  # Object identity, not ambiguous typed integer IDs.
        self.free_logical_block_ids = []
        self.next_logical_block_id = 1
        self.pending_cache_map_updates: dict[int | tuple[str, int], CacheMapUpdate] = {}
        self.pending_zero_spans: list[ByteSpan] = []
        self.cache_schemas = getattr(layout, "cache_schemas", ())
        self.block_schemas = {}
        self.group_pools = []
        if self.cache_schemas:
            from vllm_ascend.core.dsv4_cache_pages import validate_scheduler_groups

            validate_scheduler_groups(self.cache_schemas, manager.kv_cache_config.kv_cache_groups)
        for group_id, native_kv_manager in enumerate(manager.coordinator.single_type_managers):
            if native_kv_manager.req_to_blocks or native_kv_manager._partial_hit_reqs:
                raise RuntimeError("Attach hierarchical pages before admitting any request")
            cache_spec = native_kv_manager.kv_cache_spec
            schemas = tuple(page_schema for page_schema in self.cache_schemas if page_schema.group_id == group_id)
            if self.cache_schemas:
                if not schemas:
                    raise ValueError("Every DS native group must have a complete physical block schema")
                kind = "dsv4"
            elif isinstance(cache_spec, MambaSpec):
                if cache_spec.mamba_cache_mode != "none" or cache_spec.num_speculative_blocks:
                    raise ValueError("Hierarchical state blocks require Mamba cache mode none and no MTP")
                kind = "state"
            elif type(cache_spec) is FullAttentionSpec:
                kind = "kv"
            else:
                raise ValueError(f"Unsupported hierarchical cache manager spec: {type(cache_spec).__name__}")
            group_pool = CacheGroupBlockPool(self, kind, schemas)
            self.group_pools.append(group_pool)
            native_kv_manager.block_pool = group_pool
            native_kv_manager._null_block = self.null_block
            # Logical state blocks and paired KV IDs must never reach the
            # original page-zeroer. The RPC protocol zeroes actual byte grants.
            native_kv_manager._record_new_block_ids = False
        manager.block_pool = manager.coordinator.block_pool = self
        cache_manager = self

        def allocate_slots(bound_native_manager, *args, **kwargs):
            return cache_manager.allocate_slots(*args, **kwargs)

        manager.allocate_slots = MethodType(allocate_slots, manager)

    def get_required_block_counts(
        self, request_id, num_tokens, *, num_computed_tokens=0,
        num_tokens_main_model=None, apply_admission_cap=False,
    ):
        if self.cache_schemas:
            # Each original manager owns its compression/window mathematics.
            # Keep its count per group; a sum would lose the payload types.
            group_block_counts = [
                native_kv_manager.get_num_blocks_to_allocate(
                    request_id=request_id,
                    num_tokens=num_tokens,
                    new_computed_blocks=(),
                    total_computed_tokens=num_computed_tokens,
                    num_local_computed_tokens=num_computed_tokens,
                    num_tokens_main_model=num_tokens if num_tokens_main_model is None else num_tokens_main_model,
                    apply_admission_cap=apply_admission_cap,
                )
                for native_kv_manager in self.manager.coordinator.single_type_managers
            ]
            if any(type(num_blocks) is not int or num_blocks < 0 for num_blocks in group_block_counts):
                raise RuntimeError("Native DS cache admission returned an invalid group count")
            return group_block_counts
        # The supported native full-attention and Mamba-none allocation paths
        # both use ceil(tokens / block_size) - current blocks. Counting here
        # avoids summing incompatible byte units (or counting external-state
        # copies when no KV connector is enabled).
        return [
            max(
                0,
                (num_tokens + native_kv_manager.block_size - 1) // native_kv_manager.block_size
                - len(native_kv_manager.req_to_blocks.get(request_id, ())),
            )
            for native_kv_manager in self.manager.coordinator.single_type_managers
        ]

    def get_required_page_counts(self, group_block_counts):
        if len(group_block_counts) != len(self.group_pools):
            raise ValueError("Typed demand must include every logical group exactly once")
        if self.cache_schemas:
            required_pages = Counter()
            for num_blocks, group_pool in zip(group_block_counts, self.group_pools):
                for page_schema in group_pool.schemas:
                    required_pages[page_schema.kind] += num_blocks
            return dict(required_pages)
        num_kv_blocks = sum(
            num_blocks for num_blocks, group_pool in zip(group_block_counts, self.group_pools)
            if group_pool.kind == "kv"
        )
        num_state_blocks = sum(
            num_blocks for num_blocks, group_pool in zip(group_block_counts, self.group_pools)
            if group_pool.kind == "state"
        )
        return {"kv": num_kv_blocks, "conv": num_state_blocks, "ssm": num_state_blocks}

    def has_available_logical_block_ids(self, num_blocks):
        return num_blocks <= len(self.free_logical_block_ids) + self.num_gpu_blocks - self.next_logical_block_id

    def can_allocate_pages(self, required_pages, *, num_logical_blocks=None):
        num_blocks = required_pages.get("ssm", 0) if num_logical_blocks is None else num_logical_blocks
        if not self.has_available_logical_block_ids(num_blocks):
            return False
        if self.allocator.can_reserve(required_pages):
            return True
        # Defer expert shrink to this step's cache delta. Admission never
        # performs worker calls or reuses pages before execution completes.
        required_blocks = self.allocator.get_required_expert_blocks_for_allocation(required_pages)
        self.required_expert_shrink_blocks = max(self.required_expert_shrink_blocks, required_blocks)
        return False

    def create_logical_block(self, kind, physical_blocks, schemas=()):
        if kind == "kv":
            block_id = physical_blocks[0].index
        else:
            if self.free_logical_block_ids:
                block_id = heappop(self.free_logical_block_ids)
            else:
                block_id = self.next_logical_block_id
                self.next_logical_block_id += 1
            if kind == "dsv4":
                # The block is shared only logically. Each actual cache owns
                # its own page, and sends an absolute byte offset to workers.
                # Never multiply a tightly packed typed page ID by page bytes.
                for page_schema, physical_block in zip(schemas, physical_blocks, strict=True):
                    self.pending_cache_map_updates[(page_schema.name, block_id)] = (
                        page_schema.name, block_id, self.allocator.offset_bytes(physical_block)
                    )
            else:
                conv_block, ssm_block = physical_blocks
                self.pending_cache_map_updates[block_id] = (
                    block_id,
                    conv_block.index,
                    ssm_block.index,
                )
        if not 0 < block_id < self.num_gpu_blocks:
            raise RuntimeError("Hierarchical live allocation has an invalid/null backend ID")
        block = KVCacheBlock(block_id, ref_cnt=1)
        self.allocated_blocks[id(block)] = (block, kind, tuple(physical_blocks))
        if kind == "dsv4":
            self.block_schemas[id(block)] = tuple(schemas)
        self.pending_zero_spans.extend(
            (self.allocator.offset_bytes(physical_block), self.allocator.allocation_bytes(physical_block))
            for physical_block in physical_blocks
        )
        return block

    def allocate_slots(
        self,
        request,
        num_new_tokens,
        num_new_computed_tokens=0,
        new_computed_blocks=None,
        num_lookahead_tokens=0,
        num_external_computed_tokens=0,
        delay_cache_blocks=False,
        num_encoder_tokens=0,
        full_sequence_must_fit=False,
        reserved_blocks=0,
        has_scheduled_reqs=True,
    ):
        if (
            num_new_computed_tokens
            or num_external_computed_tokens
            or num_lookahead_tokens
            or num_encoder_tokens
            or reserved_blocks
            or delay_cache_blocks
            or (new_computed_blocks is not None and any(new_computed_blocks.blocks))
        ):
            raise ValueError("Hierarchical admission does not support prefix hits, KV transfer, MTP, or encoder KV")
        if num_new_tokens <= 0:
            raise ValueError("num_new_tokens must be positive")
        manager = self.manager
        num_computed_tokens = min(request.num_computed_tokens, manager.max_model_len)
        num_tokens = min(num_computed_tokens + num_new_tokens, manager.max_model_len)
        if full_sequence_must_fit:
            full_sequence_block_counts = self.get_required_block_counts(
                request.request_id,
                min(request.num_tokens, manager.max_model_len),
                num_computed_tokens=num_computed_tokens,
                apply_admission_cap=True,
            )
            full_sequence_page_counts = self.get_required_page_counts(full_sequence_block_counts)
            if not self.can_allocate_pages(
                full_sequence_page_counts,
                num_logical_blocks=sum(full_sequence_block_counts) if self.cache_schemas else None,
            ):
                return None
        manager.coordinator.remove_skipped_blocks(
            request.request_id,
            max(0, num_computed_tokens - request.num_in_flight_tokens),
            num_prompt_tokens=request.num_prompt_tokens,
        )
        group_block_counts = self.get_required_block_counts(
            request.request_id,
            num_tokens,
            num_computed_tokens=request.num_computed_tokens,
            num_tokens_main_model=num_computed_tokens + num_new_tokens,
        )
        required_pages = self.get_required_page_counts(group_block_counts)
        if not self.can_allocate_pages(
            required_pages, num_logical_blocks=sum(group_block_counts) if self.cache_schemas else None,
        ):
            return None
        try:
            allocated_pages = self.allocator.reserve(required_pages)
        except AllocationError:
            return None
        # reserve is all-or-nothing across all groups and both state types.
        # After this point an unexpected native-manager exception is fatal;
        # the engine retains ownership and stops this cache-delta sequence.
        page_iterators = {kind: iter(physical_blocks) for kind, physical_blocks in allocated_pages.items()}
        for num_blocks, group_pool in zip(group_block_counts, self.group_pools):
            if group_pool.pending_blocks:
                raise RuntimeError("Unconsumed hierarchical admission blocks")
            for _ in range(num_blocks):
                if group_pool.kind == "dsv4":
                    physical_blocks = tuple(
                        next(page_iterators[page_schema.kind]) for page_schema in group_pool.schemas
                    )
                elif group_pool.kind == "kv":
                    physical_blocks = (next(page_iterators["kv"]),)
                else:
                    physical_blocks = (next(page_iterators["conv"]), next(page_iterators["ssm"]))
                group_pool.pending_blocks.append(
                    self.create_logical_block(group_pool.kind, physical_blocks, group_pool.schemas)
                )
        result = manager.coordinator.allocate_new_blocks(
            request.request_id, num_tokens, num_computed_tokens + num_new_tokens, 0
        )
        if any(group_pool.pending_blocks for group_pool in self.group_pools):
            raise RuntimeError("Native cache manager did not consume its full typed grant")
        return manager.create_kv_cache_blocks(result)

    def free_blocks(self, blocks):
        allocation_records = []
        seen_block_objects = set()
        for block in blocks:
            if block is self.null_block:
                continue
            allocation_record = self.allocated_blocks.get(id(block))
            if (
                allocation_record is None or allocation_record[0] is not block
                or block.ref_cnt != 1 or id(block) in seen_block_objects
            ):
                raise RuntimeError("Invalid, shared, or duplicate hierarchical block release")
            seen_block_objects.add(id(block))
            allocation_records.append(allocation_record)
        released_spans = {
            (self.allocator.offset_bytes(physical_block), self.allocator.allocation_bytes(physical_block))
            for _, _, physical_blocks in allocation_records
            for physical_block in physical_blocks
        }
        self.allocator.free(
            physical_block for _, _, physical_blocks in allocation_records for physical_block in physical_blocks
        )
        # A request may be admitted and preempted in the same schedule call.
        # Cancel its not-yet-submitted zeroing before optional expert growth
        # borrows these now-free bytes; a later cache grant queues fresh zeros.
        self.pending_zero_spans = [span for span in self.pending_zero_spans if span not in released_spans]
        for block, kind, _ in allocation_records:
            del self.allocated_blocks[id(block)]
            block.ref_cnt = 0
            if kind == "state":
                self.pending_cache_map_updates[block.block_id] = (block.block_id, 0, 0)
                heappush(self.free_logical_block_ids, block.block_id)
            elif kind == "dsv4":
                for page_schema in self.block_schemas.pop(id(block)):
                    self.pending_cache_map_updates[(page_schema.name, block.block_id)] = (
                        page_schema.name, block.block_id, 0
                    )
                heappush(self.free_logical_block_ids, block.block_id)

    def take_worker_updates(self) -> tuple[list[CacheMapUpdate], list[ByteSpan]]:
        cache_map_updates, zero_spans = list(self.pending_cache_map_updates.values()), self.pending_zero_spans
        self.pending_cache_map_updates = {}
        self.pending_zero_spans = []
        return cache_map_updates, zero_spans

    def get_num_free_blocks(self):
        # Compatibility/metrics only: typed admission never uses this number.
        if self.cache_schemas:
            min_group_block_bytes = min(
                sum(page_schema.allocation_nbytes for page_schema in group_pool.schemas)
                for group_pool in self.group_pools
            )
            return min(
                self.allocator.get_num_free_bytes() // min_group_block_bytes,
                len(self.free_logical_block_ids) + self.num_gpu_blocks - self.next_logical_block_id,
            )
        return self.allocator.available("kv")

    def get_usage(self):
        return 1.0 - self.allocator.get_num_free_bytes() / self.layout.nbytes

    def take_events(self):
        return []

    def reset_prefix_cache(self):
        return True  # APC is disabled; there is no retained prefix storage.

    def evict_blocks(self, block_ids):
        if block_ids:
            raise ValueError("External block eviction is unsupported with typed hierarchical IDs")
