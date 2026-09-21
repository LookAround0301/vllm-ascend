# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Scheduler-side expert residency over the cache manager's shared page pool.

Like OMoE's ExpertManager, build one cache delta after request scheduling and
return retired pages only when model execution has completed on every worker.
"""

import heapq
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite
from typing import TYPE_CHECKING, TypeAlias
from vllm.utils.math_utils import cdiv

if TYPE_CHECKING:
    from vllm.config import VllmConfig

    from vllm_ascend.core.hierarchical_block_pool import BlockHandle
    from vllm_ascend.core.hierarchical_cache_manager import ByteSpan, CacheMapUpdate, HierarchicalCacheManager
    from vllm_ascend.expert_offload.expert_offload_manager import ExpertWorkerInfo
    from vllm_ascend.expert_offload.expert_prefetch import ExpertCacheSnapshot, ExpertSlotSnapshot
    from vllm_ascend.expert_offload.lrc_policy import ExpertPolicy

EXPERT_PROJECTION_PAGES = 3  # Gate, down, up; scales stay resident on the model.


ExpertSlotGrant: TypeAlias = tuple[int, tuple[int, int, int]]  # Slot ID and gate/down/up block IDs.


@dataclass
class ExpertCacheDelta:
    delta_id: int
    added_slots: list[ExpertSlotGrant]
    released_slots: tuple[int, ...]
    policy: "ExpertPolicy"
    cache_updates: "list[CacheMapUpdate]"
    zero_spans: "list[ByteSpan]"
    initial_resident_counts: tuple[int, ...] = ()


class ExpertManager:
    """Plan expert growth/shrink; the worker executes it with the model step."""

    MAX_GROWTH_PER_STEP = 5
    LOW_WATERMARK_EXPERT_PAGES = 2
    EXPAND_HEADROOM_EXPERTS = 3
    AUTO_CACHE_ROUND_MULTIPLE = 5  

    def __init__(
        self, vllm_config: "VllmConfig", cache_manager: "HierarchicalCacheManager",
        worker_info: "Sequence[ExpertWorkerInfo]",
    ) -> None:
        from vllm_ascend.ascend_config import ExpertOffloadConfig
        from vllm_ascend.expert_offload.config import expert_residency_limits
        from vllm_ascend.expert_offload.lrc_policy import CoordinatedLRCPolicy

        config = vllm_config
        self.worker_count = config.parallel_config.world_size
        self.cache_manager = cache_manager
        self.layout = cache_manager.layout
        self.allocator = cache_manager.allocator
        expected: ExpertWorkerInfo = {
            "kind": "hierarchical",
            "num_blocks": self.layout.num_blocks,
            "allocator": self.layout.allocator_wire_kwargs(),
            "expert_nbytes": self.layout.expert_extent_nbytes,
            "local_experts": self.layout.local_experts_per_layer,
        }
        if self.layout.cache_schemas:
            expected["cache_schemas"] = [page.to_dict() for page in self.layout.cache_schemas]
        if len(worker_info) != self.worker_count or any(info != expected for info in worker_info):
            raise ValueError("O-MoE scheduler/worker geometry differs")
        if self.layout.expert_span_pages != EXPERT_PROJECTION_PAGES:
            raise ValueError("O-MoE projection slots require three matrix pages")
        self.local_experts_per_layer = self.layout.local_experts_per_layer
        hf_config = config.model_config.hf_text_config
        self.layer_count = hf_config.num_hidden_layers
        topk = getattr(hf_config, "num_experts_per_tok", None) or getattr(hf_config, "num_experts_per_token", None)
        offload = ExpertOffloadConfig((config.additional_config or {}).get("expert_offload_config", {}))
        self.resident_floors, self.required_slots = expert_residency_limits(
            offload.offload_expert_limit, self.local_experts_per_layer, self.layer_count)
        self.prediction_weight = offload.cache_router_weight
        if not isfinite(self.prediction_weight) or self.prediction_weight < 0:
            raise ValueError("O-MoE cache_router_weight must be finite and nonnegative")
        # The hierarchical path always has an online policy. The legacy
        # cache_policy_enabled switch still controls only the original path.
        self.policy = CoordinatedLRCPolicy(
            layer_count=self.layer_count,
            local_experts=self.local_experts_per_layer,
            # TP routes are replicated; rank zero supplies the scoring input.
            worker_count=1,
            topk=topk,
            recent_window=offload.cache_recent_window,
            ema_beta=offload.cache_ema_beta,
            recent_weight=offload.cache_recent_weight,
            ema_weight=offload.cache_ema_weight,
            age_weight=offload.cache_age_weight,
        )
        # Grow up to the model's expert count while shared pages are available.
        self.total_local_experts = self.layer_count * self.local_experts_per_layer
        if self.allocator.available("expert") // EXPERT_PROJECTION_PAGES < self.required_slots:
            raise ValueError("Shared arena cannot satisfy offload_expert_limit residency")
        self.slots: dict[int, tuple[BlockHandle, ...]] = {}
        self._next_slot_id = 0
        self._worker_snapshots: list[dict[int, ExpertSlotSnapshot]] = [{} for _ in range(self.worker_count)]
        self._installed_slots: set[int] = set()
        self._next_delta_id = 1
        self.failed = False
        self._pending_delta: ExpertCacheDelta | None = None
        # As in O-MoE, initialize from available shared capacity, not from the
        # offload limit. The limit is the floor during later KV reclamation.
        experts = self.local_experts_per_layer
        adjustable_layers = self.layer_count - 1
        cached_per_layer = experts
        if adjustable_layers:
            capacity = self.allocator.available("expert") // EXPERT_PROJECTION_PAGES
            per_layer = min(experts, (capacity - experts) // adjustable_layers)
            cached_per_layer = max(0, (per_layer - 1) // self.AUTO_CACHE_ROUND_MULTIPLE
                                   * self.AUTO_CACHE_ROUND_MULTIPLE)
        self.initial_resident_counts = (experts,) + tuple(
            max(floor, cached_per_layer) for floor in self.resident_floors[1:])
        self._reserve_slots(max(self.required_slots, sum(self.initial_resident_counts)))

    def _update_worker_snapshots(
        self, replies: "Sequence[ExpertCacheSnapshot]", installed_slots: set[int], delta_id: int,
    ) -> None:
        if len(replies) != self.worker_count:
            raise RuntimeError("O-MoE execution must return every worker's expert state")
        snapshots = []
        removed_slots = self._installed_slots - installed_slots
        added_slots = installed_slots - self._installed_slots
        # The previous delta already proved coverage of retained slots. Sparse
        # replies need fresh records for additions, without copying every rank.
        for rank, reply in enumerate(replies):
            if reply["delta_id"] != delta_id:
                raise RuntimeError("O-MoE execution snapshot is stale")
            records = {record["slot_id"]: record for record in reply["slots"]}
            if len(records) != len(reply["slots"]) or not records.keys() <= installed_slots:
                raise RuntimeError("O-MoE execution snapshot contains duplicate or uninstalled slots")
            if reply.get("incremental", False):
                if not added_slots <= records.keys():
                    raise RuntimeError("O-MoE execution snapshot must cover every new slot")
            elif records.keys() != installed_slots:
                raise RuntimeError("O-MoE execution snapshot must cover every installed slot exactly once")
            snapshots.append(records)
        self.policy.observe_snapshots([replies[0]["observations"]])
        # Every worker still seals its own slots before any retired page can
        # be reused. Only the replicated routing statistics use rank zero.
        for rank, (reply, records) in enumerate(zip(replies, snapshots)):
            if reply.get("incremental", False):
                previous = self._worker_snapshots[rank]
                for slot in removed_slots:
                    del previous[slot]
                previous.update(records)
            else:
                self._worker_snapshots[rank] = records

    def _reserve_slots(self, count: int) -> None:
        if not count:
            return
        # Gate, up and down need not occupy adjacent L2s.
        handles = self.allocator.reserve({"expert": count * EXPERT_PROJECTION_PAGES})["expert"]
        for start in range(0, len(handles), EXPERT_PROJECTION_PAGES):
            self.slots[self._next_slot_id] = tuple(handles[start : start + EXPERT_PROJECTION_PAGES])
            self._next_slot_id += 1

    def _select_release_slots(self, num_experts: int) -> tuple[int, ...]:
        # TP workers retain separate pending state and prediction hints. Pages
        # selected here stay owned until every worker finishes this step.
        max_candidates = min(num_experts, max(0, len(self.slots) - self.required_slots))
        if max_candidates <= 0:
            return ()

        counts = [Counter(record["key"][0] for record in snapshot.values()
                          if record["key"] is not None)
                  for snapshot in self._worker_snapshots]
        slot_priorities = []
        priorities = self.policy.priorities
        # Reject rank-zero floor violations before building priorities. This
        # cheaply filters the common TP case; the final check covers all ranks.
        protected_layers = set(counts[0]) if sum(counts[0].values()) <= self.required_slots else {
            layer for layer, count in counts[0].items() if count <= self.resident_floors[layer]
        }
        rank_zero_snapshot = self._worker_snapshots[0]
        for slot in self._installed_slots:
            key = rank_zero_snapshot[slot]["key"]
            if key is not None and key[0] in protected_layers:
                continue
            occupied, priority, last_used = False, None, 0
            for snapshot in self._worker_snapshots:
                record = snapshot[slot]
                key = record["key"]
                score = 0.0
                if key is not None:
                    occupied = True
                    score = priorities[tuple(key)]
                    prediction = record["prediction_priority"]
                    if prediction is not None:
                        score = max(score, prediction)
                if priority is None or score > priority:
                    priority = score
                last_used = max(last_used, record["last_used"])
            slot_priorities.append((occupied, priority, last_used, slot))
        resident_totals = [sum(rank_counts.values()) for rank_counts in counts]
        # Usually only one or a few experts are reclaimed. A temporary heap
        # preserves the same priority order without sorting every cached slot.
        heapq.heapify(slot_priorities)
        eligible = []
        while slot_priorities and len(eligible) < max_candidates:
            slot = heapq.heappop(slot_priorities)[-1]
            keys = [snapshot[slot]["key"] for snapshot in self._worker_snapshots]
            if any(key is not None and (
                counts[rank][key[0]] <= self.resident_floors[key[0]]
                or resident_totals[rank] <= self.required_slots
            ) for rank, key in enumerate(keys)):
                continue
            eligible.append(slot)
            for rank, key in enumerate(keys):
                if key is not None:
                    counts[rank][key[0]] -= 1
                    resident_totals[rank] -= 1
        return tuple(eligible)

    def adjust_expert_cache_capacity(self, num_scheduled_tokens: int) -> ExpertCacheDelta:
        if self.failed:
            raise RuntimeError("O-MoE expert manager failed; restart the engine")
        if self._pending_delta is not None:
            raise RuntimeError("Complete the previous expert cache delta before scheduling another")
        victims = ()
        required_blocks = self.cache_manager.required_expert_shrink_blocks
        if num_scheduled_tokens > 0 and required_blocks > 0:
            # O-MoE retains one additional projection block beyond the deficit.
            victims = self._select_release_slots(cdiv(required_blocks + 1, EXPERT_PROJECTION_PAGES))
        elif num_scheduled_tokens > 0 and self._installed_slots:
            free_pages = self.allocator.available("expert")
            if free_pages < self.LOW_WATERMARK_EXPERT_PAGES:
                victims = self._select_release_slots(1)
            else:
                # Keep OMoE's expansion margin instead of immediately
                # regrowing every page released for the next KV allocation.
                reserved_pages = self.EXPAND_HEADROOM_EXPERTS * EXPERT_PROJECTION_PAGES
                growth = min(
                    self.MAX_GROWTH_PER_STEP,
                    self.total_local_experts - len(self.slots),
                    max(0, free_pages - reserved_pages) // EXPERT_PROJECTION_PAGES,
                )
                self._reserve_slots(growth)
        additions = [
            (slot, tuple(handle.index for handle in handles))
            for slot, handles in self.slots.items()
            if slot not in self._installed_slots
        ]
        delta_id = self._next_delta_id
        # Pure TP already uses rank zero's residency for the common load
        # plan. Other ranks' snapshots remain available for release checks.
        resident_keys = {
            tuple(record["key"]) for slot, record in self._worker_snapshots[0].items()
            if slot not in victims and record["key"] is not None
        }
        # Shared-cache expansion fills granted empty slots. This is independent
        # of the optional next-layer prediction budget and enable switch.
        available_slots = len(self.slots) - len(victims)
        policy = self.policy.build_payload(
            resident_keys=resident_keys,
            load_limit=available_slots - len(resident_keys) if self._installed_slots else 0,
            prediction_weight=self.prediction_weight,
        )
        updates, zeros = self.cache_manager.take_worker_updates()
        initial_counts = self.initial_resident_counts if delta_id == 1 else ()
        self._pending_delta = ExpertCacheDelta(delta_id, additions, victims, policy, updates, zeros, initial_counts)
        self._next_delta_id += 1
        return self._pending_delta

    def complete_cache_delta(
        self, delta: ExpertCacheDelta, worker_snapshots: "Sequence[ExpertCacheSnapshot]",
    ) -> None:
        """Commit after normal execute_model completion, never during admission."""
        try:
            if self.failed or delta is not self._pending_delta:
                raise RuntimeError("Unexpected expert cache delta completion")
            installed = set(self.slots).difference(delta.released_slots)
            # Validate all ranks before making even one retired page reusable.
            self._update_worker_snapshots(worker_snapshots, installed, delta.delta_id)
            self.allocator.free(
                handle for slot in delta.released_slots for handle in self.slots[slot]
            )
            for slot in delta.released_slots:
                del self.slots[slot]
            self._installed_slots = installed
            self._pending_delta = None
        except BaseException:
            self.failed = True
            raise
