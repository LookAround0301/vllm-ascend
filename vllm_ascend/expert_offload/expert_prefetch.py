# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Bounded prediction jobs and event-owned shared-pool expert destinations.

Prediction itself remains in ExpertOffloadManager. This module neither changes
routing nor allocates device storage: shared grants and fixed buffers are supplied by the manager.
"""

from collections import Counter, deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from threading import Condition, RLock, Thread
from typing import TYPE_CHECKING, Literal, TypedDict

import torch

from vllm_ascend.expert_offload.expert_memory import (
    CompletionEvent, ExpertProjectionTensorViews, ExpertTensorViews, projection_expert_views,
)
from vllm_ascend.expert_offload.h2d_transfer import H2DCopyTask

if TYPE_CHECKING:
    from vllm_ascend.expert_offload.expert_manager import ExpertSlotGrant
    from vllm_ascend.expert_offload.lrc_policy import ExpertKey, RoutingObservation
    from vllm_ascend.ops.fused_moe.discrete_moe_mlp import PreparedDiscreteMoEWeights


class ExpertInstallResult(TypedDict):
    """Worker-local result of installing expert page grants."""

    installed: bool
    delta_id: int
    slot_ids: list[int]


class ExpertSlotSnapshot(TypedDict):
    """A worker's resident key and eviction hints for one scheduler-owned slot."""

    slot_id: int
    key: "ExpertKey | None"
    last_used: int
    prediction_priority: float | None


class ExpertCacheSnapshot(TypedDict):
    """Sealed worker state returned with normal model execution."""

    delta_id: int
    # Incremental snapshots contain only changed slots; releases are in the delta.
    incremental: bool
    slots: list[ExpertSlotSnapshot]
    stats: dict[str, int]
    observations: "list[RoutingObservation]"


@dataclass(frozen=True, eq=False, slots=True)
class ExpertPrefetchTicket:
    """Identity-bearing CPU ticket for one source-layer forward invocation."""

    delta_id: int
    layer_idx: int


@dataclass(frozen=True, slots=True)
class ExpertPrefetchReady:
    """Opaque prediction and fresh compute event claimed exactly once."""

    ticket: ExpertPrefetchTicket
    payload: object
    event: object


@dataclass(slots=True)
class PrefetchHandoffState:
    ticket: ExpertPrefetchTicket
    payload: object = None
    event: object = None
    prediction_arrived: bool = False
    compute_arrived: bool = False
    claimed: bool = False


class ExpertPrefetchHandoff:
    """Join independent CPU notifications without waiting or submitting work.

    Prediction D2H completion publishes an opaque payload; the forward thread
    publishes its freshly recorded compute event separately. Whichever arrives
    second claims one ready packet. A claimed ticket remains current until
    ``finish``/``cancel`` so queued work can check cancellation before submitting
    H2D. Cancellation does not stop an already executing transfer: the caller
    still owns its event dependencies and must drain it before recycling pages.

    There is at most one retained state per layer. No event method, callback,
    queue operation or device wait runs in this coordinator. The manager applies
    each cache delta; a begin supersedes the previous ticket for that layer,
    irrespective of whether either side had already arrived.
    """

    def __init__(self):
        self.handoff_lock = RLock()
        self.handoffs_by_layer = {}

    def get_handoff_state(self, ticket):
        handoff_state = self.handoffs_by_layer.get(ticket.layer_idx)
        return handoff_state if handoff_state is not None and handoff_state.ticket is ticket else None

    @staticmethod
    def claim_ready_prefetch(handoff_state):
        if handoff_state.claimed or not handoff_state.prediction_arrived or not handoff_state.compute_arrived:
            return None
        handoff_state.claimed = True
        return ExpertPrefetchReady(handoff_state.ticket, handoff_state.payload, handoff_state.event)

    def begin(self, delta_id: int, layer_idx: int) -> ExpertPrefetchTicket:
        with self.handoff_lock:
            ticket = ExpertPrefetchTicket(delta_id, layer_idx)
            # Keep retired payload/event references alive until outside the
            # lock; even user-owned destructors must not run under our lock.
            retired_handoff = self.handoffs_by_layer.pop(layer_idx, None)
            self.handoffs_by_layer[layer_idx] = PrefetchHandoffState(ticket)
        del retired_handoff
        return ticket

    def current(self, delta_id: int, layer_idx: int) -> ExpertPrefetchTicket | None:
        with self.handoff_lock:
            handoff_state = self.handoffs_by_layer.get(layer_idx)
            return handoff_state.ticket if handoff_state is not None and handoff_state.ticket.delta_id == delta_id else None

    def is_current(self, ticket: ExpertPrefetchTicket) -> bool:
        with self.handoff_lock:
            return self.get_handoff_state(ticket) is not None

    def prediction_ready(self, ticket: ExpertPrefetchTicket, payload: object) -> ExpertPrefetchReady | None:
        with self.handoff_lock:
            handoff_state = self.get_handoff_state(ticket)
            if handoff_state is None or handoff_state.claimed or handoff_state.prediction_arrived:
                return None
            handoff_state.payload = payload
            handoff_state.prediction_arrived = True
            return self.claim_ready_prefetch(handoff_state)

    def compute_ready(self, ticket: ExpertPrefetchTicket, event: object) -> ExpertPrefetchReady | None:
        if event is None:
            raise ValueError("Compute readiness requires a freshly recorded event")
        with self.handoff_lock:
            handoff_state = self.get_handoff_state(ticket)
            if handoff_state is None or handoff_state.claimed or handoff_state.compute_arrived:
                return None
            handoff_state.event = event
            handoff_state.compute_arrived = True
            return self.claim_ready_prefetch(handoff_state)

    def finish(self, ticket: ExpertPrefetchTicket) -> bool:
        with self.handoff_lock:
            handoff_state = self.get_handoff_state(ticket)
            if handoff_state is None:
                return False
            retired_handoff = self.handoffs_by_layer.pop(ticket.layer_idx)
        del retired_handoff
        return True

    def cancel(self, ticket: ExpertPrefetchTicket) -> bool:
        return self.finish(ticket)

    def cancel_all(self) -> int:
        with self.handoff_lock:
            retired_handoffs = self.handoffs_by_layer
            self.handoffs_by_layer = {}
        num_cancelled_handoffs = len(retired_handoffs)
        del retired_handoffs
        return num_cancelled_handoffs


class ExpertPrefetchQueue:
    """One bounded CPU submission worker; drain waits for submissions, not H2D.

    A job owns its CPU staging and device source references until it returns.
    Failures are sticky and re-raised on the forward thread. No lock is held
    while a job waits for its prediction's D2H event or submits device work.
    """

    def __init__(self, capacity: int = 2):
        if capacity < 1:
            raise ValueError("Prefetch queue capacity must be positive")
        self.capacity = capacity
        self.job_condition = Condition()
        self.pending_jobs = deque()
        self.num_active_jobs = 0
        self.closed = False
        self.job_error = None
        self.worker_thread = Thread(target=self.run_jobs, name="ascend-expert-prefetch", daemon=True)
        self.worker_thread.start()

    def check(self):
        with self.job_condition:
            if self.job_error is not None:
                raise RuntimeError("Shared-pool expert prefetch failed") from self.job_error

    def submit(self, job: Callable[[], None]) -> bool:
        with self.job_condition:
            if self.job_error is not None:
                raise RuntimeError("Shared-pool expert prefetch failed") from self.job_error
            if self.closed:
                raise RuntimeError("Expert prefetch queue is closed")
            if len(self.pending_jobs) + self.num_active_jobs >= self.capacity:
                return False
            self.pending_jobs.append(job)
            self.job_condition.notify_all()
            return True

    def drain(self):
        with self.job_condition:
            while self.pending_jobs or self.num_active_jobs:
                self.job_condition.wait()
        self.check()

    def close(self):
        # Drain even on failure: dropping a queued job could drop the last
        # reference to a staging tensor whose D2H is still in flight.
        with self.job_condition:
            self.closed = True
            self.job_condition.notify_all()
        self.worker_thread.join()
        self.check()

    def run_jobs(self):
        while True:
            with self.job_condition:
                while not self.pending_jobs and not self.closed:
                    self.job_condition.wait()
                if not self.pending_jobs:
                    return
                job = self.pending_jobs.popleft()
                self.num_active_jobs += 1
            try:
                job()
            except BaseException as exc:
                with self.job_condition:
                    if self.job_error is None:
                        self.job_error = exc
            finally:
                with self.job_condition:
                    self.num_active_jobs -= 1
                    self.job_condition.notify_all()


@dataclass(eq=False)
class ExpertPageSlot:
    slot_id: int
    block_ids: tuple[int, int, int]
    views: ExpertProjectionTensorViews | ExpertTensorViews
    key: tuple[int, int] | None = None
    loading: bool = False
    computing: bool = False
    load_event: CompletionEvent | None = None
    compute_events: list[CompletionEvent] = field(default_factory=list)
    # Cache recency includes insertion; ``used`` records actual computation.
    last_use: int = 0
    prefetched: bool = False
    used: bool = False

    def pending_events(self) -> tuple[CompletionEvent, ...]:
        # Query before changing state so an event failure retains every reference.
        load_event = self.load_event
        if load_event is not None and load_event.query():
            load_event = None
        compute_events = [event for event in self.compute_events if not event.query()]
        self.load_event = load_event
        self.compute_events = compute_events
        return (() if load_event is None else (load_event,)) + tuple(compute_events)


class ExpertDeviceTables:
    """Persistent expert metadata, charged to model memory before arena sizing."""

    def __init__(self, layers: int, experts: int, device: torch.device):
        from vllm_ascend.ops.fused_moe.discrete_moe_mlp import DESCRIPTOR_COLUMNS

        self.descriptors = torch.empty((layers, experts, DESCRIPTOR_COLUMNS), dtype=torch.int64, device=device)
        self.resident = torch.zeros((layers, experts), dtype=torch.int32, device=device)
        self.counts = torch.zeros((layers, experts), dtype=torch.int64, device=device)
        self.source_bases = torch.empty((layers, 3), dtype=torch.int64, device=device)
        self.ready = torch.zeros((2, experts), dtype=torch.int32, device=device)


@dataclass
class DeviceComputeBuffer:
    storage: torch.Tensor | None
    ready: torch.Tensor
    context: tuple[int, int] | None = None
    event: object | None = None
    computing: bool = False


class ExpertPageCache:
    """Resident shared-cache pages plus the original O-MoE double buffers.

    Cache membership changes only on scheduler plans. A routed miss uses a
    row in the current compute buffer; prediction fills the alternate buffer.
    Neither path searches or evicts the whole shared cache during computation.
    """

    def __init__(
        self,
        *,
        arena,
        layout,
        page_bytes,
        local_experts,
        sources,
        transport,
        prediction_weight: float = 0.3,
        resident_floor: dict[int, int] | None = None,
        resident_total: int | None = None,
        compute_buffers: tuple[torch.Tensor, ...] = (),
        device_tables: ExpertDeviceTables,
        weight_layout: Literal["nd", "nz"] = "nd",  # codespell:ignore nd
    ):
        self.arena = arena
        self.layout = layout
        self.page_bytes = page_bytes
        self.local_experts = local_experts
        self.sources = sources
        self.weight_layout = weight_layout
        self.transport = transport
        self.delta_id = 0
        self.stats = Counter()
        self.slots_by_id: dict[int, ExpertPageSlot] = {}
        self.free_slots: deque[ExpertPageSlot] = deque()
        self.changed_slots: set[int] = set()
        self.slot_ids_by_expert: dict[tuple[int, int], int] = {}
        self.cache_lock = RLock()
        self.access_sequence = 0
        self.submission_error = None
        self.revoking_pages = False
        self.prediction_weight = float(prediction_weight)
        self.prediction_priorities = {}
        self.observation_revisions = {layer: 0 for layer in sources}
        self.observed_token_counts = {layer: [0] * local_experts for layer in sources}
        self.resident_floor = dict.fromkeys(sources, 0) if resident_floor is None else dict(resident_floor)
        layer_minimum = sum(self.resident_floor.values())
        self.resident_total = layer_minimum if resident_total is None else resident_total
        self.resident_counts = Counter()
        self.residency_initialized = self.resident_total == 0
        self.device_tables = device_tables
        self.device_prepared_weights: dict[int, PreparedDiscreteMoEWeights] = {}
        self.device_sources: list[torch.Tensor] = []
        self.changed_device_layers = set(sources)
        self.device_compute_event = None
        self.device_buffers = [
            DeviceComputeBuffer(raw, device_tables.ready[index])
            for index, raw in enumerate(compute_buffers or (None, None))
        ]

    def prepare_device_weights(self, layers: Sequence[torch.nn.Module], stream) -> None:
        """Publish persistent addresses after the batch's residency changes.

        CPU sources are mapped once. Resident experts point into scheduler
        grants; every other expert gets a fixed compact-buffer row for that
        layer. The residency floor bounds these rows by the original limit.
        Descriptors are published before their layer executes; each layer's
        readiness event covers its own weight loads and address updates.
        """
        from vllm.utils.math_utils import round_up

        from vllm_ascend.ops.fused_moe.discrete_moe_mlp import (
            DiscreteMoEWeights,
            prepare_discrete_moe_weights,
        )

        tables = self.device_tables
        with self.cache_lock:
            self.ensure_submissions_allowed()
            for buffer in self.device_buffers:
                if buffer.computing:
                    raise RuntimeError("Cannot update expert addresses during an unsealed computation")
                if buffer.event is not None:
                    stream.wait_event(buffer.event)
            h, i = self.layout.hidden_size, self.layout.intermediate_size
            projection_bytes = round_up(h * i, self.layout.alignment)
            device = tables.descriptors.device
            template = next(iter(self.device_prepared_weights.values()), None)
            if template is None:
                template = prepare_discrete_moe_weights(
                    DiscreteMoEWeights((), h, i, self.weight_layout), device=device
                )
            for layer_idx in self.changed_device_layers:
                layer = layers[layer_idx]
                buffer = self.device_buffers[layer_idx % len(self.device_buffers)]
                scale13, scale2 = layer.w13_weight_scale_fp32, layer.w2_weight_scale
                if layer_idx not in self.device_prepared_weights:
                    gate, down, up = self.sources[layer_idx]
                    mapped_sources = (gate, up, down)
                    addresses = torch.ops._C_ascend.register_omoe_host_sources(mapped_sources)
                    # Retain registered pinned allocations even if a later
                    # metadata allocation fails; close drains before unmapping.
                    self.device_sources.extend(mapped_sources)
                    tables.source_bases[layer_idx].copy_(
                        torch.tensor(addresses, dtype=torch.int64, pin_memory=True), non_blocking=True
                    )
                rows, resident, missing = [], [], 0
                for expert_id in range(self.local_experts):
                    slot_id = self.slot_ids_by_expert.get((layer_idx, expert_id))
                    if slot_id is not None:
                        gate, down, up = (
                            self.arena.data_ptr() + block_id * self.page_bytes
                            for block_id in self.slots_by_id[slot_id].block_ids
                        )
                        resident.append(1)
                    else:
                        if buffer.storage is None or (missing + 1) * 3 * projection_bytes > buffer.storage.numel():
                            raise RuntimeError("Nonresident experts exceed the fixed compute-buffer capacity")
                        gate = buffer.storage.data_ptr() + missing * 3 * projection_bytes
                        down, up = gate + projection_bytes, gate + 2 * projection_bytes
                        resident.append(0)
                        missing += 1
                    gate_scale = scale13.data_ptr() + expert_id * 2 * i * scale13.element_size()
                    rows.append((gate, up, down, gate_scale, gate_scale + i * scale13.element_size(),
                                 scale2.data_ptr() + expert_id * h * scale2.element_size(), i, i))
                descriptor = tables.descriptors[layer_idx]
                # Pinned sources let the recorded layer event publish these
                # copies without blocking the host behind queued weight loads.
                descriptor.copy_(torch.tensor(rows, dtype=torch.int64, pin_memory=True), non_blocking=True)
                tables.resident[layer_idx].copy_(
                    torch.tensor(resident, dtype=torch.int32, pin_memory=True), non_blocking=True
                )
                refs = (self.arena, scale13, scale2, descriptor, template.tiling)
                if buffer.storage is not None:
                    refs += (buffer.storage,)
                if template.large_tiling is not None:
                    refs += (template.large_tiling,)
                self.device_prepared_weights[layer_idx] = replace(
                    template, descriptor=descriptor, tensor_refs=refs,
                    creation_stream=stream, ready_event=stream.record_event(),
                    creation_stream_key=(stream.stream_id, stream.device_index, stream.device_type),
                )
            self.changed_device_layers.clear()

    def prepare_device_experts(
        self, layer_idx: int, groups: torch.Tensor, cumulative: bool, stream,
        *, prediction_revision: int | None = None,
    ) -> "PreparedDiscreteMoEWeights | None":
        """Select actual misses on device; CPU never reads per-layer counts."""
        with self.cache_lock:
            self.ensure_submissions_allowed()
            self.ensure_cache_installed()
            revision = self.observation_revisions[layer_idx]
            if prediction_revision is not None and prediction_revision != revision:
                self.stats["prefetch_stale_observation"] += 1
                return None
            buffer = self.device_buffers[layer_idx % len(self.device_buffers)]
            if buffer.computing:
                raise RuntimeError("A compute buffer has an unsealed reader")
            stream.wait_event(self.device_prepared_weights[layer_idx].ready_event)
            if buffer.event is not None:
                stream.wait_event(buffer.event)
            context = (layer_idx, revision)
            if buffer.context != context:
                buffer.ready.zero_()
                buffer.context = context
            observe = prediction_revision is None
            if observe:
                self.observation_revisions[layer_idx] += 1
                for expert_id in range(self.local_experts):
                    key = (layer_idx, expert_id)
                    if self.prediction_priorities.pop(key, None) is not None and key in self.slot_ids_by_expert:
                        self.changed_slots.add(self.slot_ids_by_expert[key])
                self.stats["real_observation_batches"] += 1
            tables = self.device_tables
            torch.ops._C_ascend.prepare_omoe_experts(
                tables.source_bases[layer_idx], tables.descriptors[layer_idx], tables.resident[layer_idx],
                buffer.ready, groups, tables.counts[layer_idx],
                self.layout.hidden_size * self.layout.intermediate_size, cumulative, observe,
            )
            if observe:
                buffer.computing = True
            else:
                buffer.event = stream.record_event()
            return self.device_prepared_weights[layer_idx]

    def release_device_compute(self, layer_idx: int, event) -> None:
        with self.cache_lock:
            buffer = self.device_buffers[layer_idx % len(self.device_buffers)]
            buffer.event = event
            buffer.computing = False
            # Normal model execution admits one batch at a time. The last
            # compute event fences all earlier reads before grant revocation.
            self.device_compute_event = event

    def ensure_submissions_allowed(self):
        if self.submission_error is not None:
            raise RuntimeError("Shared expert-page submission failed") from self.submission_error
        if self.revoking_pages:
            raise RuntimeError("Expert-page revocation is in flight or failed; submissions are frozen")

    def record_stat(self, name, count=1):
        with self.cache_lock:
            self.stats[name] += count

    def granted_block_ids(self) -> tuple[int, ...]:
        with self.cache_lock:
            return tuple(block_id for slot in self.slots_by_id.values() for block_id in slot.block_ids)

    def validate_cache_delta(self, delta_id: int) -> None:
        if delta_id != self.delta_id:
            raise RuntimeError("Stale expert cache delta")

    def ensure_cache_installed(self):
        if self.delta_id == 0:
            raise RuntimeError("Install expert cache grants before submitting work")

    def observation_revision(self, layer_idx: int) -> int:
        with self.cache_lock:
            self.ensure_submissions_allowed()
            return self.observation_revisions[layer_idx]

    def install(self, grants: "Sequence[ExpertSlotGrant]", delta_id: int) -> ExpertInstallResult:
        with self.cache_lock:
            self.ensure_submissions_allowed()
            if delta_id != self.delta_id + 1:
                raise RuntimeError("Expert-page install must advance exactly one cache delta")
            # The scheduler's block pool owns these pages. Build each slot's views
            # once; reloading another expert reuses the same tensors and block IDs.
            new_slots = [
                ExpertPageSlot(slot_id, block_ids, projection_expert_views(self.arena, block_ids, self.page_bytes, self.layout))
                for slot_id, block_ids in grants
            ]
            if len(self.slots_by_id) + len(new_slots) < self.resident_total:
                raise ValueError("Expert grants do not cover the admitted local working set and offload_expert_limit")
            for slot in new_slots:
                self.slots_by_id[slot.slot_id] = slot
                self.changed_slots.add(slot.slot_id)
                self.free_slots.append(slot)
            self.delta_id = delta_id
            self.changed_slots.update(
                self.slot_ids_by_expert[key] for key in self.prediction_priorities if key in self.slot_ids_by_expert
            )
            self.prediction_priorities.clear()
        return {"installed": True, "delta_id": delta_id, "slot_ids": [slot.slot_id for slot in new_slots]}

    def initialize_residency(self, stream, *, initial_resident_counts: tuple[int, ...] = ()):
        """Populate the scheduler's startup allocation before the first forward."""
        if self.residency_initialized:
            return
        counts = dict(enumerate(initial_resident_counts)) if initial_resident_counts else self.resident_floor
        targets = {
            (layer, eid)
            for layer, floor in counts.items()
            for eid in range(floor)
        }
        # Startup has no route history. Follow expert ID order; subsequent
        # residency changes arrive as explicit scheduler load plans.
        candidates = [
            (layer, eid) for layer in sorted(self.sources) for eid in range(self.local_experts)
            if (layer, eid) not in targets
        ]
        targets.update(candidates[:max(0, self.resident_total - len(targets))])
        # Initial grants are empty: bind each mandatory expert once, without
        # running the online replacement policy thousands of times at startup.
        for key in sorted(targets):
            with self.cache_lock:
                slot = self.free_slots.popleft()
                reservation = self.bind_slot(slot, key, prefetch=False)
            self.load_expert_weights(reservation, key, stream)
        stream.synchronize()
        with self.cache_lock:
            self.ensure_submissions_allowed()
            if any(
                key not in self.slot_ids_by_expert
                or not self.is_slot_load_complete(self.slots_by_id[self.slot_ids_by_expert[key]])
                for key in targets
            ):
                raise RuntimeError("Initial expert residency did not complete")
            self.residency_initialized = True
            self.stats["initial_resident_loads"] += len(targets)

    @staticmethod
    def is_slot_load_complete(slot):
        if slot.key is None or slot.loading:
            return False
        if slot.load_event is not None:
            if not slot.load_event.query():
                return False
            slot.load_event = None
        return True


    def release_slot_contents(self, slot):
        # Caller has fenced the slot's previous copy and computation.
        if slot.key is not None:
            if slot.prefetched and not slot.used:
                self.stats["unused_prefetch"] += 1
            self.slot_ids_by_expert.pop(slot.key)
            self.resident_counts[slot.key[0]] -= 1
            self.changed_device_layers.add(slot.key[0])
        slot.key = None
        slot.prefetched = False
        slot.used = False

    def bind_slot(self, slot: ExpertPageSlot, key: tuple[int, int], *, prefetch: bool):
        # Caller owns the cache lock and has proved the previous use complete.
        self.release_slot_contents(slot)
        slot.key = key
        slot.loading = True
        slot.prefetched = prefetch
        self.access_sequence += 1
        slot.last_use = self.access_sequence
        self.slot_ids_by_expert[key] = slot.slot_id
        self.changed_slots.add(slot.slot_id)
        self.resident_counts[key[0]] += 1
        self.changed_device_layers.add(key[0])
        return slot

    def load_expert_weights(self, load_reservation, expert_key, stream):
        slot = load_reservation
        load_succeeded = False
        try:
            views = slot.views
            destinations = (views.gate, views.down, views.up)
            sources = self.sources[expert_key[0]]
            tasks = [
                H2DCopyTask(
                    source=source[expert_key[1]],
                    destination=destination,
                    nbytes=destination.numel() * destination.element_size(),
                    name=f"{name}[L{expert_key[0]},E{expert_key[1]}->page-slot{slot.slot_id}]",
                )
                for name, source, destination in zip(
                    ("gate", "down", "up"), sources, destinations
                )
            ]
            with torch.npu.stream(stream):
                try:
                    self.transport.copy_batch(tasks)
                    load_succeeded = True
                finally:
                    event = stream.record_event()
                    slot.load_event = event
            with self.cache_lock:
                slot.loading = False
                self.stats["prefetch_loads" if slot.prefetched else "reactive_loads"] += int(load_succeeded)
        except BaseException as exc:
            # A failed submission leaves loading set and poisons the cache;
            # these pages cannot be reused after a partial copy.
            with self.cache_lock:
                self.submission_error = exc
            raise

    def publish_predictions(
        self, layer_idx: int, local_ids: Sequence[int], delta_id: int,
        *, scores: Sequence[float], observation_revision: int,
    ) -> bool:
        """Publish confidence for every local prediction, including residents.

        Raw router softmax values are not re-normalized over a rank's experts.
        Their independent priority is ``prediction_weight * probability``,
        bounded by the configured weight (which can exceed one). Scheduler
        LRC priorities remain normalized to [-1, 1]; the weight explicitly
        sets prediction confidence's relative influence. Neither publishing
        nor loading predictions updates the real observation counters.
        """
        with self.cache_lock:
            self.ensure_submissions_allowed()
            self.ensure_cache_installed()
            # The predictor validates IDs/scores before publishing. Only the
            # cache delta and observation may become stale while work is queued.
            if delta_id != self.delta_id:
                self.stats["prefetch_stale_delta"] += 1
                return False
            if observation_revision != self.observation_revisions[layer_idx]:
                self.stats["prefetch_stale_observation"] += 1
                return False
            for eid, score in zip(local_ids, scores):
                key = (layer_idx, eid)
                self.prediction_priorities[key] = self.prediction_weight * float(score)
                if key in self.slot_ids_by_expert:
                    self.changed_slots.add(self.slot_ids_by_expert[key])
            return True

    def prefetch(
        self, layer_idx: int, local_ids: Sequence[int], delta_id: int, stream,
        *, observation_revision: int,
    ) -> int:
        """Prepare predicted misses after their retention hints were published."""
        with self.cache_lock:
            if delta_id != self.delta_id:
                return 0
            predicted = torch.zeros(self.local_experts, dtype=torch.int64)
            predicted[list(local_ids)] = 1
            prepared = self.prepare_device_experts(
                layer_idx, predicted.to(self.arena.device), False, stream, prediction_revision=observation_revision,
            )
            if prepared is not None:
                self.stats["device_prediction_submissions"] += 1
            return len(local_ids) if prepared is not None else 0

    def load_cache_plan(self, keys: Sequence["ExpertKey"], delta_id: int, stream) -> int:
        """Fill empty shared grants without evicting any resident expert."""
        with self.cache_lock:
            self.ensure_submissions_allowed()
            self.validate_cache_delta(delta_id)
            self.ensure_cache_installed()
        submitted = 0
        for key in keys:
            with self.cache_lock:
                self.ensure_submissions_allowed()
                self.validate_cache_delta(delta_id)
                if key in self.slot_ids_by_expert:
                    continue
                if not self.free_slots:
                    raise RuntimeError("Shared-cache load plan exceeds its granted capacity")
                reservation = self.bind_slot(self.free_slots.popleft(), key, prefetch=True)
            self.load_expert_weights(reservation, key, stream)
            submitted += 1
        self.record_stat("cache_plan_loads", submitted)
        return submitted

    def is_fully_resident(self, layer_idx: int) -> bool:
        with self.cache_lock:
            return self.resident_counts[layer_idx] == self.local_experts

    def missing_candidates(self, layer_idx, local_ids):
        with self.cache_lock:
            ordered = dict.fromkeys(local_ids)
            return [eid for eid in ordered if (layer_idx, eid) not in self.slot_ids_by_expert]

    def close_compute_buffers(self) -> None:
        """Called after copy/compute streams drain; never hide an unsealed pin."""
        with self.cache_lock:
            for buffer in self.device_buffers:
                if buffer.computing:
                    raise RuntimeError("Compute buffer has an unsealed reader at close")
                if buffer.event is not None:
                    buffer.event.synchronize()
            if self.device_sources:
                torch.ops._C_ascend.unregister_omoe_host_sources(self.device_sources)
                self.device_sources.clear()
            self.device_prepared_weights.clear()

    def snapshot(
        self, delta_id: int, *, teardown: bool = False, incremental: bool = False,
    ) -> ExpertCacheSnapshot:
        with self.cache_lock:
            self.ensure_submissions_allowed()
            self.validate_cache_delta(delta_id)
            if any(buffer.computing for buffer in self.device_buffers):
                raise RuntimeError("Expert snapshot requires sealed device computation")
        device_counts = None
        if self.device_prepared_weights:
            if self.device_compute_event is not None:
                torch.npu.current_stream().wait_event(self.device_compute_event)
            device_counts = self.device_tables.counts.cpu().tolist()
        with self.cache_lock:
            if device_counts is not None:
                for layer_idx, counts in enumerate(device_counts):
                    previous = self.observed_token_counts[layer_idx]
                    for expert_id, (old, new) in enumerate(zip(previous, counts)):
                        if new == old:
                            continue
                        self.stats["real_routed_tokens"] += new - old
                        slot_id = self.slot_ids_by_expert.get((layer_idx, expert_id))
                        if slot_id is None:
                            self.stats["device_routed_cache_misses"] += 1
                        else:
                            slot = self.slots_by_id[slot_id]
                            self.access_sequence += 1
                            slot.last_use = self.access_sequence
                            slot.used = True
                            self.changed_slots.add(slot_id)
                    self.observed_token_counts[layer_idx] = counts
            self.ensure_submissions_allowed()
            self.validate_cache_delta(delta_id)
            if not teardown and delta_id and self.resident_total and (
                not self.residency_initialized
                or len(self.slot_ids_by_expert) < self.resident_total
                or any(self.resident_counts[layer] < floor for layer, floor in self.resident_floor.items())
            ):
                raise RuntimeError("Expert snapshot violates offload_expert_limit residency")
            slots: list[ExpertSlotSnapshot] = []
            # Every load/read marks its shared slot. Untouched slots cannot
            # acquire a new unsealed pin, so retain their preceding snapshot.
            snapshot_slots = (
                (self.slots_by_id[slot_id] for slot_id in self.changed_slots) if incremental else self.slots_by_id.values()
            )
            for slot in snapshot_slots:
                if slot.loading or slot.computing:
                    raise RuntimeError("Expert snapshot requires sealed load and compute submissions")
                slots.append(
                    {
                        "slot_id": slot.slot_id,
                        "key": slot.key,
                        "last_used": slot.last_use,
                        "prediction_priority": self.prediction_priorities.get(slot.key),
                    }
                )
            self.changed_slots.clear()
            return {
                "delta_id": delta_id,
                "incremental": incremental,
                "slots": slots,
                "stats": dict(self.stats),
                "observations": [
                    {"layer_idx": layer, "revision": self.observation_revisions[layer], "counts": list(counts)}
                    for layer, counts in sorted(self.observed_token_counts.items())
                ],
            }

    def release(self, slot_ids: Sequence[int], delta_id: int, *, teardown: bool = False):
        """Retire frozen submissions; any uncertain release remains sealed off.

        The manager pauses and drains prediction jobs before calling this.
        The cache also rejects new work throughout the lock-free device wait,
        including callers that bypass the manager.
        """
        ids = tuple(slot_ids)
        with self.cache_lock:
            self.ensure_submissions_allowed()
            self.validate_cache_delta(delta_id)
            if len(set(ids)) != len(ids) or any(i not in self.slots_by_id for i in ids):
                raise ValueError("Unknown or duplicate revoked expert slot")
            slots = [self.slots_by_id[i] for i in ids]
            if not teardown and self.resident_total:
                remaining = self.resident_counts.copy()
                remaining.subtract(s.key[0] for s in slots if s.key is not None)
                if (
                    len(self.slots_by_id) - len(ids) < self.resident_total
                    or sum(remaining.values()) < self.resident_total
                    or any(remaining[layer] < floor for layer, floor in self.resident_floor.items())
                ):
                    raise RuntimeError("Expert revocation violates offload_expert_limit residency or compute capacity")
            dependencies = [
                event for slot in slots for event in slot.pending_events()
            ]
            if any(buffer.computing for buffer in self.device_buffers):
                raise RuntimeError("Revocation requires a sealed device computation")
            if self.device_compute_event is not None:
                dependencies.append(self.device_compute_event)
            if any(slot.loading or slot.computing for slot in slots):
                raise RuntimeError("Revocation requires sealed expert submissions")
            self.revoking_pages = True
        for event in dependencies:
            event.synchronize()
        # New KV/state grants queue their own initialization; expert loads
        # overwrite their weights. Retirement need not clear these bytes too.
        with self.cache_lock:
            for slot in slots:
                if slot.key is not None:
                    self.slot_ids_by_expert.pop(slot.key)
                    self.resident_counts[slot.key[0]] -= 1
                    self.changed_device_layers.add(slot.key[0])
                    if slot.prefetched and not slot.used:
                        self.stats["unused_prefetch"] += 1
                del self.slots_by_id[slot.slot_id]
                self.changed_slots.discard(slot.slot_id)
            self.free_slots = deque(slot for slot in self.free_slots if slot.slot_id in self.slots_by_id)
            self.revoking_pages = False
        return {
            "released": True,
            "delta_id": delta_id,
            "slot_ids": list(ids),
        }
