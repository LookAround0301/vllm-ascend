"""Local-routing-consistency policy for expert offload decode paging."""

from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from math import isfinite
from operator import lt
from typing import NamedTuple, TypeAlias, TypedDict

import numpy as np

ExpertKey: TypeAlias = tuple[int, int]  # (MoE layer index, expert ID)


class RoutingObservation(TypedDict):
    """One layer's cumulative route counts; count indices are local expert IDs."""

    layer_idx: int
    revision: int
    counts: list[int]


class ExpertPolicy(TypedDict):
    """One shared-cache load plan executed by every TP worker."""

    # Scores stay in the scheduler; workers only need explicit load targets.
    load_plan: list[ExpertKey]
    prediction_weight: float


class RoutingObservationState(NamedTuple):
    """Validated cumulative counts for one worker and one MoE layer."""

    revision: int
    counts: tuple[int, ...]


@dataclass
class LRCLayerState:
    # Sparse ``(expert_id, route_count)`` snapshots.  ``observe`` stores one
    # snapshot per token while ``observe_global_counts`` stores one per global
    # decode step, without expanding counts into repeated expert ids.
    recent_queue: deque[tuple[tuple[int, int], ...]] = field(default_factory=deque)
    freq: list[int] = field(default_factory=list)
    # Only `ema` is vectorized (full-array C-level decay), so it is a numpy
    # array. freq / router_score / last_used stay Python lists: their hot-path
    # access is scalar, and `ndarray[i] += 1` is ~24x slower than `list[i] += 1`.
    ema: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.float32))
    router_score: list[float] = field(default_factory=list)
    last_used: list[int] = field(default_factory=list)
    step: int = 0


class LRCExpertCachePolicy:
    """Online approximation of SCH using recent frequency and EMA hotness.

    The policy is intentionally CPU-only. ExpertOffloadManager owns the actual
    CPU-to-NPU copies; this class only decides which resident expert should be
    evicted when a miss needs a device slot.
    """

    def __init__(
        self,
        num_layers: int,
        num_experts: int,
        cache_size: int,
        topk: int,
        recent_window: int = 32,
        ema_beta: float = 0.9,
        recent_weight: float = 1.0,
        ema_weight: float = 0.5,
        router_weight: float = 0.3,
        age_weight: float = 0.01,
    ) -> None:
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if num_experts < 1:
            raise ValueError("num_experts must be >= 1")
        if cache_size < 1:
            raise ValueError("cache_size must be >= 1")
        if topk < 1:
            raise ValueError("topk must be >= 1")
        if recent_window < 1:
            raise ValueError("recent_window must be >= 1")
        if not 0.0 <= ema_beta < 1.0:
            raise ValueError("ema_beta must be in [0, 1)")

        self.num_experts = num_experts
        self.cache_size = cache_size
        self.topk = topk
        self.recent_window = recent_window
        self.ema_beta = ema_beta
        self._one_minus_beta = 1.0 - ema_beta
        self.recent_weight = recent_weight
        self.ema_weight = ema_weight
        self.router_weight = router_weight
        self.age_weight = age_weight
        self.layer_states = [self._new_state() for _ in range(num_layers)]

    def add_layer(self) -> int:
        """Append a fresh per-layer state and return its index.

        Used when a MoE layer is registered after the policy was built —
        e.g. an MTP draft layer loaded after the target model's
        ``_finalize_offload``. The new layer shares the same
        ``num_experts`` / ``cache_size`` / ``topk`` as the existing layers
        and starts with empty statistics, so target- and draft-layer
        hotness are tracked independently.
        """
        self.layer_states.append(self._new_state())
        return len(self.layer_states) - 1

    def _new_state(self) -> LRCLayerState:
        return LRCLayerState(
            recent_queue=deque(),
            freq=[0 for _ in range(self.num_experts)],
            ema=np.zeros(self.num_experts, dtype=np.float32),
            router_score=[0.0 for _ in range(self.num_experts)],
            last_used=[-1 for _ in range(self.num_experts)],
        )

    def observe(
        self,
        layer_idx: int,
        topk_ids: Iterable[Iterable[int]],
        router_scores: Iterable[Iterable[float]] | None = None,
    ) -> set[int]:
        """Update per-layer routing statistics and return unique routed ids."""
        state = self.layer_states[layer_idx]
        score_rows = router_scores if router_scores is not None else []
        unique: set[int] = set()

        for row_index, row in enumerate(topk_ids):
            raw_experts = tuple(int(eid) for eid in row)

            # 保存有效专家在原始top-k中的位置。
            valid_routes = [
                (position, eid)
                for position, eid in enumerate(raw_experts)
                if 0 <= eid < self.num_experts
            ]

            experts = tuple(
                eid for _, eid in valid_routes
            )
            unique.update(experts)
            state.step += 1

            # Vectorized EMA: decay the whole array once at C level, then add
            # (1-beta) to the few hit experts. This is equivalent to the prior
            # per-expert loop because numpy indexed += does not accumulate
            # duplicate indices, matching the original set()-based hit semantics.
            state.ema *= self.ema_beta
            state.ema[np.array(experts, dtype=np.intp)] += self._one_minus_beta

            counts: dict[int, int] = {}
            for eid in experts:
                counts[eid] = counts.get(eid, 0) + 1
                state.freq[eid] += 1
                state.last_used[eid] = state.step
            state.recent_queue.append(tuple(counts.items()))

            if router_scores is not None:
                score_row = tuple(score_rows[row_index])

                if len(score_row) != len(raw_experts):
                    raise ValueError(
                        "router score width does not match "
                        "topk ID width: "
                        f"{len(score_row)} != {len(raw_experts)}"
                    )

                # 使用原始position取分数，避免过滤ID后错位。
                for position, eid in valid_routes:
                    state.router_score[eid] = float(score_row[position])

            while len(state.recent_queue) > self.recent_window:
                old_counts = state.recent_queue.popleft()
                for old_eid, old_count in old_counts:
                    state.freq[old_eid] -= old_count

        return unique

    def observe_global_counts(
        self,
        layer_idx: int,
        global_counts,
        global_router_score=None,
    ) -> set[int]:
        """Update one layer from globally aggregated decode statistics.

        ``global_counts[e]`` is the number of routes to expert ``e`` in this
        decode step.  The sparse recent-window entry preserves those real
        frequencies without materializing repeated ids.  Router scores, when
        supplied, are expected to be the global mean score per active expert.
        """
        counts = np.asarray(global_counts, dtype=np.int64)
        if counts.ndim != 1 or counts.size != self.num_experts:
            raise ValueError(
                f"global_counts must have shape ({self.num_experts},)")
        if np.any(counts < 0):
            raise ValueError("global_counts must be non-negative")

        scores = None
        if global_router_score is not None:
            scores = np.asarray(global_router_score, dtype=np.float32)
            if scores.shape != counts.shape:
                raise ValueError(
                    f"global_router_score must have shape ({self.num_experts},)")

        state = self.layer_states[layer_idx]
        state.step += 1
        state.ema *= self.ema_beta
        state.ema += counts.astype(np.float32) * self._one_minus_beta

        active = np.flatnonzero(counts)
        snapshot = tuple(zip(active.tolist(), counts[active].tolist()))
        state.recent_queue.append(snapshot)
        for eid, count in snapshot:
            state.freq[eid] += count
            state.last_used[eid] = state.step
            if scores is not None:
                state.router_score[eid] = float(scores[eid])

        while len(state.recent_queue) > self.recent_window:
            old_counts = state.recent_queue.popleft()
            for old_eid, old_count in old_counts:
                state.freq[old_eid] -= old_count
        return {eid for eid, _count in snapshot}

    def choose_victim(
        self,
        layer_idx: int,
        slot_owner: dict[int, int],
        protected: set[int],
        loading: set[int] | None = None,
    ) -> int | None:
        """Return resident expert id with the lowest predicted hotness."""
        loading = loading or set()
        skip = protected | loading
        candidates = [eid for eid in slot_owner.values() if eid not in skip]
        if not candidates:
            candidates = [eid for eid in slot_owner.values() if eid not in protected]
        if not candidates:
            return None

        return min(candidates, key=lambda eid: self._victim_key(layer_idx, eid))

    def choose_victims(
        self,
        layer_idx: int,
        slot_owner: dict[int, int],
        protected: set[int],
        count: int,
    ) -> list[int]:
        """Return up to ``count`` resident experts ordered coldest first.

        Unlike repeated :meth:`choose_victim` calls, this ranks the resident
        candidates once.  The stable sort preserves ``slot_owner`` iteration
        order when experts have equal hotness, matching ``min`` tie-breaking.
        """
        if count < 0:
            raise ValueError("victim count must be >= 0")
        if count == 0:
            return []

        candidates = [
            eid for eid in slot_owner.values() if eid not in protected
        ]
        candidates.sort(key=lambda eid: self._victim_key(layer_idx, eid))
        return candidates[:count]

    def hotness(self, layer_idx: int, expert_id: int) -> float:
        state = self.layer_states[layer_idx]
        age = 0 if state.last_used[expert_id] < 0 else state.step - state.last_used[expert_id]
        return (
            self.recent_weight * state.freq[expert_id]
            + self.ema_weight * state.ema[expert_id]
            + self.router_weight * state.router_score[expert_id]
            - self.age_weight * age
        )

    def hotness_array(self, layer_idx: int) -> np.ndarray:
        """Vectorized per-expert hotness for a layer (len == num_experts).

        Same scoring as :meth:`hotness` but computed over the whole array at
        once (used by the multi-card planner, which needs scores for all
        experts each step for placement/eviction ordering)."""
        state = self.layer_states[layer_idx]
        last_used = np.asarray(state.last_used, dtype=np.float32)
        age = np.where(last_used < 0.0, 0.0, state.step - last_used)
        return (
            self.recent_weight * np.asarray(state.freq, dtype=np.float32)
            + self.ema_weight * state.ema
            + self.router_weight * np.asarray(state.router_score, dtype=np.float32)
            - self.age_weight * age
        )

    def seed_layer_hotness(self, layer_idx: int,
                           weights: dict[int, float]) -> None:
        """Seed per-layer hotness from offline stats (decode resident preload).

        Freshly preloaded hot experts must survive early choose_victim() calls
        before runtime observe() builds up real statistics; without seeding
        they read hotness≈0 and get evicted first, undoing the preload.

        Offline weights are softmax router scores (~1e-2), far smaller than the
        freq term (recent_weight*freq, ~1 per hit) that dominates hotness, so
        writing them into ema/router_score yields hotness ~1e-2 — still evicted
        first against any runtime-hit expert (freq>=1). We therefore min-max
        normalize each layer's weights into [1, recent_window] and write that
        into freq (the dominant term), keeping the raw weight in router_score.
        This makes a preloaded hot expert's initial hotness comparable to (or
        above) a runtime-hit expert, preventing early eviction.
        """
        state = self.layer_states[layer_idx]
        if not weights:
            return
        w_max = max(weights.values()) or 1.0
        for eid, w in weights.items():
            norm = (w / w_max) if w_max > 0 else 0.0
            state.freq[eid] = max(1, round(norm * self.recent_window))
            state.router_score[eid] = float(w)
            state.last_used[eid] = state.step    # zero age penalty

    def layer_step(self, layer_idx: int) -> int:
        return self.layer_states[layer_idx].step

    def _victim_key(self, layer_idx: int, expert_id: int) -> tuple[float, int, int]:
        state = self.layer_states[layer_idx]
        return (self.hotness(layer_idx, expert_id), state.last_used[expert_id], expert_id)


class CoordinatedLRCPolicy:
    """Online cross-layer residency for TP slices or disjoint EP experts.

    This is not the offline O-MoE hotset prefix algorithm. It reuses the old
    LRC's exact *within-layer* scoring and uses a positive layer-wide scale
    for cross-layer comparisons. Router input is absent and its LRC term is
    explicitly zero; next-layer predictions remain a separate fresh hint.

    Workers report cumulative CPU dispatch counts and per-layer invocation
    revisions. A freeze consumes only new counts, without another D2H or a
    per-layer collective. If multiple invocations occur between freezes they
    form one observation: recent_window/EMA/age count coordinated observations,
    not individual tokens or hidden invocation order. An unchanged snapshot
    never advances EMA, age, or the observation window.
    """

    def __init__(
        self, *, layer_count: int, local_experts: int, worker_count: int,
        topk: int, recent_window: int = 32, ema_beta: float = 0.9, recent_weight: float = 1.0,
        ema_weight: float = 0.5, age_weight: float = 0.01,
    ) -> None:
        self.layer_count = layer_count
        self.local_experts = local_experts
        self.worker_count = worker_count
        self.lrc = LRCExpertCachePolicy(
            num_layers=layer_count, num_experts=local_experts,
            cache_size=local_experts, topk=topk, recent_window=recent_window,
            ema_beta=ema_beta, recent_weight=recent_weight,
            ema_weight=ema_weight, router_weight=0.0,
            age_weight=age_weight,
        )
        self._observations = tuple(
            tuple(RoutingObservationState(0, (0,) * local_experts) for _ in range(layer_count))
            for _ in range(worker_count)
        )
        self.priorities: dict[ExpertKey, float] = {
            (layer, expert): 0.0 for layer in range(layer_count) for expert in range(local_experts)
        }

    def observe_snapshots(
        self, observations_by_rank: Sequence[Sequence[RoutingObservation]],
    ) -> dict[ExpertKey, float]:
        """Check snapshot completeness and monotonicity before consuming deltas."""
        if len(observations_by_rank) != self.worker_count:
            raise ValueError("Online routing observations must cover every rank")
        # Each rank stores (revision, cumulative expert counts) in layer order.
        validated_observations: list[tuple[RoutingObservationState, ...]] = []
        for rank, rank_observations in enumerate(observations_by_rank):
            if len(rank_observations) != self.layer_count:
                raise ValueError("Online routing observations must cover every layer")
            observations_by_layer: dict[int, RoutingObservationState] = {}
            for observation in rank_observations:
                layer_idx = observation["layer_idx"]
                if layer_idx >= self.layer_count or layer_idx in observations_by_layer:
                    raise ValueError("Online routing observations have an unknown or duplicate layer")
                revision = observation["revision"]
                # Counts come from the worker's INT64 device table. Keep the
                # completeness and regression checks before updating LRC state.
                expert_counts = observation["counts"]
                if len(expert_counts) != self.local_experts:
                    raise ValueError("Online routing counts must cover every local expert")
                previous_revision, previous_counts = self._observations[rank][layer_idx]
                if (revision < previous_revision or any(map(lt, expert_counts, previous_counts))
                        or (revision == previous_revision and tuple(expert_counts) != previous_counts)):
                    raise ValueError("Online routing observations regressed or changed without an invocation")
                observations_by_layer[layer_idx] = RoutingObservationState(revision, tuple(expert_counts))
            validated_observations.append(
                tuple(observations_by_layer[layer_idx] for layer_idx in range(self.layer_count)))
        for layer_idx in range(self.layer_count):
            if len({rank_observations[layer_idx].revision for rank_observations in validated_observations}) != 1:
                raise ValueError("Online routing invocation revisions differ between workers")
        # All shape, monotonicity and invocation checks precede any LRC update.
        for layer_idx in range(self.layer_count):
            if validated_observations[0][layer_idx].revision != self._observations[0][layer_idx].revision:
                # TP workers report the same routes; count them once.
                count_deltas = [
                    new - old for new, old in zip(
                        validated_observations[0][layer_idx].counts, self._observations[0][layer_idx].counts)
                ]
                self.lrc.observe_global_counts(layer_idx, count_deltas)
        self._observations = tuple(validated_observations)
        self._update_priorities()
        return self.priorities

    def _update_priorities(self) -> None:
        for layer, state in enumerate(self.lrc.layer_states):
            raw = self.lrc.hotness_array(layer).astype(np.float64)
            heat = (self.lrc.recent_weight * sum(state.freq)
                    + self.lrc.ema_weight * float(np.sum(state.ema, dtype=np.float64)))
            age = max((state.step - last for last in state.last_used if last >= 0), default=0)
            # Positive, shared scale preserves every within-layer comparison.
            # max(abs(raw)) only guards float32 rounding in the original LRC.
            scale = max(1.0, heat + self.lrc.age_weight * age, float(np.max(np.abs(raw))))
            if not isfinite(scale) or not np.all(np.isfinite(raw)):
                raise ValueError("Online LRC hotness overflowed its finite normalization scale")
            for expert, priority in enumerate((raw / scale).tolist()):
                self.priorities[layer, expert] = priority

    def build_payload(
        self, *, resident_keys: set[ExpertKey], load_limit: int, prediction_weight: float,
    ) -> ExpertPolicy:
        """Fill granted shared space by online score, identically on every TP worker."""
        candidates = []
        if load_limit:
            candidates = [key for key in self.priorities if key not in resident_keys]
            candidates.sort(key=lambda key: (-self.priorities[key], *key))
        return {
            "load_plan": candidates[:load_limit],
            "prediction_weight": float(prediction_weight),
        }
