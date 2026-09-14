"""Local-routing-consistency policy for expert offload decode paging."""

from collections import deque
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np


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
        snapshot = tuple((int(eid), int(counts[eid])) for eid in active)
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
