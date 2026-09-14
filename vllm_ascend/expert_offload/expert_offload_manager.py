"""Expert Offload Manager — manages CPU-side expert weights and NPU paging."""

import json
import logging
import os
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn.functional as F
import torch_npu
from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context
from vllm.logger import logger

from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.expert_offload.decode_stats import get_decode_stats
from vllm_ascend.expert_offload.expert_predictor import maybe_create_driver
from vllm_ascend.expert_offload.h2d_transfer import (
    H2DCopyTask,
    HostPointerSource,
    TorchCopyH2DTransport,
    create_h2d_transport,
)
from vllm_ascend.expert_offload.lrc_policy import LRCExpertCachePolicy
from vllm_ascend.ops.fused_moe.experts_selector import (
    commit_expert_substitutions,
    maybe_prune_topk_experts,
    plan_expert_substitutions,
    substitute_experts,
    substitute_experts_device,
)
from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ


_SUBSCRIBED_COMPUTE_STREAMS = set()


def get_subscribed_compute_streams() -> set:
    return _SUBSCRIBED_COMPUTE_STREAMS


def _stable_int_checksum(values) -> int:
    """Return a deterministic checksum for small CPU-side integer tensors.

    Python's hash is process-randomized, so it cannot be compared across EP
    ranks. This helper is used only under ``moe_offload_debug`` and avoids an
    extra distributed collective in the decode path. Adding two preserves the
    distinction between the ``-1`` log2phy sentinel and physical slot zero.
    """
    flat = values.reshape(-1).tolist() if hasattr(values, "reshape") else values
    return sum((index + 1) * (int(value) + 2)
               for index, value in enumerate(flat))


def _stable_float_checksum(values, scale: int = 1_000_000) -> int:
    """Deterministic fixed-point checksum for cross-rank debug comparison."""
    flat = values.reshape(-1).tolist() if hasattr(values, "reshape") else values
    return sum((index + 1) * round(float(value) * scale)
               for index, value in enumerate(flat))


def _expert_weight(layer, name: str):
    """Resolve an expert weight tensor, packed-aware.

    ascend w4a8_mxfp4 with ``use_weight_packed=True`` (e.g. Kimi-K3 routed via
    compressed-tensors) stores weights as ``w13_weight_packed`` / ``w2_weight_packed``;
    otherwise plain ``w13_weight`` / ``w2_weight``. Scales (``*_scale``) are never
    packed. Return whichever attribute exists.
    """
    packed = getattr(layer, name + "_packed", None)
    if packed is not None:
        return packed
    return getattr(layer, name)


class ExpertOffloadManager:
    """Singleton manager for expert weight offloading.

    Stores all expert weights on CPU and pages the needed experts to NPU
    during forward based on routing topk_ids.
    """

    _instance: "ExpertOffloadManager | None" = None

    # Parallel weight-load pool. The strided transpose-copy in load_w13/
    # load_w2 is single-threaded (~0.2 GB/s into pinned memory); fanning the
    # ~99k shard copies out over this many workers hits ~2-4 GB/s.
    _LOAD_POOL_WORKERS = 32
    # Bound on in-flight futures before a partial drain (releases owned clones
    # early so transient memory stays small). >> workers, so no starvation.
    # 128 (= 4x workers): for big MoE (Kimi-K3 1.5T) the old 2048 piled ~0.8T
    # of owned clones on top of the 1.5T pinned buffer → host peak ~2.3T (near
    # the 2.9T limit). 128 bounds in-flight clones to ~50G with no throughput
    # loss (pool still runs 32-wide; only the drain point is denser).
    _LOAD_POOL_DRAIN_EVERY = 128
    # Keep the worker barrier dense for host-memory safety, but report loading
    # progress much less frequently. A final aggregate is always logged by
    # _finalize_offload, so small models do not need intermediate progress.
    _LOAD_PROGRESS_LOG_EVERY = 4096
    _DEBUG_EXPERT_SAMPLE_LIMIT = 16

    @classmethod
    def get_instance(cls) -> "ExpertOffloadManager":
        assert cls._instance is not None, "ExpertOffloadManager not initialized"
        return cls._instance

    def __init__(self, vllm_config: VllmConfig):
        from vllm_ascend.ascend_config import get_ascend_config

        self.offload_config = get_ascend_config().expert_offload_config
        # The minimum capacity is the conservative global dispatch threshold;
        # actual weight and placement sizes are resolved per MoE layer.
        self.num_device_experts = min(
            self.offload_config.num_device_experts_list)
        # topk: 标准 MoE 用 num_experts_per_tok;Kimi-K3 用 num_experts_per_token(在 text_config)
        _hf = vllm_config.model_config.hf_config
        _tc = getattr(_hf, "text_config", None)
        self.topk = (
            getattr(_hf, "num_experts_per_tok", None)
            or getattr(_hf, "num_experts_per_token", None)
            or getattr(_tc, "num_experts_per_tok", None)
            or getattr(_tc, "num_experts_per_token", None)
        )
        assert self.topk, ("offload: cannot find num_experts_per_tok/num_experts_per_token "
                           "on hf_config/text_config")
        self.topk = int(self.topk)
        self.offload_threshold = self.num_device_experts // self.topk

        # Multi-card EP offload (stages 1-2). ep_rank/ep_size are resolved
        # lazily on first read because the EP process group is not initialized
        # yet at manager construction time (model_runner.__init__ runs before
        # init_distributed_environment completes).
        self.enable_multi_card = self.offload_config.enable_multi_card
        self._ep_size = 1
        self._ep_rank = 0
        self._ep_info_resolved = False
        # Multi-card decode resident cache: per layer_idx, {slot: expert_id} of
        # the experts currently loaded in THIS rank's device slots. Used to turn
        # the per-step full H2D into skip-on-hit (only load misses) and to log
        # hit/miss. Keyed by slot (the planner assigns expert->slot via log2phy,
        # so a hit = same expert already in the same slot).
        self._mc_resident = {}
        # Two-timescale LRU: per-step local freq tracking + every-N-step gloo
        # all_reduce -> global hotness. Stable-slot placement uses prev_log2phy
        # (keep experts on same rank in their slot) + hotness (order new experts).
        self._mc_prev_log2phy = {}      # layer_idx -> prev step's log2phy (CPU)
        # LRC hotness policy (same one single-card uses: recent freq + EMA +
        # age), fed the GLOBAL active set each step. Built lazily in
        # _gather_global_counts_and_hotness. Replaces the old crude local-freq
        # + 32-step all_reduce hotness, which was stale and had no EMA/age.
        self._mc_lrc = None

        # Per-layer cap on experts actually H2D-loaded by _do_prefetch: only
        # the top-N highest-confidence predicted experts are loaded, the rest
        # are left to update_weights()'s reactive fallback. Clamped to
        # [1, topk]; >topk has no extra effect since the router selects at
        # most topk experts per token.
        self.expert_prefetch_num = self.offload_config.expert_prefetch_num
        # Single-card transfer cap only. _update_weights_multi_card never reads
        # it: multi-card re-plans the layer's whole global placement and loads
        # whatever that implies, so it has no per-layer transfer budget.
        self.prefetch_topk = max(1, min(self.topk, self.expert_prefetch_num))
        # How many tokens to predict from
        self.prefetch_tokens = max(1, self.offload_config.expert_prefetch_tokens)

        # CPU weight buffers (post-transpose format, matching device after
        # process_weights_after_loading):
        #   w13 per expert: [hidden_size, w13_up_dim]
        #   w2 per expert:  [intermediate_size_per_partition, hidden_size]
        self.w13_weights_cpu: list[list[torch.Tensor]] = []
        self.w2_weights_cpu: list[list[torch.Tensor]] = []

        # Registered AscendFusedMoE layers, indexed by moe_instance_id order
        self.moe_layers: list = []

        # CPU buffers for quantized model scale/offset parameters.
        # Keyed by attr_name (e.g. "w13_weight_scale", "w2_weight_offset").
        # Each value is a list of layers, each layer is a list of expert tensors.
        self.scale_cpu_buffers: dict[str, list[list[torch.Tensor]]] = {}
        self.offset_cpu_buffers: dict[str, list[list[torch.Tensor]]] = {}
        self.scale_bias_cpu_buffers: dict[str, list[list[torch.Tensor]]] = {}

        # Temporary per-expert storage for w13 scale/offset shard assembly.
        # Key: (layer_moe_idx, expert_id, attr_name), value: first shard.
        # Scale/offset arrive as w1 + w3 shards; we stash one until the
        # other arrives, then assemble and copy into scale_cpu_buffers.
        self._scale_shard_temp: dict[tuple[int, int, str], torch.Tensor] = {}

        self.num_device_layers = self.offload_config.num_device_layers
        self.num_total_experts = None  # set in init_layer_cpu_buffers
        self.cache_policy: LRCExpertCachePolicy | None = None
        self.cache_requests: list[int] = []
        self.cache_hits: list[int] = []
        self.cache_misses: list[int] = []
        self.cache_calls: list[int] = []
        self.last_hit_experts: list[list[int]] = []
        self.last_miss_experts: list[list[int]] = []
        # decode-statistics collector, resolved in _finalize_offload.
        self._stats = None
        # prefetch -> reactive handoff for the prefetch-accuracy fractions.
        self._prefetch_stats_pending: dict[int, tuple] = {}
        
        # Master debug switch for expert-offload diagnostics — UPDATE-W cache
        # trace, per-prefill-load logs, prefetch/update slot shortfalls.
        # Flipping it on surfaces them at info level (no need for global
        # VLLM_LOGGING_LEVEL=DEBUG).
        self._debug = self.offload_config.moe_offload_debug
        self._prune_activity_logged = False
        self._prune_debug = self.offload_config.experts_pruning_debug
        # Graph/collective diagnostics. Host callbacks may execute on report
        # threads, so keep the counters under a small CPU-only lock. These
        # fields are touched only when moe_offload_debug is enabled and never
        # read device tensors or introduce another distributed collective.
        self._mc_debug_lock = threading.Lock()
        self._mc_debug_schedule_seq = 0
        self._mc_debug_callback_seq = 0
        self._mc_debug_collective_seq = 0
        self._mc_debug_active_callbacks = 0
        self._mc_debug_layer_calls: dict[tuple[int, bool], int] = {}

        # Diagnostic: wall time of the parallel weight-load phase (safetensors
        # → pinned CPU buffers). Logged in _finalize_offload.
        self._weight_load_secs: float = 0.0
        self._weight_load_calls: int = 0

        # Deferred weight-load pool. load_w13/load_w2/_load_scale_shard clone
        # loaded_weight synchronously (while the safetensors mmap is still
        # mapped) and submit the strided transpose-copy to this pool. The
        # deferred copy reads the owned clone, so it stays correct after the
        # safetensors mmap is unmapped (which happens before _finalize_offload).
        # drain_load_pool() is called from _finalize_offload before the buffers
        # are read by process_weights_after_loading().
        self._load_pool: ThreadPoolExecutor | None = None
        self._load_futures: list = []
        self._load_phase_start: float = 0.0
        self._saved_num_threads: int | None = None

        ExpertOffloadManager._instance = self

        self.load_stream = torch_npu.npu.Stream()
        # MemFabric SHARED initialization is collective and needs the EP group,
        # so defer it until the first layer allocates Host expert buffers.
        self.h2d_transport = None
        if not (self.offload_config.h2d_backend == "memfabric"
                and self.enable_multi_card):
            self.h2d_transport = self._create_h2d_transport()
        self._shared_h2d_sources = {}
        self._shared_h2d_sources_ready = False
        logger.info(
            "[EXPERT-OFFLOAD-H2D] backend=%s mode=%s "
            "memfabric_pool_size_gib=%d",
            self.offload_config.h2d_backend,
            ("shared" if self.enable_multi_card else "local")
            if self.offload_config.h2d_backend == "memfabric" else "copy",
            self.offload_config.memfabric_pool_size_gib,
        )

        self._init_prefill_pool_state()
        self._is_prefetch: bool = False
        self._init_prefetch_state()

    def _init_prefill_pool_state(self) -> None:
        """Prefill-pool attribute init (ndl layers × all experts on NPU)."""
        # Prefill pool: ndl layers × all experts on NPU, shared round-robin
        self._prefill_w13: list[torch.Tensor] = []
        self._prefill_w2: list[torch.Tensor] = []
        self._prefill_w13_scale: list[torch.Tensor] = []       # W8A8 / W4A8_DYNAMIC
        self._prefill_w13_scale_fp32: list[torch.Tensor] = []   # W8A8
        self._prefill_w13_offset: list[torch.Tensor] = []       # W8A8
        self._prefill_w2_scale: list[torch.Tensor] = []         # W8A8 / W4A8_DYNAMIC
        self._prefill_w2_offset: list[torch.Tensor] = []        # W8A8
        # W4A8_DYNAMIC scale_bias (float32), per-channel new_quant_version only.
        # Allocated lazily in create_prefill_pool when the layer has
        # w13_scale_bias / w2_scale_bias parameters.
        self._prefill_w13_scale_bias: list[torch.Tensor] = []
        self._prefill_w2_scale_bias: list[torch.Tensor] = []
        self._prefill_log2phy: torch.Tensor = None              # identity [0..127]
        self._prefill_initialized: bool = False
        self._skip_prefill: bool = False  # set during profile runs

    def _init_prefetch_state(self) -> None:
        """Next-layer expert-prefetch infrastructure init."""
        # Next-layer expert prefetch infrastructure
        self._prefetch_stream = torch_npu.npu.Stream()
        # NPU copy of gate weights for graph-capturable on-device prediction
        # (predict_next_layer_experts_npu). Kept in fp32.
        self._gate_weights_npu: list[torch.Tensor | None] = []

        # Prefetch state: _prefetch_state_lock guards _prefetch_layer_npu_event,
        # which is shared by the forward thread and the graph host callback.
        self._prefetch_state_lock = threading.Lock()
        self._prefetch_layer_npu_event: dict[int, torch_npu.npu.Event] = {}

        # Pinned CPU staging buffer for graph-mode prefetch: trigger_next_
        # layer_prefetch stages the next layer's log2phy here with
        # non_blocking D2H around the host callback, mirroring update_weights
        # (blocking .cpu() on a live graph tensor would deadlock on replay).
        # Allocated lazily in _finalize_offload (num_total_experts is only
        # known after MoE layers register).
        self._prefetch_log2phy_h: torch.Tensor | None = None
        self._prefetch_log2phy_np = None

        # MoE-layer indices whose routing comes from a tid2eid table.
        # Filled in _finalize_offload; used by trigger_next_layer_prefetch to
        # keep the hash targets on this driver even when a trained predictor is configured.
        self._hash_layer_indices: frozenset[int] = frozenset()

        # the trained-predictor prefetch method, or None to leave
        # every target to the heuristic ("fate") method. Created HERE — before
        # the model is built — so decoder layers can resolve their capture site during construction
        self.expert_predictor = maybe_create_driver(self)
        
        # Prefetch-stall timing. Per-layer device event pairs bracketing the
        # join in update_weights[_multi_card]
        self._pf_wait_timing = bool(
            self.offload_config.expert_prefetch_wait_timing)
        self._pf_wait_begin: list = []
        self._pf_wait_end: list = []
        self._pf_wait_armed: list[bool] = []
        # Measured cost of the instrument itself (ms), subtracted from every
        # sample so pf_wait reports the stall and not the two event records
        # that bracket it. Calibrated in _finalize_offload.
        self._pf_wait_bias = 0.0
        self._pf_wait_ok = 0        # successful elapsed_time reads
        self._pf_wait_fail = 0      # failed reads; first one logs
        self._pf_wait_probed = False

    def _resolve_ep_info(self) -> None:
        """Lazily resolve ep_rank/ep_size from the EP group on first access.

        No-op (stays ep_size=1, ep_rank=0) when ``enable_multi_card`` is False,
        so the single-card path is unchanged.
        """
        if self._ep_info_resolved:
            return
        if self.enable_multi_card:
            from vllm.distributed.parallel_state import get_ep_group
            ep_group = get_ep_group()
            self._ep_size = ep_group.world_size
            self._ep_rank = ep_group.rank_in_group
        self._ep_info_resolved = True

    @property
    def ep_size(self) -> int:
        self._resolve_ep_info()
        return self._ep_size

    @property
    def ep_rank(self) -> int:
        self._resolve_ep_info()
        return self._ep_rank

    def _create_h2d_transport(self):
        enable_shared = (
            self.offload_config.h2d_backend == "memfabric"
            and self.enable_multi_card)
        return create_h2d_transport(
            self.offload_config.h2d_backend,
            device_id=torch_npu.npu.current_device(),
            memfabric_pool_size_gib=(
                self.offload_config.memfabric_pool_size_gib),
            memfabric_log_level=self.offload_config.memfabric_log_level,
            enable_multi_card=enable_shared,
            world_size=self.ep_size if enable_shared else 1,
            rank_id=self.ep_rank if enable_shared else 0,
        )

    # ------------------------------------------------------------------ #
    #  Lifecycle: called during model init and after weight loading       #
    # ------------------------------------------------------------------ #

    def num_device_experts_for_layer(self, layer_idx: int) -> int:
        """Per-layer device-expert buffer size (delegates to offload config).

        The config is the single source of truth: a scalar broadcasts, a list
        indexes by MoE-layer registration order.
        """
        return self.offload_config.num_device_experts_for_layer(layer_idx)

    def init_layer_cpu_buffers(self, layer, layer_moe_idx: int):
        """Allocate CPU weight + scale/offset buffers for one MoE layer.

        Called from AscendFusedMoE.__init__ after device tensors are set up,
        so CPU buffers exist before the safetensors weight loader runs.
        """
        ntotal = layer.global_num_experts
        if self.num_total_experts is None:
            self.num_total_experts = ntotal
        assert ntotal == self.num_total_experts, \
            f"MoE layers must have same expert count: {ntotal} vs {self.num_total_experts}"

        _w13 = _expert_weight(layer, "w13_weight")
        _w2 = _expert_weight(layer, "w2_weight")
        params_dtype = _w13.dtype
        w13_shape = (_w13.shape[2], _w13.shape[1])
        w2_shape = (_w2.shape[2], _w2.shape[1])

        use_shard = self.offload_config.shard_per_rank
        if use_shard:
            # shard-per-rank: each rank holds ONLY its EP shard of weight
            # experts (ntotal // ep_size), as per-expert pinned tensors (like
            # the non-shared path, but shard-sized). No mmap, no cross-process
            # sharing, no staging — H2D reads each expert's own pinned storage.
            # Scales/offsets are sharded the same way (see
            # _init_layer_scale_buffers). Placement must
            # be constrained to EP ownership (expert e → rank e // shard) so a
            # rank only loads experts it actually holds.
            shard = ntotal // max(1, self.ep_size)
            self._shard_size = shard
            self._shard_base = self.ep_rank * shard
            w13_list = [
                self._allocate_expert_host_tensor(w13_shape, params_dtype)
                for _ in range(shard)
            ]
            w2_list = [
                self._allocate_expert_host_tensor(w2_shape, params_dtype)
                for _ in range(shard)
            ]
            self.w13_weights_cpu.append(w13_list)
            self.w2_weights_cpu.append(w2_list)
        else:
            w13_list = [
                self._allocate_expert_host_tensor(w13_shape, params_dtype)
                for _ in range(ntotal)
            ]
            w2_list = [
                self._allocate_expert_host_tensor(w2_shape, params_dtype)
                for _ in range(ntotal)
            ]
            self.w13_weights_cpu.append(w13_list)
            self.w2_weights_cpu.append(w2_list)

        # Per-expert storage size (works for both list[0] and big_tensor[0]).
        first_w13 = self.w13_weights_cpu[-1][0]
        first_w2 = self.w2_weights_cpu[-1][0]
        self.w13_expert_size_bytes = first_w13.nelement() * first_w13.element_size()
        self.w2_expert_size_bytes = first_w2.nelement() * first_w2.element_size()

        # Scale / offset CPU buffers (W8A8)
        self._init_layer_scale_buffers(layer, layer_moe_idx, ntotal)

        self.moe_layers.append(layer)
        # If the cache policy was already built (this layer is registered
        # after _finalize_offload, e.g. an MTP draft MoE layer loaded after
        # the target model), extend the policy and per-layer stats so LRC
        # eviction applies uniformly to target and draft layers. Keeps the
        # invariant that every registered MoE layer has a matching cache
        # state and stats slot.
        self._extend_cache_for_layer()
        # Same post-finalize path for prefetch gate weights: register this
        # layer's gate so _gate_weights_npu stays index-aligned with
        # moe_layers. Without it, len(moe_layers) > len(_gate_weights_npu)
        # and predict_next_layer_experts_npu returns None for the boundary
        # layer. Pre-finalize layers are covered in bulk by
        # register_gate_weights(); the cache_policy sentinel skips them.
        self._register_layer_gate(layer)

    def _extend_cache_for_layer(self):
        """Grow cache_policy and stats lists to cover one more MoE layer.

        No-op before _finalize_offload has built the policy (the target
        layers are all covered in one shot there). Afterwards each newly
        registered layer (e.g. the MTP draft layer) gets its own fresh
        LRC state, so draft-layer hotness is tracked independently from
        the target layers.
        """
        if self.cache_policy is None:
            return
        new_idx = self.cache_policy.add_layer()
        self.cache_requests.append(0)
        self.cache_hits.append(0)
        self.cache_misses.append(0)
        self.cache_calls.append(0)
        self.last_hit_experts.append([])
        self.last_miss_experts.append([])
        logger.info(
            "[EXPERT-OFFLOAD-CACHE] extended cache policy to layer=%d "
            "(total_layers=%d)",
            new_idx, len(self.cache_policy.layer_states))

    @staticmethod
    def _cpu_tensor_storage_bytes(tensors) -> int:
        """Count unique CPU tensor storages in a nested list structure."""
        total = 0
        seen: set[tuple[int, int]] = set()

        def visit(value):
            nonlocal total
            if isinstance(value, torch.Tensor):
                storage = value.untyped_storage()
                key = (storage.data_ptr(), storage.nbytes())
                if key not in seen:
                    seen.add(key)
                    total += storage.nbytes()
            elif isinstance(value, dict):
                for child in value.values():
                    visit(child)
            elif isinstance(value, (list, tuple)):
                for child in value:
                    visit(child)

        visit(tensors)
        return total

    @staticmethod
    def _host_memory_snapshot() -> tuple[int | None, int | None, int | None]:
        """Return Linux process RSS/HWM and system available bytes."""
        rss = hwm = available = None
        try:
            with open("/proc/self/status", encoding="utf-8") as status_file:
                for line in status_file:
                    if line.startswith("VmRSS:"):
                        rss = int(line.split()[1]) * 1024
                    elif line.startswith("VmHWM:"):
                        hwm = int(line.split()[1]) * 1024
        except (OSError, ValueError, IndexError):
            pass
        try:
            with open("/proc/meminfo", encoding="utf-8") as meminfo_file:
                for line in meminfo_file:
                    if line.startswith("MemAvailable:"):
                        available = int(line.split()[1]) * 1024
                        break
        except (OSError, ValueError, IndexError):
            pass
        return rss, hwm, available

    def _log_cpu_expert_memory(self) -> None:
        """Log per-rank expert buffers and host memory for replica auditing."""
        if not self._debug:
            return
        weights_bytes = self._cpu_tensor_storage_bytes(
            (self.w13_weights_cpu, self.w2_weights_cpu))
        quant_bytes = self._cpu_tensor_storage_bytes((
            self.scale_cpu_buffers,
            self.offset_cpu_buffers,
            self.scale_bias_cpu_buffers,
        ))
        expert_bytes = weights_bytes + quant_bytes
        rss, hwm, available = self._host_memory_snapshot()
        cpu_mode = ("sharded" if self.offload_config.shard_per_rank
                    else "replicated")
        experts_per_layer = (
            len(self.w13_weights_cpu[0]) if self.w13_weights_cpu else 0)
        replica_factor = 1 if self.offload_config.shard_per_rank else self.ep_size

        def mib(value):
            return None if value is None else round(value / (1024 ** 2), 1)

        logger.info(
            "[CPU_MEM] rank=%s/%s pid=%s cpu_mode=%s "
            "experts_per_layer=%s global_experts=%s layers=%s "
            "expert_buffers_mib=%.1f weights_mib=%.1f quant_mib=%.1f "
            "process_rss_mib=%s process_hwm_mib=%s host_available_mib=%s "
            "expected_full_weight_copies_across_ep=%s",
            self.ep_rank, self.ep_size, os.getpid(), cpu_mode,
            experts_per_layer, self.num_total_experts, len(self.moe_layers),
            mib(expert_bytes), mib(weights_bytes), mib(quant_bytes),
            mib(rss), mib(hwm), mib(available), replica_factor)

    def _init_layer_scale_buffers(self, layer, layer_moe_idx: int,
                                   ntotal: int):
        """Allocate CPU scale/offset buffers for a single MoE layer."""
        # shard-per-rank: like the weight buffers, each rank holds only its EP
        # shard (ntotal // ep_size) of scale/offset/scale_bias, indexed LOCALLY.
        # global<->local via _shard_local (mirrors the weight path); non-shard
        # (single-card) keeps the full ntotal, global==local.
        use_shard = self.offload_config.shard_per_rank
        nalloc = (ntotal // max(1, self.ep_size)) if use_shard else ntotal
        attr_specs = [
            ("scale_cpu_buffers", "w13_weight_scale"),
            ("scale_cpu_buffers", "w2_weight_scale"),
            ("offset_cpu_buffers", "w13_weight_offset"),
            ("offset_cpu_buffers", "w2_weight_offset"),
            ("scale_bias_cpu_buffers", "w13_scale_bias"),
            ("scale_bias_cpu_buffers", "w2_scale_bias"),
        ]
        for buffer_dict_name, attr_name in attr_specs:
            if not hasattr(layer, attr_name):
                continue
            dev_tensor = getattr(layer, attr_name)
            dtype = dev_tensor.dtype
            if "scale_bias" in attr_name:
                per_expert_shape = tuple(dev_tensor.shape[1:])
            elif dtype.itemsize == 1:
                from vllm_ascend.quantization.methods.w4a8_mxfp4 import (
                    apply_mxfp4_weight_scale_layout)
                dtype = torch.uint8
                per_expert_shape = tuple(
                    apply_mxfp4_weight_scale_layout(dev_tensor[0].view(torch.uint8)).shape)
            else:
                per_expert_shape = (dev_tensor[0].numel(),)
            buffer_dict: dict = getattr(self, buffer_dict_name)
            if attr_name not in buffer_dict:
                buffer_dict[attr_name] = []
            buffers = buffer_dict[attr_name]
            while len(buffers) <= layer_moe_idx:
                buffers.append([])
            for _ in range(nalloc):
                buffers[layer_moe_idx].append(
                    self._allocate_expert_host_tensor(
                        per_expert_shape, dtype))

    def _finalize_offload(self, model):
        """Post-weight-loading finalization.

        Must be called AFTER get_model() has finished loading all weights.
        Performs NZ format conversion, cache policy init, forward buffer
        init, fp32 scale refresh, prefill pool creation, and gate weight
        registration.
        """
        if not self.moe_layers:
            return
        # Barrier: ensure all deferred load_w13/load_w2/_load_scale_shard
        # copies have landed before process_weights_after_loading reads them.
        self.drain_load_pool()
        t0 = time.perf_counter()
        logger.info(
            "[OFFLOAD] weight load (safetensors→CPU buffer): %.1fs over %d calls",
            self._weight_load_secs, self._weight_load_calls)
        t1 = time.perf_counter()
        self.process_weights_after_loading()
        self._publish_shared_h2d_sources()
        t2 = time.perf_counter()

        num_moe_layers = len(self.moe_layers)
        # Validate a per-layer num_device_experts list covers every MoE layer.
        # Scalars and single-element lists broadcast; a multi-element list is
        # indexed by MoE-layer registration order and must cover at least the
        # layers registered so far. It may be longer to also cover MoE layers
        # that register AFTER _finalize_offload — notably the MTP draft MoE,
        # which loads with the drafter after the target model is finalized, so
        # the count seen here (target layers only) is smaller than the final
        # total. Requiring equality would reject the extra draft-layer entry.
        nde_list = self.offload_config.num_device_experts_list
        self.offload_config.validate_num_moe_layers(num_moe_layers)
        if self._debug:
            logger.info(
                "[OFFLOAD] num_device_experts per layer (n_layers=%d): %s",
                num_moe_layers, nde_list if len(nde_list) > 1 else nde_list[0])
        if self.offload_config.cache_policy_enabled:
            self.cache_requests = [0 for _ in range(num_moe_layers)]
            self.cache_hits = [0 for _ in range(num_moe_layers)]
            self.cache_misses = [0 for _ in range(num_moe_layers)]
            self.cache_calls = [0 for _ in range(num_moe_layers)]
            self.last_hit_experts = [[] for _ in range(num_moe_layers)]
            self.last_miss_experts = [[] for _ in range(num_moe_layers)]
            self.cache_policy = LRCExpertCachePolicy(
                num_layers=num_moe_layers,
                num_experts=self.num_total_experts,
                # cache_size is informational only (LRC eviction keys on
                # hotness, not slot count). Use the representative min; the
                # real per-layer slot count is each layer's device weight size,
                # set per layer via the expert_map_offload.
                cache_size=self.num_device_experts,
                topk=self.topk,
                recent_window=self.offload_config.cache_recent_window,
                ema_beta=self.offload_config.cache_ema_beta,
                recent_weight=self.offload_config.cache_recent_weight,
                ema_weight=self.offload_config.cache_ema_weight,
                router_weight=self.offload_config.cache_router_weight,
                age_weight=self.offload_config.cache_age_weight,
            )

        # Hash-routed layers are identified by their gate carrying a
        # tid2eid table — the same test predict_next_layer_experts_npu uses —
        # so no new config key is needed and the notion cannot drift.
        self._hash_layer_indices = frozenset(
            index for index, moe_layer in enumerate(self.moe_layers)
            if getattr(getattr(moe_layer, "gate", None), "tid2eid", None)
            is not None
        )

        # register the MoE-layer topology with the decode-statistics collector.
        self._stats = get_decode_stats()
        if self._stats is not None:
            self._stats.set_topology(num_moe_layers,
                                     set(self._hash_layer_indices),
                                     multi_card=self.enable_multi_card)
            logger.info(
                "[DECODE-STATS] topology: moe_layers=%d hash_layers=%s "
                "multi_card=%s", num_moe_layers,
                sorted(self._hash_layer_indices), self.enable_multi_card)
        t3 = time.perf_counter()

        ntotal = self.num_total_experts
        self.topk_ids_h = torch.zeros(
            [self.offload_threshold, self.topk],
            dtype=torch.int32, device="cpu", pin_memory=True)
        # pre-substitution routed ids. Substitution now runs on the NPU
        # before topk_ids_h is staged, so the host callback can no longer clone
        # the "before" ids out of topk_ids_h.
        self.topk_ids_gt_h = None
        if self.offload_config.expert_substitution_enabled:
            self.topk_ids_gt_h = torch.zeros(
                [self.offload_threshold, self.topk],
                dtype=torch.int32, device="cpu", pin_memory=True)
        self.topk_weights_h = torch.zeros(
            [self.offload_threshold, self.topk],
            dtype=torch.float32, device="cpu", pin_memory=True)
        self.prune_debug_h = torch.zeros(
            [self.offload_threshold, 3, self.topk],
            dtype=torch.int32,
            device="cpu",
            pin_memory=True,
        )
        self.router_logits_h = None
        self.e_score_correction_bias_h = None
        if self.offload_config.expert_substitution_enabled:
            self.router_logits_h = torch.zeros(
                [self.offload_threshold, ntotal],
                dtype=torch.float32, device="cpu", pin_memory=True)
            self.e_score_correction_bias_h = torch.zeros(
                ntotal, dtype=torch.float32, device="cpu", pin_memory=True)
        # Per-rank active-token mask (1=real, 0=pad), mirrored to pinned CPU so
        # the multi-card host callback can drop pad rows before counting. Under
        # single-batch TP, ranks past the real-token count route zero-hidden
        # PAD tokens whose topk is garbage; without this filter they pollute
        # global_counts (placement), the LRU freq, and hit/miss stats.
        self.mc2_mask_h = torch.zeros(
            self.offload_threshold, dtype=torch.int32,
            device="cpu", pin_memory=True)
        self.log2phy_h = torch.zeros(ntotal, dtype=torch.int32,
                                     device='cpu', pin_memory=True)
        self.log2phy_np = self.log2phy_h.numpy()
        t4 = time.perf_counter()

        self.refresh_fp32_scales()
        t5 = time.perf_counter()
        self._preload_hot_experts()
        t_hot = time.perf_counter()
        self.create_prefill_pool()
        t6 = time.perf_counter()
        if self.offload_config.expert_prefetch_enabled:
            self.register_gate_weights(model)
            # Pinned staging buffer for graph-mode prefetch: trigger_next_
            # layer_prefetch stages the next layer's log2phy here with
            # non_blocking D2H before launching the host callback, mirroring
            # update_weights (blocking .cpu() on a live graph tensor would
            # deadlock during replay).
            self._prefetch_log2phy_h = torch.zeros(
                self.num_total_experts, dtype=torch.int32,
                device='cpu', pin_memory=True)
            self._prefetch_log2phy_np = self._prefetch_log2phy_h.numpy()
            # load the trained head and allocate its buffers —
            # after every weight has landed, and before graph capture, because
            # the head's parameters must be resident when the graph is
            # recorded and the model geometry is only known once the model exists.
            if self.expert_predictor is not None:
                self.expert_predictor.finalize(model)
            if self._pf_wait_timing:
                # One pair per MoE layer, allocated ONCE and re-recorded every
                # step. Creating them per call would add 2 events per layer per
                # capture size and risk the 207008 stream/event ceiling; reusing
                # them keeps the graph's event-record nodes pointing at stable
                # handles, which is what makes elapsed_time meaningful on replay.
                n = len(self.moe_layers)
                self._pf_wait_begin = [
                    torch_npu.npu.Event(enable_timing=True) for _ in range(n)]
                self._pf_wait_end = [
                    torch_npu.npu.Event(enable_timing=True) for _ in range(n)]
                self._pf_wait_armed = [False] * n
                # Calibrate the bracket's own cost, and validate elapsed_time
                # HERE — before any graph is captured. A failure at this point
                # disables recording cleanly
                probe_a = torch_npu.npu.Event(enable_timing=True)
                probe_b = torch_npu.npu.Event(enable_timing=True)
                probe_stream = torch_npu.npu.current_stream()
                try:
                    samples = []
                    for _ in range(16):
                        probe_a.record(probe_stream)
                        probe_b.record(probe_stream)
                        probe_b.synchronize()
                        samples.append(float(probe_a.elapsed_time(probe_b)))
                    self._pf_wait_bias = min(samples)
                    logger.info(
                        "[PREFETCH-WAIT] timing enabled: %d event pairs; "
                        "bracket cost %.4f ms is subtracted from every sample. "
                        "pf_wait measures the compute stream's block at the "
                        "prefetch join, so TPOT from this run is NOT a baseline",
                        n, self._pf_wait_bias)
                except Exception:
                    self._pf_wait_timing = False
                    self._pf_wait_begin = []
                    self._pf_wait_end = []
                    self._pf_wait_armed = []
                    logger.warning(
                        "[PREFETCH-WAIT] elapsed_time is not usable on this "
                        "build; prefetch-stall timing disabled and NOTHING will "
                        "be recorded into the graph. The run is otherwise "
                        "unaffected.", exc_info=True)
        t7 = time.perf_counter()
        self._log_cpu_expert_memory()
        logger.info(
            "[OFFLOAD] finalize breakdown: process_weights=%.1fs "
            "cache_policy=%.1fs buffers=%.1fs init_device=%.1fs "
            "hot_preload=%.1fs prefill_pool=%.1fs gate=%.1fs | total=%.1fs",
            t2 - t1, t3 - t2, t4 - t3, t5 - t4, t_hot - t5, t6 - t_hot,
            t7 - t6, t7 - t0)

    def process_weights_after_loading(self):
        """Convert resident CPU expert buffers to the on-device weight format.

        W8A8 (int8): fractal NZ cast on the transpose-after buffer layout
        (the device path transposes first, then casts NZ).
        W4A8_DYNAMIC (int8 cpu, int32 device): mirror the device path which
        transposes, casts NZ, then packs 4 int8 → 1 int32.
        W4A8_MXFP (uint8): mirror the device process_weights_after_loading,
        which casts29 (mxfp4) on the *pre-transpose* shape and then
        transposes — so we restore the pre-transpose shape first, cast, and
        transpose back to match the device slot layout byte-for-byte.

        After this runs each w13/w2 CPU tensor still reports its original
        shape but its storage holds on-device-format bytes — a "liar tensor".
        Touch it only via untyped_storage() slicing, never through the tensor
        view. No-op for non-quantized (other dtype) models.
        """
        first_w13 = self.w13_weights_cpu[0][0]
        first_layer = self.moe_layers[0]
        # Detect W4A8_DYNAMIC by scale dtype: process_scale converts float32
        # to int64 on the device for both modelslim and compressed_tensors
        # paths. The weight dtype is NOT a reliable signal — modelslim leaves
        # it as int8 (no pack_to_int32) while compressed_tensors packs to
        # int32. Without this check, modelslim W4A8 would be misrouted to the
        # W8A8 branch, skipping scale encoding and producing garbled output.
        is_w4a8 = (hasattr(first_layer, 'w13_weight_scale') and
                   first_layer.w13_weight_scale.dtype == torch.int64)
        if first_w13.dtype == torch.int8:
            first_dev = _expert_weight(first_layer, "w13_weight")
            if first_dev.dtype == torch.int32:
                # compressed_tensors W4A8: weight packed to int32
                self._cast_cpu_weights_to_device_format(w4a8_dynamic=True)
            else:
                # W8A8 or modelslim W4A8: weight is int8, just NZ cast
                self._cast_cpu_weights_to_device_format(mxfp4=False)
            # Scale encoding for W4A8 (both modelslim and compressed_tensors).
            # Must run AFTER _cast_cpu_weights_to_device_format so the CPU
            # buffers are already in device byte layout.
            if is_w4a8:
                self._process_scale_bias_cpu_buffers()
                self._encode_w4a8_dynamic_weight_scales()
        elif first_w13.dtype == torch.uint8:
            self._cast_cpu_weights_to_device_format(mxfp4=True)
            # W4A8_MXFP: also stamp the on-device expert weight slots with
            # format-29 (NZ) so the format METADATA matches the NZ bytes that
            # _cast_cpu_weights_to_device_format produced in the CPU buffer and
            # that decode/prefill H2D writes into the slots at runtime. Without
            # this the slot stays base-format; the fused-swiglu GMM reads it via
            # raw storage (V4 works), but the SiTU raw npu_grouped_matmul path
            # (Kimi-K3) checks the format and rejects fp4_e2m1 on base format
            # (AclNN EZ1001 / error 161002). Bytes set here are overwritten by
            # H2D; only the format persists.
            self._cast_device_slots_to_mxfp4_nz()
        # else: non-quantized model, no-op

    def _cast_cpu_weights_to_device_format(self, mxfp4: bool = False,
                                            w4a8_dynamic: bool = False):
        """Relayout resident CPU w13/w2 expert buffers into the device format.

        NZ (W8A8) and format-29 mxfp4 (W4A8_MXFP) are equal-length relayouts,
        so per-expert on-device bytes == nelement * element_size; we still
        recompute from the cast storage to stay correct if that ever changes.

        W4A8_DYNAMIC mirrors the device path: transpose → NZ cast →
        pack 4 int8 into 1 int32 (view as int32).  The storage size is
        preserved (4× fewer elements at 4× element size), so the CPU buffer
        can hold the packed bytes without reallocation.
        """
        num_moe_layers = len(self.w13_weights_cpu)
        num_experts = len(self.w13_weights_cpu[0])
        use_shard = self.offload_config.shard_per_rank
        for layer_id in range(num_moe_layers):
            w13 = torch.stack(self.w13_weights_cpu[layer_id]).to('npu')
            w2 = torch.stack(self.w2_weights_cpu[layer_id]).to('npu')
            if mxfp4:
                w13 = w13.transpose(1, 2).contiguous()
                w2 = w2.transpose(1, 2).contiguous()
                w13 = torch_npu.npu_format_cast(
                    w13.view(torch.uint8), 29,
                    customize_dtype=torch.float8_e4m3fn,
                    input_dtype=torch_npu.float4_e2m1fn_x2,
                )
                w2 = torch_npu.npu_format_cast(
                    w2.view(torch.uint8), 29,
                    customize_dtype=torch.float8_e4m3fn,
                    input_dtype=torch_npu.float4_e2m1fn_x2,
                )
                w13 = w13.transpose(1, 2)
                w2 = w2.transpose(1, 2)
            elif w4a8_dynamic:
                # CPU buffer is already (E, H, dim2) — _copy_w13_shard stored
                # owned.t(), matching the post-transpose device layout. The
                # device path (process_weights_after_loading_modelslim) does
                # transpose(1,2) → NZ cast → pack_to_int32, where transpose
                # converts (E, dim2, H) → (E, H, dim2). Since the CPU buffer
                # is already (E, H, dim2), we NZ-cast directly — NO extra
                # transpose. A double transpose here would apply NZ blocking
                # to (dim2, H) instead of (H, dim2), producing wrong block
                # layout and garbled output.
                w13 = torch_npu.npu_format_cast(w13, ACL_FORMAT_FRACTAL_NZ)
                w2 = torch_npu.npu_format_cast(w2, ACL_FORMAT_FRACTAL_NZ)
                w13 = w13.view(torch.int32)
                w2 = w2.view(torch.int32)
            else:
                w13 = torch_npu.npu_format_cast(w13, ACL_FORMAT_FRACTAL_NZ)
                w2 = torch_npu.npu_format_cast(w2, ACL_FORMAT_FRACTAL_NZ)
            w13_storage = w13.untyped_storage()
            w2_storage = w2.untyped_storage()
            per_w13 = w13_storage.nbytes() // num_experts
            per_w2 = w2_storage.nbytes() // num_experts
            self.w13_expert_size_bytes = per_w13
            self.w2_expert_size_bytes = per_w2
            for local_i in range(num_experts):
                # _expert_dst_storage takes a GLOBAL eid (shard-per-rank
                # remaps it to the local slot); w13/w2_storage are indexed by
                # the local position in the stacked tensor (== global id when
                # not sharded).
                geid = (self._shard_base + local_i) if use_shard else local_i
                self._expert_dst_storage(layer_id, geid, 'w13').copy_(
                    w13_storage[local_i * per_w13 : (local_i + 1) * per_w13]
                )
                self._expert_dst_storage(layer_id, geid, 'w2').copy_(
                    w2_storage[local_i * per_w2 : (local_i + 1) * per_w2]
                )

    def _cast_device_slots_to_mxfp4_nz(self):
        """Stamp on-device W4A8_MXFP expert weight slots with format-29 (NZ).

        The device slot is already transposed by process_weights (offload
        branch). Mirror the non-offload cast-then-transpose — and the CPU
        relayout in _cast_cpu_weights_to_device_format — by transposing back
        to the original shape, casting 29, then transposing forward again, so
        the slot's format-29 layout is transpose(cast(original)). That is the
        form npu_grouped_matmul's NZ kernel accepts for fp4_e2m1: it infers
        transposeWeight from the layout (EZ1001 if base format, EZ0026 if the
        cast landed on the transposed shape). Slot bytes are refreshed by
        decode/prefill H2D at runtime (CPU buffer holds the matching NZ bytes);
        this call fixes the format + layout metadata.

        Fallback: some NPU devices/drivers only allow base format
        (allow_internal_format=False) and reject the format-29 cast at the ACL
        layer (device error 361001). This call only stamps format *metadata* on
        the resident slots — the bytes themselves are refreshed by decode/prefill
        H2D at runtime, and the fused-swiglu GMM path (DeepSeek-V4) reads them
        via raw storage, so skipping it is safe for V4. The SiTU raw
        npu_grouped_matmul path (Kimi-K3) checks the format and would reject
        fp4_e2m1 on base format (EZ1001); such models are unsupported on these
        devices. On the first cast failure we warn once and return, leaving the
        slots in base format. w.data is reassigned only after a successful cast,
        so a raised exception never leaves a slot half-mutated.
        """
        for layer in self.moe_layers:
            for name in ("w13_weight", "w2_weight"):
                w = _expert_weight(layer, name)
                if w is None:
                    continue
                d = w.data.transpose(1, 2).contiguous()
                try:
                    d = torch_npu.npu_format_cast(
                        d.view(torch.uint8), 29,
                        customize_dtype=torch.float8_e4m3fn,
                        input_dtype=torch_npu.float4_e2m1fn_x2,
                    )
                except RuntimeError as e:
                    logger.warning(
                        "[OFFLOAD] npu_format_cast to format-29 (mxfp4 NZ) "
                        "failed on layer slot %r; leaving expert slots in base "
                        "format. Safe for DeepSeek-V4 (fused-swiglu GMM reads "
                        "raw storage); breaks the Kimi-K3 SiTU "
                        "npu_grouped_matmul path. Error: %s", name, e)
                    return
                w.data = d.transpose(1, 2)

    def _process_scale_bias_cpu_buffers(self):
        """Apply update_bias transformation to scale_bias CPU buffers.

        Mirrors the device-side update_bias for W4A8_DYNAMIC new_quant_version:
        w13_scale_bias: (D1, 1) -> transpose -> (1, D1) -> sum(axis=0) -> (D1,)
        w2_scale_bias: (D1, D2) -> transpose -> (D2, D1) -> sum(axis=0) -> (D1,)
        """
        for attr_name, layer_buffers in self.scale_bias_cpu_buffers.items():
            for layer_idx, expert_buffers in enumerate(layer_buffers):
                new_buffers = []
                for buf in expert_buffers:
                    transformed = buf.transpose(0, 1).contiguous().sum(dim=0)
                    backend_buffer = self._allocate_expert_host_tensor(
                        transformed.shape, transformed.dtype)
                    backend_buffer.copy_(transformed)
                    new_buffers.append(backend_buffer)
                layer_buffers[layer_idx] = new_buffers

    def _encode_w4a8_dynamic_weight_scales(self):
        """Encode W4A8_DYNAMIC weight_scale CPU buffers to device int64 format.

        The safetensors checkpoint stores ``w13_weight_scale`` /
        ``w2_weight_scale`` as float32 tensors, but the device-side
        ``AscendW4A8DynamicFusedMoEMethod.process_scale`` reinterprets the
        float32 bytes as uint32 and zero-extends to int64 before storing it
        on the NPU. The decode-path H2D ``copy_`` therefore must write
        int64-encoded bytes — copying raw float32 into an int64 device tensor
        would corrupt the kernel's scale decoding.

        This method mirrors the per-channel branch of ``process_scale``
        (the only branch supported by expert offload today): float32 →
        uint32 bit-reinterpret → int64 zero-extension. Each expert buffer is
        encoded independently (per-channel encoding is element-wise), so the
        transformation is applied per-expert without cross-expert ops.

        After this runs the CPU buffer dtype changes from float32 to int64,
        matching ``layer.w13_weight_scale.dtype`` on the NPU.
        """
        import numpy as np
        for attr_name in ("w13_weight_scale", "w2_weight_scale"):
            if attr_name not in self.scale_cpu_buffers:
                continue
            for layer_idx, expert_buffers in enumerate(
                    self.scale_cpu_buffers[attr_name]):
                encoded_buffers = []
                for buf in expert_buffers:
                    # buf: float32, shape per-expert (e.g. (2*IN,) for w13)
                    scale_np = np.ascontiguousarray(
                        buf.cpu().numpy()).astype(np.float32)
                    # Bit-reinterpret float32 bytes as uint32, then
                    # zero-extend to int64 — identical to device process_scale
                    # per-channel branch.
                    scale_np.dtype = np.uint32
                    encoded = scale_np.astype(np.int64)
                    encoded_tensor = torch.from_numpy(np.ascontiguousarray(
                        encoded.copy()))
                    encoded_buf = self._allocate_expert_host_tensor(
                        encoded_tensor.shape, encoded_tensor.dtype)
                    encoded_buf.copy_(encoded_tensor)
                    encoded_buffers.append(encoded_buf)
                self.scale_cpu_buffers[attr_name][layer_idx] = encoded_buffers

    # ------------------------------------------------------------------ #
    #  Deferred weight-load pool                                          #
    # ------------------------------------------------------------------ #
    #
    # Weight loading is callback-driven: the safetensors loader calls
    # load_w13/load_w2/_load_scale_shard once per shard (~99k calls), serially
    # in the main thread. The per-call strided transpose-copy into pinned
    # memory is ~0.2 GB/s single-threaded, which dominated startup (~9 min).
    #
    # Strategy: each loader callback (a) owns the shard via a synchronous
    # .clone() while the safetensors mmap is still mapped, then (b) submits
    # the strided transpose-copy to a worker pool and returns immediately.
    # The main thread keeps pulling shards while the pool churns through
    # copies concurrently. drain_load_pool() barriers before _finalize_offload
    # reads the buffers. Because the deferred copy reads the owned clone (not
    # the mmap view), it stays correct after the safetensors mmap is unmapped
    # (which happens before _finalize_offload runs).

    def _get_load_pool(self) -> ThreadPoolExecutor:
        if self._load_pool is None:
            # Pin torch intra-op threads to 1: otherwise each copy_ spawns
            # nproc libgomp threads and 32 workers x 640 cores exhausts the
            # thread limit (EAGAIN). Parallelism comes from the pool itself.
            self._saved_num_threads = torch.get_num_threads()
            torch.set_num_threads(1)
            self._load_pool = ThreadPoolExecutor(
                max_workers=self._LOAD_POOL_WORKERS,
                thread_name_prefix="offload-load")
            self._load_phase_start = time.perf_counter()
            logger.info(
                "[OFFLOAD] starting parallel weight load (workers=%d)",
                self._LOAD_POOL_WORKERS)
        return self._load_pool

    def _track_load_future(self, fut) -> None:
        self._load_futures.append(fut)
        if len(self._load_futures) >= self._LOAD_POOL_DRAIN_EVERY:
            self._drain_futures()

    def _drain_futures(self) -> None:
        if not self._load_futures:
            return
        # f.result() re-raises any worker exception (e.g. shape mismatch).
        n = len(self._load_futures)
        for f in self._load_futures:
            f.result()
        self._load_futures.clear()
        previously_drained = getattr(self, "_drained_shards", 0)
        self._drained_shards = previously_drained + n
        t0 = getattr(self, "_load_phase_start", None)
        crossed_progress_boundary = (
            self._drained_shards // self._LOAD_PROGRESS_LOG_EVERY
            > previously_drained // self._LOAD_PROGRESS_LOG_EVERY
        )
        if t0 and crossed_progress_boundary:
            elapsed = time.perf_counter() - t0
            logger.info(
                "[OFFLOAD] weight load progress: %d shards copied "
                "(%.0f shards/s, %.0fs elapsed)",
                self._drained_shards,
                self._drained_shards / max(elapsed, 1e-6),
                elapsed,
            )

    def drain_load_pool(self) -> None:
        """Wait for all deferred weight copies to finish.

        Safe to call after the safetensors mmap is unmapped: deferred copies
        read owned clones, not mmap views.
        """
        self._drain_futures()
        if self._load_pool is not None:
            self._load_pool.shutdown(wait=True)
            self._load_pool = None
            if self._saved_num_threads is not None:
                torch.set_num_threads(self._saved_num_threads)
                self._saved_num_threads = None
            self._weight_load_secs = time.perf_counter() - self._load_phase_start

    # -- int4 packing helper for W4A8_DYNAMIC checkpoints -- #

    @staticmethod
    def _pack_int4_dim0(weight: torch.Tensor) -> torch.Tensor:
        """Pack pairs of int4 values along dim 0.

        W4A8_DYNAMIC (msModelSlim new_quant_version) checkpoint weights store
        one int4 value per int8 element along the output dimension.  The
        device tensor expects two int4 values packed into one int8 byte,
        halving dim 0.  This helper performs that packing.

        For w1/w3 shard ``(IN, H)`` -> ``(IN // 2, H)``.
        For w2 ``(H, IN)`` -> ``(H // 2, IN)``.
        """
        if weight.dtype != torch.int8:
            weight = weight.to(torch.int8)
        assert weight.shape[0] % 2 == 0, (
            f"dim 0 must be even for int4 packing, got {weight.shape[0]}")
        pairs = weight.reshape(weight.shape[0] // 2, 2, *weight.shape[1:])
        lo = pairs[:, 0] & 0x0F
        hi = pairs[:, 1] & 0x0F
        return ((hi << 4) | lo).contiguous()

    # -- worker copy kernels (static: no self, no shared mutable state) -- #

    @staticmethod
    def _copy_w13_shard(cpu: torch.Tensor, owned: torch.Tensor,
                        shard_id: str, intermed: int) -> None:
        if shard_id == "w1":
            cpu[:, :intermed].copy_(owned.t())
        elif shard_id == "w3":
            cpu[:, intermed: intermed + owned.shape[0]].copy_(owned.t())

    @staticmethod
    def _copy_w2(dst: torch.Tensor, owned: torch.Tensor) -> None:
        dst.copy_(owned.t())

    @staticmethod
    def _copy_scale_assembled(target: torch.Tensor,
                              w1: torch.Tensor, w3: torch.Tensor) -> None:
        assembled = torch.cat([w1, w3], dim=0)
        if target.dtype == torch.uint8:
            # W4A8_MXFP: store the post-layout bytes so the 1D buffer matches
            # the post-process device slot element order (the device path
            # applies reshape(...,k//2,2).transpose to the e8m0 scale).
            from vllm_ascend.quantization.methods.w4a8_mxfp4 import (
                apply_mxfp4_weight_scale_layout)
            assembled = apply_mxfp4_weight_scale_layout(assembled.view(torch.uint8))
        target.copy_(assembled.reshape(target.shape))

    @staticmethod
    def _copy_scale_direct(target: torch.Tensor, owned: torch.Tensor) -> None:
        if target.dtype == torch.uint8:
            from vllm_ascend.quantization.methods.w4a8_mxfp4 import (
                apply_mxfp4_weight_scale_layout)
            owned = apply_mxfp4_weight_scale_layout(owned.view(torch.uint8))
        target.copy_(owned.reshape(target.shape))

    # ------------------------------------------------------------------ #
    #  Weight-load entry points (called by the safetensors loader)        #
    # ------------------------------------------------------------------ #

    def register_gate_weights(self, _model):
        """Store an fp32 NPU copy of gate.weight for each MoE layer.

        Called from _finalize_offload() after all MoE layers are registered.
        Used by predict_next_layer_experts_npu() so prediction runs on-device
        and can be captured in a CUDA/NPU graph.
        """
        # moe_layers is the authoritative registration order used by every
        # per-layer offload array. Its entries are RoutedExperts objects, so
        # the runner propagates the owning model gate onto each entry. This is
        # model-agnostic (DeepSeek and Kimi K3 use different wrapper classes)
        # and keeps missing gates represented by None rather than shifting all
        # later layer indices.
        self._gate_weights_npu = []
        for layer in self.moe_layers:
            gate = getattr(layer, "gate", None)
            gate_param = getattr(gate, "weight", None)
            self._gate_weights_npu.append(
                None if gate_param is None else gate_param.data.float().clone())
        logger.info("[PREFETCH] registered gate weights for %d MoE layers",
                    len(self._gate_weights_npu))

    def _register_layer_gate(self, layer):
        """Stage one MoE layer's gate.weight for prefetch prediction.

        Single-layer counterpart to register_gate_weights(), for layers
        registered after _finalize_offload (e.g. the MTP draft MoE). Keeps
        _gate_weights_npu index-aligned with moe_layers so
        predict_next_layer_experts_npu can look up
        _gate_weights_npu[next_idx] for every registered layer.

        No-op before _finalize_offload has built the runtime buffers (the
        target layers are covered in bulk there). A missing gate is appended
        as None to preserve index alignment.
        """
        if not self.offload_config.expert_prefetch_enabled:
            return
        if not hasattr(self, "topk_ids_h"):
            return
        gate = getattr(layer, 'gate', None)
        gate_param = getattr(gate, 'weight', None)
        self._gate_weights_npu.append(
            None if gate_param is None else gate_param.data.float().clone())
        logger.info(
            "[PREFETCH] registered gate weight for post-finalize layer "
            "(total gates=%d, moe_layers=%d)",
            len(self._gate_weights_npu), len(self.moe_layers))

    def _shard_local(self, global_eid: int) -> int | None:
        """Map a GLOBAL expert id to the CPU weight buffer's local index.

        shard-per-rank: this rank owns shard [base, base+shard); return the
        local index, or None if the expert isn't owned here (caller skips the
        load). Other modes: identity — the buffer is full (global-indexed) or
        shared (global-indexed mmap slice).
        """
        if not self.offload_config.shard_per_rank:
            return global_eid
        local = global_eid - self._shard_base
        return local if 0 <= local < self._shard_size else None

    def load_w13(self, layer_moe_idx: int, expert_id: int,
                 loaded_weight: torch.Tensor, shard_id: str):
        """Store w1/w3 shard to CPU buffer (transposed) via the load pool."""
        self._weight_load_calls += 1
        idx = self._shard_local(expert_id)
        if idx is None:
            return  # shard-per-rank: expert not owned by this rank
        cpu = self.w13_weights_cpu[layer_moe_idx][idx]
        intermed = cpu.shape[1] // 2
        if loaded_weight.ndim > 0 and loaded_weight.shape[0] > intermed:
            if loaded_weight.shape[0] == 2 * intermed:
                loaded_weight = self._pack_int4_dim0(loaded_weight)
            else:
                loaded_weight = loaded_weight.narrow(0, 0, intermed)
        owned = loaded_weight.cpu().clone()
        fut = self._get_load_pool().submit(
            self._copy_w13_shard, cpu, owned, shard_id, intermed)
        self._track_load_future(fut)

    def load_w2(self, layer_moe_idx: int, expert_id: int,
                loaded_weight: torch.Tensor):
        """Store w2 weight to CPU buffer (transposed) via the load pool."""
        self._weight_load_calls += 1
        idx = self._shard_local(expert_id)
        if idx is None:
            return  # shard-per-rank: expert not owned by this rank
        dst = self.w2_weights_cpu[layer_moe_idx][idx]
        owned = loaded_weight.cpu().clone()
        fut = self._get_load_pool().submit(self._copy_w2, dst, owned)
        self._track_load_future(fut)

    # ------------------------------------------------------------------ #
    #  Scale / offset helpers (quantized models only)                     #
    # ------------------------------------------------------------------ #

    def _load_scale_shard(self, layer_moe_idx: int, expert_id: int,
                          attr_name: str, shard_id: str,
                          loaded_weight: torch.Tensor):
        """Load a scale/offset shard into its CPU buffer via the load pool.

        w13 scale/offset arrives as two shards (w1, w3) that must be
        concatenated along dim 0. We stash the first-arriving owned clone in
        _scale_shard_temp and assemble when the second shard arrives.
        w2 scale/offset is a single shard — clone and defer directly.
        """
        self._weight_load_calls += 1
        assert shard_id in ("w1", "w2", "w3"), f"unexpected shard_id: {shard_id}"
        if "scale_bias" in attr_name:
            target_dict = self.scale_bias_cpu_buffers
        elif "scale" in attr_name:
            target_dict = self.scale_cpu_buffers
        else:
            target_dict = self.offset_cpu_buffers
        # shard-per-rank: skip scales for non-owned experts and index the
        # shard-sized buffer locally (mirrors load_w13/load_w2).
        local_eid = self._shard_local(expert_id)
        if local_eid is None:
            return  # shard-per-rank: scale not owned by this rank
        target = target_dict[attr_name][layer_moe_idx][local_eid]
        if attr_name.startswith("w13_"):
            key = (layer_moe_idx, expert_id, attr_name)
            pending_shard = self._scale_shard_temp.pop(key, None)
            if pending_shard is not None:
                # Second shard — own it, then defer cat + copy.
                cur_shard = loaded_weight.cpu().clone()
                if shard_id == "w1":
                    w1, w3 = cur_shard, pending_shard
                else:
                    w1, w3 = pending_shard, cur_shard
                fut = self._get_load_pool().submit(
                    self._copy_scale_assembled, target, w1, w3)
                self._track_load_future(fut)
            else:
                # First shard — stash an owned clone.
                self._scale_shard_temp[key] = loaded_weight.cpu().clone()
        else:
            # w2 scale/offset — single shard.
            owned = loaded_weight.cpu().clone()
            fut = self._get_load_pool().submit(
                self._copy_scale_direct, target, owned)
            self._track_load_future(fut)

    def refresh_fp32_scales(self):
        """Recompute the derived fp32 per-expert scale after weight loading.

        Device experts are already in place (loaded by the weight loader and
        process_weights_after_loading); this only refreshes
        w13_weight_scale_fp32 from the freshly-loaded w13_weight_scale.
        """
        for i, layer in enumerate(self.moe_layers):
            ndev = min(self.num_device_experts_for_layer(i),
                       _expert_weight(layer, "w13_weight").shape[0])
            if hasattr(layer, 'w13_weight_scale_fp32'):
                for j in range(ndev):
                    layer.w13_weight_scale_fp32[j].copy_(
                        layer.w13_weight_scale.data[j].to(torch.float32))

    def create_prefill_pool(self):
        """Allocate prefill pool tensors on NPU with full expert count.

        Called from _finalize_offload() after decode buffers are set up.
        Creates ndl device tensors each holding all experts (e.g. 128).
        These are used when num_tokens > offload_threshold (large-batch
        prefill), loaded via full-overwrite in _prefill_load_layer.
        """
        if self._prefill_initialized:
            return
        if not self.moe_layers:
            return
        ndl = self.num_device_layers
        pool_layer = self.moe_layers[0]
        _pool_w13 = _expert_weight(pool_layer, "w13_weight")
        dev = _pool_w13.device
        dt = _pool_w13.dtype
        # Size the pool to the per-rank EP shard, NOT the global expert count.
        # The All2All prefill GMM (aclnnGroupedMatmulWeightNz) requires
        # groupList == weight.dim0; groupList = shard, so the pool must hold
        # exactly `shard` experts per rank. mc_shard_size == num_total//ep_size,
        # which is ntotal for single-card (ep_size=1) — so single-card is
        # unchanged (pool holds all experts), multi-card holds the rank's shard.
        ntotal = self.mc_shard_size

        for _ in range(ndl):
            self._alloc_prefill_pool_slot(pool_layer, dev, dt, ntotal)

        # Cast prefill pool weight tensors to the on-device format (kernel
        # requires it). Must happen BEFORE loading data — same order as decode
        # path: create → format-cast → copy_(cpu → npu).
        self._cast_prefill_pool_format(dev, dt)

        # Prefill log2phy: identity — all experts mapped to their slots
        self._prefill_log2phy = torch.arange(ntotal, dtype=torch.int32, device=dev)

        # Pre-initialize all pool slots with layer 0 weights so that
        # profile_run / _dummy_run (which may use prefill path) has
        # valid data.  Subsequent _prefill_load_layer calls will
        # overwrite with the correct per-layer weights.
        self._init_prefill_pool_data(dev, ntotal, ndl)
        self._prefill_initialized = True
        logger.info("[PREFILL_POOL] allocated %d layers × %d experts, "
                    "w13[0].shape=%s w2[0].shape=%s",
                    ndl, ntotal,
                    tuple(self._prefill_w13[0].shape),
                    tuple(self._prefill_w2[0].shape))

    def _alloc_prefill_pool_slot(self, pool_layer, dev, dt, ntotal: int):
        """Append one prefill-pool slot (weights always; scales/offsets/scale_bias
        only if the layer carries them). Weights use the layer dtype `dt`;
        per-channel fp32 scales use float32; the rest use their source dtype."""
        # (target_attr, source_attr, dtype_override)
        quant_specs = [
            ("_prefill_w13_scale", "w13_weight_scale", None),
            ("_prefill_w13_scale_fp32", "w13_weight_scale_fp32", torch.float32),
            ("_prefill_w13_offset", "w13_weight_offset", None),
            ("_prefill_w2_scale", "w2_weight_scale", None),
            ("_prefill_w2_offset", "w2_weight_offset", None),
            ("_prefill_w13_scale_bias", "w13_scale_bias", None),
            ("_prefill_w2_scale_bias", "w2_scale_bias", None),
        ]
        self._prefill_w13.append(torch.empty(
            (ntotal,) + tuple(_expert_weight(pool_layer, "w13_weight").shape[1:]), dtype=dt, device=dev))
        self._prefill_w2.append(torch.empty(
            (ntotal,) + tuple(_expert_weight(pool_layer, "w2_weight").shape[1:]), dtype=dt, device=dev))
        for tgt, src, dtype_override in quant_specs:
            if not hasattr(pool_layer, src):
                continue
            src_t = getattr(pool_layer, src)
            dtype = dtype_override if dtype_override is not None else src_t.dtype
            getattr(self, tgt).append(torch.empty(
                (ntotal,) + tuple(src_t.shape[1:]), dtype=dtype, device=dev))

    def _cast_prefill_pool_format(self, dev, dt):
        """Cast prefill-pool weight tensors to the on-device (kernel) format.

        Must run BEFORE data is loaded (same create → format-cast ordering as
        the decode path). dtype-dispatched:
          - int8 (W8A8): straight FRACTAL_NZ cast.
          - int32 (W4A8_DYNAMIC): rebuild via int8 backing → NZ → view int32
            (an empty int32 tensor can't be NZ-cast directly).
          - uint8 (W4A8_MXFP): cast29 on the pre-transpose shape, then transpose.
        """
        n = len(self._prefill_w13)
        if dt == torch.int8:
            from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ
            for i in range(n):
                self._prefill_w13[i] = torch_npu.npu_format_cast(
                    self._prefill_w13[i], ACL_FORMAT_FRACTAL_NZ)
                self._prefill_w2[i] = torch_npu.npu_format_cast(
                    self._prefill_w2[i], ACL_FORMAT_FRACTAL_NZ)
        elif dt == torch.int32:
            from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ
            for i in range(n):
                t13 = self._prefill_w13[i]
                t2 = self._prefill_w2[i]
                t13_nz = torch_npu.npu_format_cast(
                    torch.empty(t13.shape[:-1] + (t13.shape[-1] * 4,),
                                dtype=torch.int8, device=dev),
                    ACL_FORMAT_FRACTAL_NZ)
                t2_nz = torch_npu.npu_format_cast(
                    torch.empty(t2.shape[:-1] + (t2.shape[-1] * 4,),
                                dtype=torch.int8, device=dev),
                    ACL_FORMAT_FRACTAL_NZ)
                self._prefill_w13[i] = t13_nz.view(torch.int32)
                self._prefill_w2[i] = t2_nz.view(torch.int32)
        elif dt == torch.uint8:
            for i in range(n):
                for attr in ("_prefill_w13", "_prefill_w2"):
                    t = getattr(self, attr)[i]
                    t = torch_npu.npu_format_cast(
                        t.transpose(1, 2).contiguous().view(torch.uint8), 29,
                        customize_dtype=torch.float8_e4m3fn,
                        input_dtype=torch_npu.float4_e2m1fn_x2,
                    )
                    getattr(self, attr)[i] = t.transpose(1, 2)

    def _build_prefill_h2d_tasks(
        self,
        layer_idx: int,
        pool_slot: int,
        source_eid: int,
        destination_eid: int,
    ) -> list[H2DCopyTask]:
        """Build weight and quant tasks for one prefill-pool expert."""
        w13_start = destination_eid * self.w13_expert_size_bytes
        w2_start = destination_eid * self.w2_expert_size_bytes
        w13_dst = self._prefill_w13[pool_slot].untyped_storage()[
            w13_start:w13_start + self.w13_expert_size_bytes]
        w2_dst = self._prefill_w2[pool_slot].untyped_storage()[
            w2_start:w2_start + self.w2_expert_size_bytes]
        tasks = [
            H2DCopyTask(
                source=self._expert_src_storage(
                    layer_idx, source_eid, 'w13'),
                destination=w13_dst,
                nbytes=self.w13_expert_size_bytes,
                name=(f"prefill-w13[L{layer_idx},E{source_eid}"
                      f"->P{pool_slot}:E{destination_eid}]"),
            ),
            H2DCopyTask(
                source=self._expert_src_storage(
                    layer_idx, source_eid, 'w2'),
                destination=w2_dst,
                nbytes=self.w2_expert_size_bytes,
                name=(f"prefill-w2[L{layer_idx},E{source_eid}"
                      f"->P{pool_slot}:E{destination_eid}]"),
            ),
        ]

        local_eid = self._shard_local(source_eid)
        quant_specs = (
            (self.scale_cpu_buffers, "w13_weight_scale",
             self._prefill_w13_scale),
            (self.scale_cpu_buffers, "w2_weight_scale",
             self._prefill_w2_scale),
            (self.offset_cpu_buffers, "w13_weight_offset",
             self._prefill_w13_offset),
            (self.offset_cpu_buffers, "w2_weight_offset",
             self._prefill_w2_offset),
            (self.scale_bias_cpu_buffers, "w13_scale_bias",
             self._prefill_w13_scale_bias),
            (self.scale_bias_cpu_buffers, "w2_scale_bias",
             self._prefill_w2_scale_bias),
        )
        for cpu_buffers, attr_name, prefill_buffers in quant_specs:
            if (pool_slot >= len(prefill_buffers)
                    or attr_name not in cpu_buffers
                    or layer_idx >= len(cpu_buffers[attr_name])):
                continue
            dst = prefill_buffers[pool_slot][destination_eid]
            if (local_eid is not None
                    and local_eid < len(cpu_buffers[attr_name][layer_idx])):
                src_tensor = cpu_buffers[attr_name][layer_idx][local_eid]
                source = src_tensor.reshape(dst.shape)
                nbytes = src_tensor.numel() * src_tensor.element_size()
            else:
                source = self._shared_h2d_source(
                    layer_idx, source_eid, attr_name)
                nbytes = dst.numel() * dst.element_size()
            tasks.append(H2DCopyTask(
                source=source,
                destination=dst,
                nbytes=nbytes,
                name=(f"prefill-{attr_name}[L{layer_idx},E{source_eid}"
                      f"->P{pool_slot}:E{destination_eid}]"),
            ))
        return tasks

    def _refresh_prefill_fp32_scale(self, pool_slot: int,
                                    num_experts: int) -> None:
        if (pool_slot >= len(self._prefill_w13_scale_fp32)
                or pool_slot >= len(self._prefill_w13_scale)):
            return
        count = min(num_experts,
                    self._prefill_w13_scale[pool_slot].shape[0])
        for eid in range(count):
            self._prefill_w13_scale_fp32[pool_slot][eid].copy_(
                self._prefill_w13_scale[pool_slot][eid].to(torch.float32),
                non_blocking=True)

    def _init_prefill_pool_data(self, dev, ntotal: int, ndl: int):
        """Load layer 0 weights into all prefill pool slots.

        Prefill pool tensors are already NZ-cast at this point (done in
        create_prefill_pool). Route the initial full overwrite through the
        configured H2D transport, just like runtime prefill/decode loading.
        """
        del dev
        num_source_experts = min(ntotal, len(self.w13_weights_cpu[0]))
        with torch_npu.npu.stream(self.load_stream):
            for slot in range(ndl):
                tasks = []
                for local_eid in range(num_source_experts):
                    source_eid = (
                        self._shard_base + local_eid
                        if self.offload_config.shard_per_rank else local_eid)
                    tasks.extend(self._build_prefill_h2d_tasks(
                        0, slot, source_eid, local_eid))
                self._get_h2d_transport().copy_batch(tasks)
                self._refresh_prefill_fp32_scale(slot, num_source_experts)
            self._synchronize_h2d()

    def _prefill_load_layer(self, layer_idx: int, log2phy: torch.Tensor):
        """Load ALL experts for model layer layer_idx into the prefill pool.

        For W8A8: loads into normal-format scratch, then casts to NZ.
        For unquantized: loads directly into pool tensors via copy_().
        Full-overwrite into pool_slot = layer_idx % ndl.  No slot_owner
        tracking needed — log2phy is set to identity for prefill.
        """
        ndl = self.num_device_layers
        pool_slot = layer_idx % ndl
        ntotal = self.num_total_experts
        is_w8a8 = self._prefill_w13[pool_slot].dtype == torch.int8

        if self._debug:
            logger.info("[PREFILL_LOAD] layer=%d pool_slot=%d ntotal=%d is_w8a8=%s",
                        layer_idx, pool_slot, ntotal, is_w8a8)

        with torch_npu.npu.stream(self.load_stream):
            tasks = []
            for eid in range(ntotal):
                tasks.extend(self._build_prefill_h2d_tasks(
                    layer_idx, pool_slot, eid, eid))
            self._get_h2d_transport().copy_batch(tasks)
            self._refresh_prefill_fp32_scale(pool_slot, ntotal)
            self._synchronize_h2d()

        # NOTE: Do NOT modify the layer's own log2phy here — decode path
        # relies on it staying with 32-expert mapping.  Prefill path in
        # apply() explicitly uses self._prefill_log2phy instead.

    # ------------------------------------------------------------------ #
    #  Multi-card prefill: per-rank EP shard into the prefill pool        #
    # ------------------------------------------------------------------ #
    @property
    def mc_shard_size(self) -> int:
        """Experts per rank in standard EP shard (num_total_experts // ep_size)."""
        return self.num_total_experts // max(1, self.ep_size)

    def _get_shard_expert_map(self) -> torch.Tensor:
        """Standard EP shard expert_map for THIS rank, len = num_total_experts.
        Maps global experts in this rank's shard [base, base+shard) to local
        index [0..shard), everything else to -1. Consumed by the AllGather
        dispatcher (it masks topk_ids to local via expert_map != -1 and uses
        active_expert_range = [rank*nel, rank*nel+nel]).
        """
        if getattr(self, '_mc_shard_expert_map', None) is not None:
            return self._mc_shard_expert_map
        shard = self.mc_shard_size
        base = self.ep_rank * shard
        emap = torch.full((self.num_total_experts,), -1, dtype=torch.int32)
        for i in range(shard):
            emap[base + i] = i
        self._mc_shard_expert_map = emap
        return emap

    def _prefill_load_layer_shard(self, layer_idx: int):
        """Multi-card prefill: load THIS rank's EP shard into the prefill pool.

        Standard EP AllGather has rank r own global experts [r*shard:(r+1)*shard]
        and compute them in LOCAL slots [0:shard]. So we load only the rank's
        shard (not all experts) into pool local slots [0:shard]. The pool buffer
        is sized for num_total_experts; slots [shard:] stay unused this forward.
        Mirrors _prefill_load_layer but sharded + per-rank.
        """
        if not self._prefill_initialized or not self.moe_layers:
            return
        ndl = self.num_device_layers
        pool_slot = layer_idx % ndl
        shard = self.mc_shard_size
        base = self.ep_rank * shard

        if self.offload_config.h2d_backend == "memfabric":
            with torch_npu.npu.stream(self.load_stream):
                tasks = []
                for local_i, eid in enumerate(range(base, base + shard)):
                    tasks.extend(self._build_prefill_h2d_tasks(
                        layer_idx, pool_slot, eid, local_i))
                self._get_h2d_transport().copy_batch(tasks)
                self._refresh_prefill_fp32_scale(pool_slot, shard)
                self._synchronize_h2d()
            return

        with torch_npu.npu.stream(self.load_stream):
            for local_i in range(shard):
                eid = base + local_i
                self._prefill_w13[pool_slot].untyped_storage()[local_i * self.w13_expert_size_bytes : (local_i + 1) * self.w13_expert_size_bytes].copy_(
                    self._expert_src_storage(layer_idx, eid, 'w13'))
                self._prefill_w2[pool_slot].untyped_storage()[local_i * self.w2_expert_size_bytes : (local_i + 1) * self.w2_expert_size_bytes].copy_(
                    self._expert_src_storage(layer_idx, eid, 'w2'))
            # quant scales / offsets / scale_bias (w4a8) — shard only
            for scale_name, prefill_list in [("w13_weight_scale", self._prefill_w13_scale),
                                             ("w2_weight_scale", self._prefill_w2_scale)]:
                if pool_slot < len(prefill_list) and scale_name in self.scale_cpu_buffers \
                        and layer_idx < len(self.scale_cpu_buffers[scale_name]):
                    for local_i in range(min(shard, len(self.scale_cpu_buffers[scale_name][layer_idx]))):
                        src = self.scale_cpu_buffers[scale_name][layer_idx][local_i]
                        prefill_list[pool_slot][local_i].copy_(src.reshape(prefill_list[pool_slot][local_i].shape))
            for off_name, prefill_list in [("w13_weight_offset", self._prefill_w13_offset),
                                           ("w2_weight_offset", self._prefill_w2_offset)]:
                if pool_slot < len(prefill_list) and off_name in self.offset_cpu_buffers \
                        and layer_idx < len(self.offset_cpu_buffers[off_name]):
                    for local_i in range(min(shard, len(self.offset_cpu_buffers[off_name][layer_idx]))):
                        src = self.offset_cpu_buffers[off_name][layer_idx][local_i]
                        prefill_list[pool_slot][local_i].copy_(src.reshape(prefill_list[pool_slot][local_i].shape))
            for sb_name, prefill_list in [("w13_scale_bias", self._prefill_w13_scale_bias),
                                          ("w2_scale_bias", self._prefill_w2_scale_bias)]:
                if pool_slot < len(prefill_list) and sb_name in self.scale_bias_cpu_buffers \
                        and layer_idx < len(self.scale_bias_cpu_buffers[sb_name]):
                    for local_i in range(min(shard, len(self.scale_bias_cpu_buffers[sb_name][layer_idx]))):
                        src = self.scale_bias_cpu_buffers[sb_name][layer_idx][local_i]
                        prefill_list[pool_slot][local_i].copy_(src.reshape(prefill_list[pool_slot][local_i].shape))
            # Sync the load stream so the pool data is valid before the GMM reads
            # it (the apply path continues on the default stream after we return;
            # the host-side block here guarantees the copies are done before the
            # next op is queued).
            self._synchronize_h2d()

    # ------------------------------------------------------------------ #
    #  Forward path: page in experts based on topk_ids                   #
    # ------------------------------------------------------------------ #
    
    def _read_pf_wait(self, layer_idx: int):
        """Milliseconds the compute stream was blocked at this layer's join.

        Called from the reactive host callback, which is stream-ordered after
        the `end` record, so under graph replay both timestamps are already
        committed. The explicit `end.synchronize()` covers eager, where the
        callback runs inline on the forward thread and the compute stream may
        not have reached the record yet.

        `_pf_wait_bias` is the calibrated cost of the bracket itself (see
        _finalize_offload) and is subtracted so the number approximates the
        stall alone rather than stall + instrument.

        Returns None rather than raising: a device that cannot time events
        inside a captured graph must degrade to "no samples", not kill the run.
        An all-None series makes the summary row VANISH rather than print zeros
        (decode_stats._summarize_series returns None for an empty series), so
        the log lines below are the only way to tell "no stall measured" apart
        from "feature never applied".
        """
        if not self._pf_wait_timing:
            return None
        if layer_idx >= len(self._pf_wait_armed):
            return None
        if not self._pf_wait_probed:
            # One line, on the first read attempt of the run, that separates
            # every failure mode at once: armed=0 means no layer had a wait node
            # to bracket, armed=N with no later "first sample" line means
            # elapsed_time is failing inside the graph.
            self._pf_wait_probed = True
            logger.info(
                "[PREFETCH-WAIT] first read: layer=%d armed=%d/%d bias=%.4f ms",
                layer_idx, sum(self._pf_wait_armed),
                len(self._pf_wait_armed), self._pf_wait_bias)
        if not self._pf_wait_armed[layer_idx]:
            return None
        try:
            end = self._pf_wait_end[layer_idx]
            end.synchronize()
            raw = float(self._pf_wait_begin[layer_idx].elapsed_time(end))
        except Exception:
            self._pf_wait_fail += 1
            if self._pf_wait_fail == 1:
                # count and carry on. Latching _pf_wait_timing off here
                # also stopped the RECORDING, and because the eager warmup runs
                # before the capture, that silently emptied the captured graph.
                logger.warning(
                    "[PREFETCH-WAIT] elapsed_time failed; pf_wait samples will "
                    "be dropped (recording left on so a warmup-only failure "
                    "does not empty the captured graph)", exc_info=True)
            return None
        # Clamp: a stall shorter than the bracket's own cost is indistinguishable
        # from no stall, and a negative millisecond in the summary is worse than
        # a zero.
        value = raw - self._pf_wait_bias
        if value < 0.0:
            value = 0.0
        self._pf_wait_ok += 1
        if self._pf_wait_ok == 1:
            # Mirrors '[DECODE-STATS] first sample recorded' — a positive
            # signal, so an empty summary row is never ambiguous.
            logger.info(
                "[PREFETCH-WAIT] first sample: layer=%d stall=%.3f ms "
                "(raw=%.3f bias=%.4f)", layer_idx, value, raw,
                self._pf_wait_bias)
        return value
        
    def _finish_pending_predict(self, layer_idx: int) -> None:
        """Consume a trained predictor's in-flight prediction for this layer.

        Called from update_weights and update_weights_multi_card as their first
        act, before the prefetch event pop — the latest point at which the
        prefetch can still be dispatched, and the first at which the real
        routed experts are known.

        It lives here rather than in the decoder layer because the decoder
        layer's forward is traced by Dynamo (DeepseekV4Model is
        @support_torch_compile with fullgraph) and nothing the driver's
        finish() does is traceable. This method is on the apply() side, already
        behind the MoE custom-op boundary — the same boundary that has always
        made this class's own streams, locks and host callbacks safe.

        All four apply() implementations call update_weights /
        update_weights_multi_card unconditionally inside their
        `enable_expert_offload` block, before any prefill-regime branch, and a
        layer can only have launched if it is one of those layers — so the
        launch/finish pairing is structural.

        No-op with the heuristic method (no driver), and for hash layers and
        any layer that did not launch (no pending entry).
        """
        predictor = self.expert_predictor
        if predictor is not None:
            predictor.finish(layer_idx)
            
    def _finish_next_layer_predict(self, layer_idx: int,
                                   compute_stream) -> None:
        """Dispatch a layer-shifted predictor's prediction for layer_idx + 1.

        Called at the END of update_weights[_multi_card], after this layer's
        reactive host callback has been enqueued. The event recorded here is the
        ordering edge that keeps the prefetch H2D for layer_idx + 1 off the
        report thread and off load_stream until layer_idx's on-demand load has
        drained — the reactive callback internally synchronizes load_stream, so
        the compute stream cannot reach this record until that transfer is done.

        No-op for fate and for mode2_har (nothing is ever pending under the
        next layer's key), and the event is only allocated when there is
        something to dispatch, so neither method pays an extra Event per layer
        per capture size.
        """
        predictor = self.expert_predictor
        if predictor is None or not predictor.has_pending(layer_idx + 1):
            return
        ondemand_done = torch_npu.npu.Event()
        compute_stream.record_event(ondemand_done)
        predictor.finish(layer_idx + 1, ondemand_done)

    def update_weights(self, layer, topk_ids: torch.Tensor,
                        log2phy: torch.Tensor,
                        topk_weights: torch.Tensor | None = None,
                        hidden_states: torch.Tensor | None = None,
                        router_logits: torch.Tensor | None = None,
                        renormalize: bool = False,
                        scoring_func: str = "softmax",
                        e_score_correction_bias: torch.Tensor | None = None,
                        routed_scaling_factor: float = 1.0,
                        is_hash_routed: bool = False) -> int:
        """Incrementally page in needed experts, overwriting unused slots.

        Routes to prefill pool (full-overwrite) when num_tokens exceeds
        offload_threshold, otherwise uses per-expert paging (decode path).

        Args:
            layer: AscendFusedMoE instance.
            topk_ids: [num_tokens, top_k] routed expert indices.
            log2phy: [global_num_experts] CPU tensor, modified in-place.
            topk_weights: Optional routing weights for cache policy.
            hidden_states: Optional [num_tokens, hidden_dim] tensor used
                           for next-layer expert prefetch prediction.

        Returns: number of CPU→NPU copies performed (decode path),
                 0 for prefill path (full-overwrite via pool).
        """
        try:
            layer_idx = self.moe_layers.index(layer)
        except ValueError:
            return 0
        self._finish_pending_predict(layer_idx)
        # Wait for prefetch NPU copies to complete before using the weights.
        # Use stream wait (graphable) instead of host synchronize.
        with self._prefetch_state_lock:
            npu_event = self._prefetch_layer_npu_event.pop(layer_idx, None)
        if npu_event is not None:
            wait_stream = torch_npu.npu.current_stream()
            # Bracket the join with device events so the stall is measurable
            # under graph replay, where no Python runs. Both records are graph
            # nodes re-executed every replay; _read_pf_wait consumes them from
            # the reactive callback. Nothing is recorded when the key is off.
            if self._pf_wait_timing:
                self._pf_wait_begin[layer_idx].record(wait_stream)
            wait_stream.wait_event(npu_event)
            if self._pf_wait_timing:
                self._pf_wait_end[layer_idx].record(wait_stream)
                self._pf_wait_armed[layer_idx] = True
        elif (self._pf_wait_timing
              and layer_idx < len(self._pf_wait_armed)
              # only a DECODE-regime call may clear the flag
              and topk_ids.size(0) <= self.offload_threshold):
            # No prefetch for this layer this step: no wait node exists, so
            # there is nothing to time and the layer contributes no sample.
            self._pf_wait_armed[layer_idx] = False

        # Multi-card offload sets routed layers' log2phy=None (standard-EP
        # dispatch) and uses update_weights_multi_card instead. Shared experts
        # (per-card replicated, not dispatched) may still reach here with
        # log2phy=None — bail out to avoid copy_(None). TODO: give shared
        # experts a proper single-card log2phy so they still page in.
        if log2phy is None:
            return 0
        num_tokens = topk_ids.size(0)
        if num_tokens > self.offload_threshold:
            # Prefill: layerwise reuse + full-overwrite of all experts
            if (self._prefill_initialized
                    and not self._skip_prefill):
                # reuse the layer_idx resolved above
                self._prefill_load_layer(layer_idx, log2phy)
                return 0
            else:
                # Profile run or pool not ready — bail out gracefully
                return 0

        prune_debug = None
        if (self.offload_config.experts_pruning_enabled
                and topk_weights is not None):
            logger.info_once(
                "[EXPERT-PRUNE] entered NPU pruning path: "
                "topk=%d thresholds=%s",
                self.topk,
                tuple(self.offload_config.experts_pruning_threshold),
            )
            pruned_weights, pruned_ids, prune_debug = maybe_prune_topk_experts(
                topk_weights,
                topk_ids,
                log2phy=log2phy,
            )
            # 必须原地写回，因为w4a8.py仍然持有原Tensor。
            topk_weights.copy_(pruned_weights)
            topk_ids.copy_(pruned_ids)
        prune_debug_h = None
        if prune_debug is not None:
            prune_debug_npu = prune_debug.permute(
                1, 0, 2
            ).contiguous()
            prune_debug_h = self.prune_debug_h[:num_tokens]
            prune_debug_h.copy_(
                prune_debug_npu,
                non_blocking=_EXTRA_CTX.capturing,
            )
        topk_ids_h = self.topk_ids_h[:num_tokens]
        do_substitution = (
            self.offload_config.expert_substitution_enabled
            and not is_hash_routed
            and router_logits is not None
            and router_logits.shape[-1] == self.num_total_experts
        )
        if (self.offload_config.expert_substitution_enabled
                and not is_hash_routed and router_logits is not None
                and router_logits.shape[-1] != self.num_total_experts):
            logger.warning_once(
                "[SUBST] router_logits width %d != num_total_experts %d; "
                "expert substitution is disabled for this layer",
                router_logits.shape[-1], self.num_total_experts)

        topk_weights_h = None
        if (topk_weights is not None and self.cache_policy is not None
                and self.offload_config.cache_router_weight != 0):
            topk_weights_h = self.topk_weights_h[:num_tokens]
            topk_weights_h.copy_(topk_weights.to(dtype=torch.float32), non_blocking=_EXTRA_CTX.capturing)
        # substitution now runs on the NPU
        # Replaces the router_logits_h / e_score_correction_bias_h staging (an
        # [n, 256] fp32 D2H per layer per step) and the whole
        # plan_expert_substitutions Python block
        topk_ids_gt_h = None
        if do_substitution:
            if self.topk_ids_gt_h is not None:
                # Ground truth G for hit_pre / pred_acc / subst (§3.6), staged
                # BEFORE the mutation below. Deliberately NOT gated on
                # stats.collecting the way the old host clone was: this line is
                # captured, so a branch on a flag that flips at arming time
                # would be frozen to whatever it was at capture. 72 bytes.
                topk_ids_gt_h = self.topk_ids_gt_h[:num_tokens]
                topk_ids_gt_h.copy_(topk_ids[:, :self.topk],
                                    non_blocking=_EXTRA_CTX.capturing)
            substituted_ids = substitute_experts_device(
                router_logits,
                topk_ids[:, :self.topk],
                log2phy,
                expert_substitution_threshold=(
                    self.offload_config.expert_substitution_threshold),
                scoring_func=scoring_func,
                e_score_correction_bias=e_score_correction_bias,
            )
            # In-place so every downstream consumer (log2phy gather, dispatch,
            # GMM) sees the substituted ids. The previous code mutated this
            # same tensor the same way, just after the callback instead of
            # before. Same stream, so the gt D2H above reads pre-mutation.
            topk_ids[:, :self.topk].copy_(substituted_ids)
        log2phy_h = self.log2phy_h
        log2phy_np = self.log2phy_np
        topk_ids_h.copy_(topk_ids, non_blocking=_EXTRA_CTX.capturing)
        log2phy_h.copy_(log2phy, non_blocking=_EXTRA_CTX.capturing)

        current_compute_stream = torch_npu.npu.current_stream()
        subscribed_compute_streams = get_subscribed_compute_streams()
        if current_compute_stream not in subscribed_compute_streams:
            torch_npu.npu._subscribe_report(current_compute_stream)
            subscribed_compute_streams.add(current_compute_stream)
        self._is_prefetch = False
        # router_logits_h / scoring_func / correction_bias_h
        # are gone (the callback no longer substitutes); topk_ids_gt_h carries
        # the pre-substitution ids the statistics need. The 6-tuple prefetch
        # form built by _build_prefetch_call is unchanged.
        args = (
            topk_ids_h,
            log2phy_np,
            layer,
            layer_idx,
            topk_weights_h,
            self._is_prefetch,
            do_substitution,
            topk_ids_gt_h,
            prune_debug_h,
        )
        # launch the guarded wrapper — see _note_cb_failure.
        if _EXTRA_CTX.capturing:
            torch_npu.npu._launch_host_func(
                current_compute_stream,
                self._update_weights_guarded,
                args,
            )
        else:
            self._update_weights_guarded(args)

        # The substituted ids were written into topk_ids on the device above,
        # so this H2D write-back has nothing left to publish.
        log2phy.copy_(log2phy_h, non_blocking=_EXTRA_CTX.capturing)

        # dispatch a layer-shifted predictor's prediction for the NEXT
        # layer here, behind an event recorded after this layer's on-demand
        # load. No-op for fate and mode2_har.
        self._finish_next_layer_predict(layer_idx, current_compute_stream)

    def _mc_handle_prefill_regime(self, layer_idx) -> bool:
        """Multi-card PREFILL (non-MC2 comm): load this rank's EP shard into the
        prefill pool. Returns True when handled so the caller returns early.

        We load even during profile_run: the decode buffer is too small for the
        EP shard, so multi-card prefill MUST use the pool, and real weights
        avoid GMM errors on garbage scales (single-card can skip during profile
        because its AllGather reuses decode weights; multi-card All2All cannot).
        """
        from vllm_ascend.ascend_forward_context import MoECommType
        if _EXTRA_CTX.moe_comm_type == MoECommType.MC2:
            return False
        if self._prefill_initialized:
            self._prefill_load_layer_shard(layer_idx)
            if self._debug and logger.isEnabledFor(logging.DEBUG):
                base = self.ep_rank * self.mc_shard_size
                logger.debug(
                    "[MC_OBS] rank=%s L=%s PREFILL: loaded EP shard "
                    "experts[%d..%d] (%d experts) into pool on rank%d "
                    "(static shard, reloaded each prefill forward)",
                    self.ep_rank, layer_idx, base, base + self.mc_shard_size - 1,
                    self.mc_shard_size, self.ep_rank)
        return True

    def _log_mc_debug_event(self, event: str, context=None, **details) -> None:
        """Emit one parseable CPU-only multi-card diagnostic record."""
        if not self._debug:
            return
        context = context or {}
        details_text = " ".join(
            f"{key}={value}" for key, value in details.items())
        logger.info(
            "[MC_DEBUG] event=%s rank=%s layer=%s layer_call=%s "
            "callback_seq=%s source=%s prefetch=%s pid=%s thread=%s "
            "ts_ns=%s %s",
            event,
            self.ep_rank,
            context.get("layer_idx", "-"),
            context.get("layer_call", "-"),
            context.get("callback_seq", "-"),
            context.get("source", "-"),
            context.get("is_prefetch", False),
            os.getpid(),
            threading.get_ident(),
            time.time_ns(),
            details_text,
        )

    def _log_mc_debug_schedule(self, layer_idx: int,
                               is_prefetch: bool) -> None:
        if not self._debug:
            return
        with self._mc_debug_lock:
            self._mc_debug_schedule_seq += 1
            schedule_seq = self._mc_debug_schedule_seq
        self._log_mc_debug_event(
            "CB_SCHEDULE",
            {
                "layer_idx": layer_idx,
                "source": "graph_callback",
                "is_prefetch": is_prefetch,
            },
            schedule_seq=schedule_seq,
        )

    def _begin_mc_debug_callback(self, layer_idx: int, is_prefetch: bool,
                                 from_graph_callback: bool):
        if not self._debug:
            return None
        with self._mc_debug_lock:
            self._mc_debug_callback_seq += 1
            callback_seq = self._mc_debug_callback_seq
            layer_key = (layer_idx, is_prefetch)
            layer_call = self._mc_debug_layer_calls.get(layer_key, 0) + 1
            self._mc_debug_layer_calls[layer_key] = layer_call
            self._mc_debug_active_callbacks += 1
            active_callbacks = self._mc_debug_active_callbacks
        context = {
            "layer_idx": layer_idx,
            "layer_call": layer_call,
            "callback_seq": callback_seq,
            "source": ("graph_callback" if from_graph_callback
                       else "eager_inline"),
            "is_prefetch": is_prefetch,
            "start_ns": time.perf_counter_ns(),
        }
        self._log_mc_debug_event(
            "CB_ENTER", context, active_callbacks=active_callbacks)
        return context

    def _end_mc_debug_callback(self, context, status: str) -> None:
        if context is None:
            return
        with self._mc_debug_lock:
            self._mc_debug_active_callbacks = max(
                0, self._mc_debug_active_callbacks - 1)
            active_callbacks = self._mc_debug_active_callbacks
        elapsed_us = (time.perf_counter_ns() - context["start_ns"]) // 1000
        self._log_mc_debug_event(
            "CB_EXIT",
            context,
            status=status,
            active_callbacks=active_callbacks,
            elapsed_us=elapsed_us,
        )

    def _gather_cpu_with_mc_debug(self, local_values, cpu_group, kind: str,
                                  context):
        """Wrap one Gloo all-reduce with enter/exit sequence diagnostics."""
        from vllm_ascend.expert_offload.multi_card_planner import (
            gather_global_counts_cpu)

        if context is None:
            return gather_global_counts_cpu(local_values, cpu_group)
        with self._mc_debug_lock:
            self._mc_debug_collective_seq += 1
            collective_seq = self._mc_debug_collective_seq
        start_ns = time.perf_counter_ns()
        group_name = getattr(cpu_group, "group_name", "none")
        group_id = hex(id(cpu_group)) if cpu_group is not None else "none"
        self._log_mc_debug_event(
            "GLOO_ENTER",
            context,
            kind=kind,
            collective_seq=collective_seq,
            group_name=group_name,
            local_group_id=group_id,
            dtype=local_values.dtype,
            numel=local_values.numel(),
        )
        try:
            global_values = gather_global_counts_cpu(local_values, cpu_group)
        except BaseException as exc:
            self._log_mc_debug_event(
                "GLOO_ERROR",
                context,
                kind=kind,
                collective_seq=collective_seq,
                error_type=type(exc).__name__,
            )
            raise
        elapsed_us = (time.perf_counter_ns() - start_ns) // 1000
        self._log_mc_debug_event(
            "GLOO_EXIT",
            context,
            kind=kind,
            collective_seq=collective_seq,
            elapsed_us=elapsed_us,
        )
        return global_values

    def update_weights_multi_card(self, layer, topk_ids, log2phy,
                                  topk_weights=None, hidden_states=None,
                                  mc2_mask=None,
                                  router_logits=None,
                                  renormalize=False,
                                  scoring_func="softmax",
                                  e_score_correction_bias=None,
                                  routed_scaling_factor=1.0,
                                  is_hash_routed=False):
        """Multi-card EP offload: planner decides global placement; this rank
        H2D-loads only its assigned experts and writes the placement into
        ``log2phy`` (which the MC2 dispatcher then consumes).

        MVP: full H2D of this rank's assigned experts every layer. No
        skip-if-resident, no LRC victim selection, no hot pool yet (those are
        later stages). Determinism comes from the planner (decision 8): every
        rank feeds the same all-reduced counts and gets the same placement.
        """
        try:
            layer_idx = self.moe_layers.index(layer)
        except ValueError:
            return
        self._finish_pending_predict(layer_idx)
        # Wait for this layer's prefetch (if any) to finish H2D before reading
        # the device slots — mirror the single-card update_weights stream-join
        # (graphable: stream wait_event, not host sync). Without it the reactive
        # GMM could read a slot before the prefetch's load_stream H2D lands.
        with self._prefetch_state_lock:
            npu_event = self._prefetch_layer_npu_event.pop(layer_idx, None)
        if npu_event is not None:
            wait_stream = torch_npu.npu.current_stream()
            if self._pf_wait_timing:
                self._pf_wait_begin[layer_idx].record(wait_stream)
            wait_stream.wait_event(npu_event)
            if self._pf_wait_timing:
                self._pf_wait_end[layer_idx].record(wait_stream)
                self._pf_wait_armed[layer_idx] = True
        elif (self._pf_wait_timing
              and layer_idx < len(self._pf_wait_armed)
              # decode-regime calls only
              and topk_ids.size(0) <= self.offload_threshold):
            self._pf_wait_armed[layer_idx] = False
            
        num_tokens = topk_ids.size(0)

        # NOTE: the decode profile dummy must use the real dynamic placement
        # (NOT a spread shortcut) — spread maps dummy topk all to rank0 and
        # deadlocks MC2. Debug observability + the overflow-spread fallback now
        # live in _update_weights_multi_card (graph-safe: they read the pinned
        # CPU topk_ids_h, not the live NPU tensor).

        # PREFILL regime: the MC2 dispatch kernel caps at 512 tokens, so prefill
        # uses AllGather + a per-rank EP shard loaded into the prefill pool
        # (selected by select_moe_comm_method -> ALLGATHER for multi-card large
        # batches). Drive off the comm TYPE (MC2=decode, else prefill) — the
        # single source of truth — so this stays in lockstep with apply().
        if self._mc_handle_prefill_regime(layer_idx):
            return
        # ---- DECODE (MC2) branch: graph-aware (mirror single-card) ----
        # Dynamic placement varies per step (router-driven) and is incompatible
        # with cudagraph's fixed op sequence. Mirror single-card update_weights:
        # D2H router outputs into pinned CPU buffers, run planning+H2D as a host
        # callback (_launch_host_func, re-executed every replay with the current
        # topk) or inline (eager), H2D log2phy back. The host callback's
        # load_stream.synchronize() gates the compute stream until H2D is done.
        # The cross-rank expert-count all_reduce uses gloo cpu_group
        # (get_ep_group().cpu_group), while HCCL all_reduce cannot be a captured
        # graph op and the planner needs the counts on host. Every EP rank must
        # still enter the Gloo collectives in exactly the same order; MC_DEBUG
        # traces that ordering across graph host callbacks.
        per_rank_slots = self.offload_config.num_device_experts_for_rank(
            layer_idx, self.ep_size)
        topk_ids_h = self.topk_ids_h[:num_tokens]
        log2phy_h = self.log2phy_h
        topk_ids_h.copy_(topk_ids, non_blocking=_EXTRA_CTX.capturing)
        log2phy_h.copy_(log2phy, non_blocking=_EXTRA_CTX.capturing)
        do_substitution = (
            self.offload_config.expert_substitution_enabled
            and not is_hash_routed
            and router_logits is not None
            and router_logits.shape[-1] == self.num_total_experts
        )
        if (self.offload_config.expert_substitution_enabled
                and not is_hash_routed and router_logits is not None
                and router_logits.shape[-1] != self.num_total_experts):
            logger.warning_once(
                "[SUBST-MC] router_logits width %d != num_total_experts %d; "
                "expert substitution is disabled for this layer",
                router_logits.shape[-1], self.num_total_experts)
        router_logits_h = None
        correction_bias_h = None
        if do_substitution:
            router_logits_h = self.router_logits_h[:num_tokens]
            router_logits_h.copy_(
                router_logits.to(dtype=torch.float32),
                non_blocking=_EXTRA_CTX.capturing)
            if e_score_correction_bias is not None:
                correction_bias_h = self.e_score_correction_bias_h
                correction_bias_h.copy_(
                    e_score_correction_bias.to(dtype=torch.float32),
                    non_blocking=_EXTRA_CTX.capturing)
        # Mirror the per-rank active-token mask to pinned CPU on the same
        # stream as topk_ids_h, so the host callback (graph replay) reads it
        # after the copy lands — same ordering contract as topk_ids_h. None
        # means all-active (e.g. non-uniform global_bs path): no filtering,
        # fully backward compatible.
        if mc2_mask is not None:
            mc2_mask_h = self.mc2_mask_h[:num_tokens]
            # Cast bool->int32 on the NPU first: a direct bool D2H on the
            # captured stream forces a sync ("stream is captured", rtMemcpy
            # 107027) because Ascend has no async bool memcpy path. int32 D2H
            # is the same async path topk_ids_h.copy_ already uses, so it
            # records cleanly into the graph.
            mc2_mask_h.copy_(mc2_mask.to(torch.int32),
                             non_blocking=_EXTRA_CTX.capturing)
        else:
            mc2_mask_h = None
        current_compute_stream = torch_npu.npu.current_stream()
        subscribed = get_subscribed_compute_streams()
        if current_compute_stream not in subscribed:
            torch_npu.npu._subscribe_report(current_compute_stream)
            subscribed.add(current_compute_stream)
        topk_weights_h = None
        if (topk_weights is not None and self.cache_policy is not None
                and self.offload_config.cache_router_weight != 0):
            topk_weights_h = self.topk_weights_h[:num_tokens]
            topk_weights_h.copy_(topk_weights.to(dtype=torch.float32),
                                 non_blocking=_EXTRA_CTX.capturing)
        from_graph_callback = _EXTRA_CTX.capturing
        args = (
            topk_ids_h, log2phy_h, layer, layer_idx, per_rank_slots, False,
            mc2_mask_h, do_substitution, router_logits_h, scoring_func,
            correction_bias_h, topk_weights_h,
        )
        if self._debug:
            args += (from_graph_callback,)
            
        # dispatch through the guarded wrapper rather than calling the tracer directly
        if from_graph_callback:
            self._log_mc_debug_schedule(layer_idx, is_prefetch=False)
            torch_npu.npu._launch_host_func(
                current_compute_stream,
                self._update_weights_multi_card_guarded, args)
        else:
            self._update_weights_multi_card_guarded(args)
        if do_substitution:
            topk_ids[:, :self.topk].copy_(
                topk_ids_h[:, :self.topk], non_blocking=True)
        # Copy the (host-func-mutated) log2phy_h back to the NPU tensor so the
        # MC2 dispatcher reads the fresh placement.
        log2phy.copy_(log2phy_h, non_blocking=_EXTRA_CTX.capturing)
        
        # same layer-shifted dispatch as single-card update_weights.
        self._finish_next_layer_predict(layer_idx, current_compute_stream)

    def _expert_src_storage(self, layer_idx, eid, which='w13'):
        """Return expert eid's bytes as UntypedStorage for H2D **read**.

        shard-per-rank: eid is GLOBAL; remap to this rank's local shard slot.
        Otherwise: the expert tensor's own (already pinned) storage, global eid.
        """
        cpu_buf = getattr(self, f'{which}_weights_cpu')[layer_idx]
        if self.offload_config.shard_per_rank:
            local_eid = self._shard_local(eid)
            if local_eid is not None:
                return cpu_buf[local_eid].untyped_storage()
            return self._shared_h2d_source(layer_idx, eid, which)
        return cpu_buf[eid].untyped_storage()

    def _expert_dst_storage(self, layer_idx, eid, which='w13'):
        """Return expert eid's storage for fill **write**.

        shard-per-rank: eid is GLOBAL; remap to this rank's local shard slot.
        Otherwise: the expert tensor's own storage, global eid.
        """
        cpu_buf = getattr(self, f'{which}_weights_cpu')[layer_idx]
        if self.offload_config.shard_per_rank:
            return cpu_buf[eid - self._shard_base].untyped_storage()
        return cpu_buf[eid].untyped_storage()

    def _shared_h2d_source(self, layer_idx, eid, name):
        pointer = self._shared_h2d_sources.get((layer_idx, eid, name))
        if pointer is None:
            raise RuntimeError(
                "Missing MemFabric SHARED source pointer: "
                f"layer={layer_idx}, expert={eid}, name={name}")
        return HostPointerSource(pointer)

    def _shared_h2d_layer_ready(self, layer_idx: int) -> bool:
        if not getattr(self, '_shared_h2d_sources_ready', False):
            return False
        shared_sources = getattr(self, '_shared_h2d_sources', {})
        return all(
            (layer_idx, eid, name) in shared_sources
            for eid in range(self.num_total_experts)
            for name in ("w13", "w2"))

    def _publish_shared_h2d_sources(self) -> None:
        """All-gather peer shard pointers once after format conversion."""
        transport = self._get_h2d_transport()
        if not getattr(transport, "supports_remote_sources", False):
            return

        local_sources = {}
        for layer_idx in range(len(self.w13_weights_cpu)):
            for local_eid, tensor in enumerate(
                    self.w13_weights_cpu[layer_idx]):
                eid = self._shard_base + local_eid
                local_sources[(layer_idx, eid, "w13")] = tensor.data_ptr()
                local_sources[(layer_idx, eid, "w2")] = (
                    self.w2_weights_cpu[layer_idx][local_eid].data_ptr())
        for buffer_dict in (self.scale_cpu_buffers,
                            self.offset_cpu_buffers,
                            self.scale_bias_cpu_buffers):
            for name, layer_buffers in buffer_dict.items():
                for layer_idx, expert_buffers in enumerate(layer_buffers):
                    for local_eid, tensor in enumerate(expert_buffers):
                        eid = self._shard_base + local_eid
                        local_sources[(layer_idx, eid, name)] = tensor.data_ptr()

        from torch import distributed as dist
        from vllm.distributed.parallel_state import get_ep_group
        gathered_sources = [None] * self.ep_size
        dist.all_gather_object(
            gathered_sources, local_sources,
            group=get_ep_group().cpu_group)
        shared_sources = {}
        for rank_sources in gathered_sources:
            overlap = shared_sources.keys() & rank_sources.keys()
            if overlap:
                raise RuntimeError(
                    "Duplicate MemFabric SHARED expert sources: "
                    f"{list(overlap)[:4]}")
            shared_sources.update(rank_sources)
        self._shared_h2d_sources = shared_sources
        self._shared_h2d_sources_ready = True
        if not all(self._shared_h2d_layer_ready(layer_idx)
                   for layer_idx in range(len(self.w13_weights_cpu))):
            raise RuntimeError("Incomplete MemFabric SHARED source table")
        logger.info(
            "[EXPERT-OFFLOAD-H2D] published %d SHARED source pointers "
            "across %d EP ranks", len(shared_sources), self.ep_size)

    def _build_quant_attr_h2d_tasks(self, layer, layer_idx, eid,
                                    slot) -> list[H2DCopyTask]:
        """Build scale/offset/scale_bias H2D tasks for one expert slot."""
        tasks = []
        for buffer_dict in (self.scale_cpu_buffers,
                            self.offset_cpu_buffers,
                            self.scale_bias_cpu_buffers):
            for attr_name, buffers in buffer_dict.items():
                local_eid = self._shard_local(eid)
                dev_tensor = getattr(layer, attr_name, None)
                if dev_tensor is None or layer_idx >= len(buffers):
                    continue
                dst = dev_tensor.data[slot]
                if (local_eid is not None
                        and local_eid < len(buffers[layer_idx])):
                    src_tensor = buffers[layer_idx][local_eid]
                    source = src_tensor.reshape(dst.shape)
                    nbytes = src_tensor.numel() * src_tensor.element_size()
                elif getattr(self._get_h2d_transport(),
                             "supports_remote_sources", False):
                    source = self._shared_h2d_source(
                        layer_idx, eid, attr_name)
                    nbytes = dst.numel() * dst.element_size()
                else:
                    continue
                tasks.append(H2DCopyTask(
                    source=source,
                    destination=dst,
                    nbytes=nbytes,
                    name=f"{attr_name}[L{layer_idx},E{eid}->S{slot}]",
                ))
        return tasks

    def _copy_quant_attrs_into_slot(self, layer, layer_idx, eid, slot):
        """Copy one expert's quant attributes through the H2D transport."""
        self._get_h2d_transport().copy_batch(
            self._build_quant_attr_h2d_tasks(layer, layer_idx, eid, slot))

    def _apply_multi_card_substitution(
        self,
        layer_idx,
        topk_ids_h,
        router_logits_h,
        log2phy_h,
        mc2_mask_h,
        scoring_func,
        correction_bias_h,
        cpu_group=None,
    ):
        """Atomically substitute source experts across all active EP rows."""
        if mc2_mask_h is None:
            active_rows = torch.arange(topk_ids_h.shape[0])
        else:
            active_rows = mc2_mask_h.bool().nonzero(
                as_tuple=True)[0]

        original_ids = topk_ids_h.index_select(
            0, active_rows)[:, :self.topk]
        active_logits = router_logits_h.index_select(0, active_rows)
        plan = plan_expert_substitutions(
            active_logits,
            original_ids,
            log2phy_h,
            expert_substitution_threshold=(
                self.offload_config.expert_substitution_threshold),
            scoring_func=scoring_func,
            e_score_correction_bias=correction_bias_h,
        )
        from vllm_ascend.expert_offload.multi_card_planner import (
            gather_global_substitution_state_cpu,
        )

        global_referenced, global_blocked = \
            gather_global_substitution_state_cpu(
                plan.referenced, plan.blocked, cpu_group)
        allowed = global_referenced & ~global_blocked
        substituted_ids = commit_expert_substitutions(
            plan,
            allowed,
            original_ids,
        )
        if self._debug:
            self._log_expert_substitution(
                layer_idx, original_ids, substituted_ids)
        updated_ids = topk_ids_h.index_select(0, active_rows)
        updated_ids[:, :self.topk].copy_(substituted_ids)
        topk_ids_h.index_copy_(0, active_rows, updated_ids)

    def _update_weights_multi_card(self, args):
        """Trace and run one multi-card decode callback or eager update."""
        if not self._debug:
            return self._update_weights_multi_card_impl(args)
        layer_idx = args[3]
        is_prefetch = bool(args[5])
        from_graph_callback = (
            (len(args) == 8 and bool(args[7]))
            or (len(args) == 13 and bool(args[12])))
        context = self._begin_mc_debug_callback(
            layer_idx, is_prefetch, from_graph_callback)
        status = "ok"
        try:
            return self._update_weights_multi_card_impl(args, context)
        except BaseException as exc:
            status = f"error:{type(exc).__name__}"
            self._log_mc_debug_event(
                "CB_ERROR", context, error_type=type(exc).__name__)
            raise
        finally:
            self._end_mc_debug_callback(context, status)

    def _update_weights_multi_card_impl(self, args, debug_context=None):
        """Host callback (graph replay) / inline (eager) for multi-card DECODE
        placement + H2D. Reads the pinned CPU topk_ids_h, does CPU bincount +
        gloo all_reduce (cpu_group) for global expert counts, plans the
        load-balanced placement, H2D-loads misses on load_stream (synced to gate
        the compute stream), and writes placement.log2phy into the pinned
        log2phy_h (the wrapper H2D-copies it back to the NPU tensor).
        """
        if len(args) in (7, 8):
            (topk_ids_h, log2phy_h, layer, layer_idx, per_rank_slots,
             is_prefetch, mc2_mask_h) = args[:7]
            do_substitution = False
            topk_weights_h = None
        else:
            (topk_ids_h, log2phy_h, layer, layer_idx, per_rank_slots,
             is_prefetch, mc2_mask_h, do_substitution, router_logits_h,
             scoring_func, correction_bias_h, topk_weights_h) = args[:12]
        decode_start = time.perf_counter() if self._debug else None
        from vllm.distributed.parallel_state import get_ep_group

        from vllm_ascend.expert_offload.multi_card_planner import plan_placement

        cpu_group = get_ep_group().cpu_group if self.ep_size > 1 else None

        # Substitute against the previous FULL global placement before global
        # counting. Each rank changes only its active local rows; the following
        # all-reduce makes the substituted route counts globally consistent.
        if do_substitution:
            self._apply_multi_card_substitution(
                layer_idx, topk_ids_h, router_logits_h, log2phy_h,
                mc2_mask_h, scoring_func, correction_bias_h, cpu_group)

        # Drop pad-token rows (mc2_mask==0) BEFORE any counting / placing / LRU.
        # Under single-batch TP the ranks past the real-token count hold PAD
        # tokens (zero hidden) whose topk is garbage; counting them inflates
        # global_counts (-> wrong placement + wasted H2D), corrupts the LRU
        # freq, and distorts hit/miss stats. An all-pad rank contributes an
        # empty [0, topk] view -> zero local counts; the all_reduce still
        # carries the real ranks' counts, and the global placement still
        # assigns that rank the real experts MC2 dispatches to it. mc2_mask_h
        # None -> all-active (backward compatible).
        if mc2_mask_h is not None:
            active_mask = mc2_mask_h.bool()
            topk_for_count = topk_ids_h[active_mask]
            weights_for_count = (topk_weights_h[active_mask]
                                 if topk_weights_h is not None else None)
        else:
            topk_for_count = topk_ids_h
            weights_for_count = topk_weights_h

        if self._debug:
            self._log_mc_router_observation(layer_idx, topk_for_count)

        counts_start = time.perf_counter() if self._debug else None
        global_counts, cache_on, hotness, prev_log2phy = \
            self._gather_global_counts_and_hotness(layer_idx, topk_for_count,
                                                    cpu_group,
                                                    weights_for_count,
                                                    debug_context)
        counts_ms = ((time.perf_counter() - counts_start) * 1000.0
                     if counts_start is not None else 0.0)

        # MemFabric SHARED exposes peer shard pointers, so published layers may
        # use global load-balanced placement. Unpublished late layers (MTP)
        # retain owner-shard placement until their pointers are available.
        force_shard = (
            getattr(self, '_shard_size', None)
            if (self.offload_config.shard_per_rank
                and not self._shared_h2d_layer_ready(layer_idx)) else None)
        placement_start = time.perf_counter() if self._debug else None
        placement = plan_placement(global_counts, self.ep_size, per_rank_slots,
                                   prev_log2phy, hotness, force_shard=force_shard)
        placement_ms = ((time.perf_counter() - placement_start) * 1000.0
                        if placement_start is not None else 0.0)
        if cache_on:
            self._mc_prev_log2phy[layer_idx] = placement.log2phy.clone()
        if self._debug:
            self._log_mc_decode_plan(
                layer_idx, global_counts, placement, per_rank_slots,
                counts_ms, placement_ms)

        # Communication selection should conservatively keep an overflowing
        # batch out of MC2.  Reaching this point means the admission invariant
        # was violated (or configuration changed after selection).  It is too
        # late to switch collectives safely: every rank has already entered the
        # MC2 execution path.  Fail explicitly instead of mapping experts to
        # unrelated weights via a spread log2phy and silently corrupting output.
        if placement.unassigned:
            layer_capacity = per_rank_slots * self.ep_size
            cpu_mode = ("sharded" if self.offload_config.shard_per_rank
                        else "replicated")
            sample = [int(eid) for eid in placement.unassigned[
                :self._DEBUG_EXPERT_SAMPLE_LIMIT]]
            raise RuntimeError(
                "multi-card expert-offload MC2 placement overflow: "
                f"rank={self.ep_rank} layer={layer_idx} cpu_mode={cpu_mode} "
                f"global_active={int((global_counts > 0).sum())} "
                f"global_capacity={layer_capacity} "
                f"per_rank_slots={per_rank_slots} "
                f"unassigned_count={len(placement.unassigned)} "
                f"unassigned_sample={sample}. The batch should have been "
                "routed to ALLTOALL by conservative MC2 admission."
            )

        my_experts = placement.per_rank_experts[self.ep_rank]
        # active_set = this step's token topk (the NEEDED experts). Only these
        # count in the hit/miss metric — retained-but-unneeded experts stay
        # cached but aren't counted as hits — matching single-card's
        # needed-based rate so multi vs single hit rates are comparable now
        # that placement retains a persistent hot set across steps.
        active_set = (set(global_counts.nonzero(as_tuple=True)[0].tolist())
                      if global_counts is not None else None)
        resident_map, hits, misses = self._compute_resident_hits(
            layer_idx, my_experts, cache_on, active_set)
        h2d_start = time.perf_counter() if self._debug and misses else None
        if misses:
            self._h2d_load_mc_misses(layer, layer_idx, misses, resident_map,
                                     debug_context)
        h2d_ms = ((time.perf_counter() - h2d_start) * 1000.0
                  if h2d_start is not None else 0.0)
        # Write the FULL global placement into the pinned log2phy_h; the wrapper
        # H2D-copies it back to the NPU log2phy. Must be the FULL placement (not
        # just this rank) so MC2 routes tokens cross-rank correctly — writing only
        # my_experts would leave remote experts at -1 -> clamp 0 -> zero cross-
        # rank traffic -> MC2 uniform-mode dispatch deadlocks.
        log2phy_h.copy_(placement.log2phy)
        if self._debug:
            total_ms = (time.perf_counter() - decode_start) * 1000.0
            self._log_mc_decode_cache(
                layer_idx, my_experts, hits, misses, resident_map,
                placement.log2phy, per_rank_slots, is_prefetch, h2d_ms,
                total_ms)

    def _log_mc_router_observation(self, layer_idx, topk_ids_h):
        if not self._debug or not logger.isEnabledFor(logging.DEBUG):
            return
        num_tokens = topk_ids_h.size(0)
        topk = topk_ids_h.size(1) if topk_ids_h.dim() > 1 else 1
        counts = Counter(int(e) for e in topk_ids_h.reshape(-1).tolist())
        hottest = sorted(counts.items(), key=lambda x: (-x[1], x[0]))[:8]
        logger.debug(
            "[MC_OBS] rank=%s L=%s router: tokens=%s topk=%s "
            "uniq_experts=%d/%d top8(expert:count)=%s",
            self.ep_rank, layer_idx, num_tokens, topk, len(counts),
            self.num_total_experts, hottest)

    def _log_mc_decode_plan(self, layer_idx, global_counts, placement,
                            per_rank_slots, counts_ms, placement_ms):
        """Log one compact copy of the deterministic global placement plan."""
        if not self._debug or self.ep_rank != 0:
            return
        active_ids = global_counts.nonzero(as_tuple=True)[0].tolist()
        active_per_rank = [0 for _ in range(self.ep_size)]
        for eid in active_ids:
            physical_id = int(placement.log2phy[int(eid)])
            if physical_id >= 0:
                rank = physical_id // per_rank_slots
                if 0 <= rank < self.ep_size:
                    active_per_rank[rank] += 1
        assigned_per_rank = [
            sum(int(eid) >= 0 for eid in experts)
            for experts in placement.per_rank_experts
        ]
        capacity_ok = (
            not placement.unassigned
            and all(count <= per_rank_slots for count in assigned_per_rank)
        )
        cpu_mode = ("sharded" if self.offload_config.shard_per_rank
                    else "replicated")
        unassigned = [int(eid) for eid in placement.unassigned]
        logger.info(
            "[MC_OBS] rank=%s L=%s DECODE plan: cpu_mode=%s "
            "global_routes=%d global_active=%d global_capacity=%d "
            "active_per_rank=%s assigned_per_rank=%s capacity_ok=%s "
            "global_counts_checksum=%d log2phy_checksum=%d "
            "unassigned_count=%d unassigned_sample=%s per_rank_load=%s "
            "timing_ms{counts_lrc=%.3f,placement=%.3f}",
            self.ep_rank, layer_idx, cpu_mode, int(global_counts.sum()),
            len(active_ids), per_rank_slots * self.ep_size, active_per_rank,
            assigned_per_rank, capacity_ok,
            _stable_int_checksum(global_counts),
            _stable_int_checksum(placement.log2phy),
            len(unassigned), unassigned[:self._DEBUG_EXPERT_SAMPLE_LIMIT],
            placement.per_rank_load,
            counts_ms, placement_ms)

    def _log_mc_lrc_state(self, layer_idx, global_counts,
                          global_router_score, hotness):
        """Log identical-on-every-rank LRC inputs/state for verification."""
        if not self._debug or self._mc_lrc is None:
            return
        state = self._mc_lrc.layer_states[layer_idx]
        active_ids = global_counts.nonzero(as_tuple=True)[0].tolist()
        sample_limit = self._DEBUG_EXPERT_SAMPLE_LIMIT
        hottest = sorted(
            active_ids,
            key=lambda eid: (-float(hotness[eid]), int(eid)),
        )[:sample_limit]
        hot_sample = [
            {
                "expert": int(eid),
                "count": int(global_counts[eid]),
                "freq": int(state.freq[eid]),
                "ema": round(float(state.ema[eid]), 6),
                "router": round(float(state.router_score[eid]), 6),
                "hotness": round(float(hotness[eid]), 6),
            }
            for eid in hottest
        ]
        router_input_checksum = (
            _stable_float_checksum(global_router_score)
            if global_router_score is not None else None)
        logger.info(
            "[MC_LRC] rank=%s/%s L=%s step=%s global_routes=%s "
            "global_active=%s counts_checksum=%s freq_sum=%s "
            "freq_checksum=%s ema_checksum=%s router_enabled=%s "
            "router_input_checksum=%s router_state_checksum=%s "
            "hotness_checksum=%s top_hot=%s",
            self.ep_rank, self.ep_size, layer_idx, state.step,
            int(global_counts.sum()), len(active_ids),
            _stable_int_checksum(global_counts), sum(state.freq),
            _stable_int_checksum(state.freq),
            _stable_float_checksum(state.ema),
            global_router_score is not None, router_input_checksum,
            _stable_float_checksum(state.router_score),
            _stable_float_checksum(hotness), hot_sample)

    def _gather_global_counts_and_hotness(self, layer_idx, topk_ids_h,
                                          cpu_group, topk_weights_h=None,
                                          debug_context=None):
        """CPU bincount + gloo all_reduce -> global expert counts, then update
        the LRC hotness policy (the same one single-card uses: recent freq +
        EMA + age + optional router score) from the real GLOBAL route counts
        and return its per-expert hotness. global_counts and router score sums
        are all-reduced EVERY step (identical across ranks after the mc2_mask
        filter), so the LRC state + hotness are too -> placement/eviction stay
        deterministic. Gloo runs on cpu_group rather than the captured NPU
        stream, but all callbacks must still enter its collectives in the same
        order on every rank."""
        from vllm_ascend.expert_offload.multi_card_planner import (
            local_expert_counts_cpu)
        local_counts = local_expert_counts_cpu(topk_ids_h, self.num_total_experts)
        global_counts = self._gather_cpu_with_mc_debug(
            local_counts, cpu_group, "count", debug_context)
        cache_on = self.offload_config.cache_policy_enabled
        if not cache_on:
            return global_counts, cache_on, None, None
        # Reuse the configured single-card policy so every cache tuning knob
        # has identical meaning in multi-card mode.  The lazy fallback only
        # protects unusual unit-test/partial-initialization paths.
        if self._mc_lrc is None:
            self._mc_lrc = self.cache_policy
        if self._mc_lrc is None:
            return (global_counts, False, None,
                    self._mc_prev_log2phy.get(layer_idx))
        while len(self._mc_lrc.layer_states) <= layer_idx:
            self._mc_lrc.add_layer()

        global_router_score = None
        if topk_weights_h is not None:
            ids = topk_ids_h.reshape(-1).to(torch.int64)
            scores = topk_weights_h.reshape(-1).to(torch.float32)
            local_score_sum = torch.zeros(self.num_total_experts,
                                          dtype=torch.float32)
            local_score_sum.scatter_add_(0, ids, scores)
            global_score_sum = self._gather_cpu_with_mc_debug(
                local_score_sum, cpu_group, "router_score", debug_context)
            global_router_score = torch.zeros_like(global_score_sum)
            active = global_counts > 0
            global_router_score[active] = (
                global_score_sum[active] /
                global_counts[active].to(global_score_sum.dtype))
        self._mc_lrc.observe_global_counts(
            layer_idx, global_counts, global_router_score)
        hotness = self._mc_lrc.hotness_array(layer_idx)
        self._log_mc_lrc_state(
            layer_idx, global_counts, global_router_score, hotness)
        return (global_counts, cache_on, hotness,
                self._mc_prev_log2phy.get(layer_idx))

    def _compute_resident_hits(self, layer_idx, my_experts, cache_on,
                               active_set=None):
        """Split this rank's ACTIVE placed experts into cache hits (expert
        already resident in its assigned slot) vs misses (need H2D). Returns
        (resident_map, hits, misses).

        Only ACTIVE experts (in ``active_set`` = this step's token topk) are
        counted: retained-but-not-needed experts stay cached but don't count
        as hits, mirroring single-card's (needed ∩ on_device)/needed so the
        hit rate is comparable across configs. ``active_set=None`` counts all
        (backward-compatible fallback)."""
        if not cache_on:
            # No cache: every (active) expert is a miss (full H2D every step).
            misses = [(s, int(e)) for s, e in enumerate(my_experts)
                      if e >= 0 and (active_set is None or int(e) in active_set)]
            return {}, [], misses
        resident_map = self._mc_resident.setdefault(layer_idx, {})
        hits, misses = [], []
        for slot, eid in enumerate(my_experts):
            if eid < 0:
                continue
            eid = int(eid)
            if active_set is not None and eid not in active_set:
                continue  # retained but not needed this step: cached, not counted
            (hits if resident_map.get(slot) == eid else misses).append((slot, eid))
        return resident_map, hits, misses

    def _get_h2d_transport(self):
        """Return the transport, lazily initializing collective SHARED."""
        if not hasattr(self, 'h2d_transport'):
            self.h2d_transport = TorchCopyH2DTransport()
        elif self.h2d_transport is None:
            self.h2d_transport = self._create_h2d_transport()
        return self.h2d_transport

    def _allocate_expert_host_tensor(self, shape, dtype) -> torch.Tensor:
        """Allocate weight/quant storage owned by the selected H2D backend."""
        return self._get_h2d_transport().allocate_host_tensor(shape, dtype)

    def _synchronize_h2d(self) -> None:
        """Wait for the load stream and retire backend copy descriptors."""
        self._get_h2d_transport().synchronize(self.load_stream)

    def close(self) -> None:
        """Release H2D backend resources; safe to call more than once."""
        transport = getattr(self, 'h2d_transport', None)
        if transport is not None:
            transport.synchronize(self.load_stream)
            transport.close()

    def _build_expert_h2d_tasks(self, layer, layer_idx, eid,
                                slot) -> list[H2DCopyTask]:
        """Build format-preserving weight and quant H2D tasks for one expert."""
        w13_start = slot * self.w13_expert_size_bytes
        w2_start = slot * self.w2_expert_size_bytes
        w13_dst = _expert_weight(layer, "w13_weight").data.untyped_storage()[
            w13_start:w13_start + self.w13_expert_size_bytes]
        w2_dst = _expert_weight(layer, "w2_weight").data.untyped_storage()[
            w2_start:w2_start + self.w2_expert_size_bytes]
        tasks = [
            H2DCopyTask(
                source=self._expert_src_storage(layer_idx, eid, 'w13'),
                destination=w13_dst,
                nbytes=self.w13_expert_size_bytes,
                name=f"w13[L{layer_idx},E{eid}->S{slot}]",
            ),
            H2DCopyTask(
                source=self._expert_src_storage(layer_idx, eid, 'w2'),
                destination=w2_dst,
                nbytes=self.w2_expert_size_bytes,
                name=f"w2[L{layer_idx},E{eid}->S{slot}]",
            ),
        ]
        tasks.extend(
            self._build_quant_attr_h2d_tasks(layer, layer_idx, eid, slot))
        return tasks

    @staticmethod
    def _refresh_expert_fp32_scale(layer, slot):
        if hasattr(layer, 'w13_weight_scale_fp32'):
            layer.w13_weight_scale_fp32[slot].copy_(
                layer.w13_weight_scale.data[slot].to(torch.float32))

    def _load_expert_weights_into_slots(self, layer, layer_idx, loads):
        """Batch H2D-copy ``(slot, eid)`` loads, then refresh derived data."""
        loads = list(loads)
        tasks = []
        for slot, eid in loads:
            tasks.extend(
                self._build_expert_h2d_tasks(layer, layer_idx, eid, slot))
        self._get_h2d_transport().copy_batch(tasks)
        for slot, _ in loads:
            self._refresh_expert_fp32_scale(layer, slot)

    def _load_expert_weights_into_slot(self, layer, layer_idx, eid, slot):
        """Compatibility wrapper for loading one expert into one device slot."""
        self._load_expert_weights_into_slots(layer, layer_idx, [(slot, eid)])

    def _h2d_load_mc_misses(self, layer, layer_idx, misses, resident_map,
                            debug_context=None):
        """H2D-load missed experts (w13/w2 + quant attrs) into their slots on
        load_stream, then synchronize to gate the compute stream."""
        if (self._debug and self._shared_h2d_layer_ready(layer_idx)
                and misses):
            remote_count = sum(
                self._shard_local(eid) is None for _, eid in misses)
            logger.info(
                "[MEMFABRIC-SHARED-H2D] layer=%d rank=%d loads=%d "
                "remote=%d local=%d",
                layer_idx, self.ep_rank, len(misses), remote_count,
                len(misses) - remote_count)
        if not self._debug:
            with torch_npu.npu.stream(self.load_stream):
                self._load_expert_weights_into_slots(
                    layer, layer_idx, misses)
                for slot, eid in misses:
                    resident_map[slot] = eid
                self._synchronize_h2d()
            return
        start_ns = time.perf_counter_ns()
        self._log_mc_debug_event(
            "H2D_ENTER", debug_context, misses=len(misses))
        try:
            with torch_npu.npu.stream(self.load_stream):
                self._load_expert_weights_into_slots(
                    layer, layer_idx, misses)
                for slot, eid in misses:
                    resident_map[slot] = eid
                self._log_mc_debug_event(
                    "H2D_SYNC_ENTER", debug_context, misses=len(misses))
                self._synchronize_h2d()
                self._log_mc_debug_event(
                    "H2D_SYNC_EXIT", debug_context, misses=len(misses))
        except BaseException as exc:
            self._log_mc_debug_event(
                "H2D_ERROR",
                debug_context,
                misses=len(misses),
                error_type=type(exc).__name__,
            )
            raise
        elapsed_us = (time.perf_counter_ns() - start_ns) // 1000
        self._log_mc_debug_event(
            "H2D_EXIT",
            debug_context,
            misses=len(misses),
            elapsed_us=elapsed_us,
        )

    def _log_mc_decode_cache(self, layer_idx, my_experts, hits, misses,
                             resident_map, log2phy, per_rank_slots,
                             is_prefetch=False, h2d_ms=0.0, total_ms=0.0):
        if not self._debug:
            return
        expected = {
            slot: int(eid)
            for slot, eid in enumerate(my_experts) if int(eid) >= 0
        }
        resident_mismatches = {
            slot: {"expected": eid, "resident": resident_map.get(slot)}
            for slot, eid in expected.items()
            if resident_map.get(slot) != eid
        }
        mapping_mismatches = {
            eid: {
                "expected_physical": self.ep_rank * per_rank_slots + slot,
                "actual_physical": int(log2phy[eid]),
            }
            for slot, eid in expected.items()
            if int(log2phy[eid]) != self.ep_rank * per_rank_slots + slot
        }
        requests = len(hits) + len(misses)
        hit_rate = len(hits) / requests if requests else 1.0
        cpu_mode = ("sharded" if self.offload_config.shard_per_rank
                    else "replicated")
        if not misses and not resident_mismatches and not mapping_mismatches:
            return
        sample_limit = self._DEBUG_EXPERT_SAMPLE_LIMIT
        logger.info(
            "[MC_OBS] rank=%s L=%s DECODE cache: cpu_mode=%s placed=%d "
            "hit=%d miss=%d hit_rate=%.4f resident_ok=%s mapping_ok=%s "
            "h2d_load_sample=%s prefetch=%s "
            "timing_ms{h2d=%.3f,total=%.3f}",
            self.ep_rank, layer_idx, cpu_mode, len(expected), len(hits),
            len(misses), hit_rate, not resident_mismatches,
            not mapping_mismatches, misses[:sample_limit], is_prefetch,
            h2d_ms, total_ms)
        if resident_mismatches or mapping_mismatches:
            logger.warning(
                "[MC_OBS] rank=%s L=%s DECODE cache mismatch: "
                "resident_sample=%s mapping_sample=%s",
                self.ep_rank, layer_idx,
                list(resident_mismatches.items())[:sample_limit],
                list(mapping_mismatches.items())[:sample_limit])

    def _log_expert_substitution(
        self,
        layer_idx: int,
        original_ids: torch.Tensor,
        substituted_ids: torch.Tensor,
    ) -> None:
        """Log CPU-side expert replacements when offload debug is enabled."""
        if not self._debug:
            return

        changed = original_ids != substituted_ids
        changed_positions = changed.nonzero(as_tuple=False)
        replacements = [
            {
                "token": int(token_idx),
                "position": int(position),
                "original": int(original_ids[token_idx, position]),
                "substitute": int(substituted_ids[token_idx, position]),
            }
            for token_idx, position in changed_positions.tolist()
        ]
        if not replacements:
            return
        sample_limit = self._DEBUG_EXPERT_SAMPLE_LIMIT
        logger.info(
            "[SUBST] layer=%d replacement_count=%d threshold=%.4f "
            "replacement_sample=%s truncated=%d",
            layer_idx,
            len(replacements),
            self.offload_config.expert_substitution_threshold,
            replacements[:sample_limit],
            max(0, len(replacements) - sample_limit),
        )
        
    def _note_cb_failure(self, where: str) -> None:
        """Turn a report-thread exception into a log line.

        Under ACL-graph replay these callbacks run on the report thread,
        where an exception reaches neither the forward thread nor the log — the
        run just silently stops paging and stops recording statistics. The
        counter is lazily initialised so this needs no __init__ edit.
        """
        count = getattr(self, "_cb_failures", 0) + 1
        self._cb_failures = count
        if count <= 5:
            logger.exception(
                "[EXPERT-OFFLOAD] %s host callback failed (#%d) — this layer's "
                "paging did not complete", where, count)
        elif count == 6:
            logger.error(
                "[EXPERT-OFFLOAD] %s host callback keeps failing; further "
                "tracebacks suppressed", where)

    def _update_weights_guarded(self, args):
        """_update_weights, with the report-thread exception made visible."""
        try:
            self._update_weights(args)
        except Exception:
            self._note_cb_failure("update_weights")

    def _update_weights_multi_card_guarded(self, args):
        """_update_weights_multi_card, same guard.

        Note this also catches the deliberate placement-overflow RuntimeError.
        That was already being swallowed under replay; the guard only makes it
        loud. It is a hard configuration error, not a recoverable condition.
        """
        try:
            self._update_weights_multi_card(args)
        except Exception:
            self._note_cb_failure("update_weights_multi_card")

    def _update_weights(self, args):
        # The reactive form is a 9-tuple. Substitution runs on device in
        # update_weights; topk_ids_gt_h carries the pre-substitution IDs and
        # prune_debug_h carries optional pruning diagnostics. The 6-element
        # prefetch form is unchanged.
        if len(args) == 6:
            (topk_ids_h, log2phy_np, layer, layer_idx, topk_weights_h,
             is_prefetch) = args
            do_substitution = False
            topk_ids_gt_h = None
            prune_debug_h = None
        else:
            (topk_ids_h, log2phy_np, layer, layer_idx, topk_weights_h,
             is_prefetch, do_substitution, topk_ids_gt_h,
             prune_debug_h) = args

        if (self.offload_config.experts_pruning_enabled
                and not is_prefetch and prune_debug_h is not None):
            miss_routes = [
                int(x)
                for x in prune_debug_h[:, 0, :].reshape(-1).tolist()
                if int(x) >= 0
            ]
            pruned_routes = [
                int(x)
                for x in prune_debug_h[:, 1, :].reshape(-1).tolist()
                if int(x) >= 0
            ]
            remaining_routes = [
                int(x)
                for x in prune_debug_h[:, 2, :].reshape(-1).tolist()
                if int(x) >= 0
            ]
            miss_ids = sorted(set(miss_routes))
            pruned_ids = sorted(set(pruned_routes))
            saved_h2d_ids = sorted(
                set(miss_routes) - set(remaining_routes))
            prune_record = {
                "layer": layer_idx,
                "num_tokens": topk_ids_h.shape[0],
                "miss_ids": miss_ids,
                "miss_count": len(miss_ids),
                "miss_routes": len(miss_routes),
                "pruned_ids": pruned_ids,
                "pruned_unique_count": len(pruned_ids),
                "pruned_routes": len(pruned_routes),
                "remaining_miss_ids": sorted(set(remaining_routes)),
                "saved_h2d_ids": saved_h2d_ids,
                "saved_h2d_count": len(saved_h2d_ids),
            }
            logger.info(
                "[EXPERT-PRUNE-LAYER-JSON] %s",
                json.dumps(
                    prune_record,
                    separators=(",", ":"),
                    sort_keys=True,
                ),
            )

        # Resolve the collector once per call. ``collecting`` is false during
        # profile runs, warmups, and graph capture, and is re-read on replay.
        stats = self._stats if (self._stats is not None
                                and self._stats.collecting) else None
        gt_ids = None
        subst_count = 0.0
        # Substitution already happened on NPU before this callback. Only the
        # pre- and post-substitution ID buffers remain to be inspected here.
        if do_substitution and topk_ids_gt_h is not None:
            if self._debug:
                self._log_expert_substitution(
                    layer_idx, topk_ids_gt_h, topk_ids_h)
            if stats is not None:
                gt_ids = set(topk_ids_gt_h.reshape(-1).tolist())
                subst_count = float((topk_ids_gt_h != topk_ids_h).sum())
        with torch_npu.npu.stream(self.load_stream):
            # Hotness observation only on the reactive (non-prefetch) H2D path
            # with LRC policy enabled.
            def _valid_eids(ids_t):
                return [e for e in ids_t.reshape(-1).tolist()
                        if 0 <= e < self.num_total_experts]

            if not is_prefetch and self.cache_policy is not None:
                id_rows = topk_ids_h.tolist()
                router_scores = topk_weights_h.tolist() if topk_weights_h is not None else None

                needed = self.cache_policy.observe(
                    layer_idx,
                    id_rows,
                    router_scores=router_scores,
                )
            else:
                needed = set(_valid_eids(topk_ids_h))

            l2p_list = log2phy_np.tolist()
            slot_owner = {s: e for e, s in enumerate(l2p_list) if s >= 0}
            on_device = set(slot_owner.values())

            if is_prefetch:
                # Prefetch: load at most prefetch_topk experts for the layer — a
                # GLOBAL cap across all predicted token rows, not one per token —
                # chosen as the highest-scoring predictions not already resident.
                # An expert predicted by several tokens is one transfer and takes
                # its best score across them.
                rows_ids = topk_ids_h.tolist()
                rows_w = topk_weights_h.tolist() if topk_weights_h is not None else None
                n_rows = len(rows_ids)
                best_score: dict[int, float] = {}
                for r, row in enumerate(rows_ids):
                    for c, eid in enumerate(row):
                        if eid in on_device:
                            continue          # resident: nothing to transfer
                        if rows_w is not None:
                            score = rows_w[r][c]
                        else:
                            # No scores available: interleave rows by rank so
                            # every row's rank-0 outranks every row's rank-1,
                            # rather than concatenating rows — the bias this
                            # block exists to remove. At one row this is plain
                            # column order.
                            score = -float(c * n_rows + r)
                        if score > best_score.get(eid, float("-inf")):
                            best_score[eid] = score
                need_to_load = set(
                    sorted(best_score, key=best_score.get, reverse=True)
                    [:self.prefetch_topk])
            else:
                need_to_load = needed - on_device

            already_there = needed & on_device              # for cache_stats / debug

            # both hit-rate numerators must be taken HERE — the load loop
            # below mutates on_device. `needed` is the post-substitution routed
            # set; gt_set is what the router originally selected (identical when
            # substitution is off, which is what makes the two series equal in
            # that case, exactly).
            stat_hit_post = stat_hit_pre = None
            gt_set = None
            if stats is not None and not is_prefetch:
                gt_set = gt_ids if gt_ids is not None else needed
                stat_hit_post = (len(already_there) / len(needed)
                                 if needed else 0.0)
                stat_hit_pre = (len(gt_set & on_device) / len(gt_set)
                                if gt_set else 0.0)

            # Reactive pass only. On the prefetch pass `needed` is the
            # PREDICTED set, so these legacy counters were mixing predicted with
            # actual routing and their hit rate moved when prefetch was toggled.
            if self.cache_policy is not None and not is_prefetch:
                self._record_cache_stats(layer_idx, already_there, need_to_load, needed, on_device)
            reusable_slots = [s for s, e in slot_owner.items()
                            if e not in needed]          # slots to recycle

            if self._debug:
                flag = '[PREFETCH-W]' if is_prefetch else '[UPDATE-W]'
                already_there_layer = (set(_valid_eids(topk_ids_h[0:1])) & on_device)
                logger.info("%s l=%d expert_hit=%s expert_miss=%s hit_rate=%.2f layer_expert_hit=%s needed=%s topk_ids_h=%s" ,
                            flag,layer_idx, sorted(already_there),
                            # sorted(need_to_load), len(already_there_layer) / topk_ids_h.shape[1],
                            sorted(need_to_load), len(already_there) / (len(already_there) + len(need_to_load)) if len(already_there) + len(need_to_load) > 0 else 0,
                            already_there_layer, needed, topk_ids_h)
                if need_to_load and len(need_to_load) > len(reusable_slots):
                    logger.info("%s l=%d SHORTFALL: need %d load but only %d slots, "
                                "to_load=%s",
                                flag,layer_idx, len(need_to_load), len(reusable_slots),
                                sorted(need_to_load)[:20])

            # (merge): rank all resident candidates once with
            # choose_victims(count=len(need_to_load)) instead of calling
            # choose_victim() per miss.
            n_copies = 0
            planned_loads = []
            victims = None
            if self.cache_policy is not None:
                victims = iter(self.cache_policy.choose_victims(
                    layer_idx,
                    slot_owner,
                    protected=needed,
                    count=len(need_to_load),
                ))
            for eid in need_to_load:
                if self.cache_policy is not None:
                    victim = next(victims, None)
                    slot = int(log2phy_np[victim]) if victim is not None else -1
                elif reusable_slots:
                    slot = reusable_slots.pop()
                    victim = slot_owner[slot]
                else:
                    slot = -1
                    victim = None

                if slot < 0:
                    # count the shortfall
                    if stats is not None and not is_prefetch:
                        stats.note_shortfall()
                    if self._debug:
                        logger.info(
                            "[UPDATE-W] l=%d NO SLOTS: %d experts could not be loaded, "
                            "missed=%s",
                            layer_idx, len(need_to_load) - n_copies,
                            sorted(list(need_to_load))[n_copies:][:20])
                    break  # no free slots — should not happen in normal usage

                # (merge): defer the copy. The H2D is submitted as one
                # transport batch after the loop so MemFabric sparse_copy sees a single descriptor list
                planned_loads.append((slot, eid))
                # Update mapping
                if victim is None:
                    victim = slot_owner[slot]
                log2phy_np[victim] = -1             # evict old occupant
                on_device.discard(victim)
                log2phy_np[eid] = slot               # assign slot to new expert
                slot_owner[slot] = eid
                on_device.add(eid)
                if slot in reusable_slots:
                    reusable_slots.remove(slot)
                n_copies += 1

            # (merge): submit the batch first so the CPU-only stats
            # bookkeeping below overlaps the transfer, then synchronize
            self._load_expert_weights_into_slots(
                layer, layer_idx, planned_loads)

            # what was ACTUALLY transferred this pass — identical to the
            # per-expert loop's `loaded_ids`, derived from planned_loads.
            loaded_ids = [eid for _slot, eid in planned_loads]

            # hand the prefetch pass's set arithmetic across to the
            # reactive pass, then record. Placed after the load loop
            if stats is not None:
                if is_prefetch:
                    # P = predicted set, |A| = predicted-and-already-resident,
                    # N = actually transferred. G is not known on this pass.
                    # Last-writer-wins; the reactive pass for this same layer
                    # pops it later in this very forward.
                    with self._prefetch_state_lock:
                        self._prefetch_stats_pending[layer_idx] = (
                            set(needed), len(already_there), set(loaded_ids))
                else:
                    with self._prefetch_state_lock:
                        pending = self._prefetch_stats_pending.pop(
                            layer_idx, None)
                    pred_acc = pred_prec = pf_in_lrc = pf_useful = None
                    psize = pf_loads = pf_hit = pf_waste = None
                    # the compute stream's block at this layer's prefetch
                    # join. Read here because this callback is stream-ordered
                    # after the `end` record and runs on every graph replay.
                    pf_wait = (self._read_pf_wait(layer_idx) if self._pf_wait_timing else None)
                    if pending is not None:
                        predicted, n_already, transferred = pending
                        psize = float(len(predicted))
                        pf_loads = float(len(transferred))
                        overlap = len(predicted & gt_set) if gt_set else 0
                        # |N&G| computed ONCE and shared with pf_useful
                        transferred_hits = len(transferred & gt_set) if gt_set else 0
                        pf_hit = float(transferred_hits)
                        pf_waste = float(len(transferred) - transferred_hits)
                        if gt_set:
                            pred_acc = overlap / len(gt_set)
                            if predicted:
                                pred_prec = overlap / len(predicted)
                        if predicted:
                            pf_in_lrc = n_already / len(predicted)
                        if transferred:
                            # reuses transferred_hits. Still no sample when nothing was transferred.
                            pf_useful = transferred_hits / len(transferred)
                    stats.record_layer(
                        layer_idx,
                        # |G| is the layer's unique routed-expert count, which
                        # is what shows how much MTP's extra token positions
                        # overlap: 3 positions x topk 6 = 18 routing slots collapse to |G| unique experts.
                        gsize=float(len(gt_set)) if gt_set else 0.0,
                        psize=psize,
                        hit_post=stat_hit_post,
                        hit_pre=stat_hit_pre,
                        loads=float(n_copies),
                        pf_loads=pf_loads,
                        pf_hit=pf_hit,
                        pf_waste=pf_waste,
                        subst=subst_count,
                        pred_acc=pred_acc,
                        pred_prec=pred_prec,
                        pf_in_lrc=pf_in_lrc,
                        pf_useful=pf_useful,
                        pf_wait=pf_wait,
                    )

            # (merge): _synchronize_h2d() replaces load_stream.synchronize(). On the torch backend they are the same
            # call; on MemFabric it also retires the in-flight sparse-copy
            # descriptors. Still load-bearing under replay.
            self._synchronize_h2d()

    def _preload_hot_experts(self):
        """Preload each layer's top-N hot experts into device resident slots
        from offline statistics, and seed LRC hotness so they aren't evicted
        before runtime observe() builds up real stats.

        Triggered from _finalize_offload() when hot_expert_preload is on.
        Reads hot_experts_file: {"<layer_idx>": [[expert_id, weight], ...]},
        weight descending. Slot i = the i-th hottest expert. Fully rewrites
        layer.log2phy (non-preloaded experts -> -1). No-op when the switch
        is off, so default behavior is unchanged.
        """
        if not self.offload_config.hot_expert_preload:
            return
        path = self.offload_config.hot_experts_file
        if not path:
            logger.warning("[HOT-PRELOAD] hot_expert_preload=true but "
                           "hot_experts_file empty, skip")
            return
        # 相对路径相对 expert_offload 模块目录 resolve（热点 JSON 始终放该目录）
        if not os.path.isabs(path):
            path = os.path.join(os.path.dirname(__file__), path)
        import json
        with open(path) as f:
            hot = json.load(f)                  # {"<layer_idx>": [[eid,w],...]}
        if self.enable_multi_card and self.cache_policy is not None:
            # Multi-card decode uses _mc_lrc; share the already configured
            # per-layer policy so offline seeds and runtime observations evolve
            # from the same state.
            self._mc_lrc = self.cache_policy
        with torch_npu.npu.stream(self.load_stream):
            for layer_idx, layer in enumerate(self.moe_layers):
                pairs = hot.get(str(layer_idx))
                if not pairs:
                    logger.warning("[HOT-PRELOAD] l=%d missing in json, skip",
                                   layer_idx)
                    continue
                if self.enable_multi_card:
                    from vllm_ascend.expert_offload.multi_card_planner import (
                        plan_hot_preload)
                    per_rank_slots = (
                        self.offload_config.num_device_experts_for_rank(
                            layer_idx, self.ep_size))
                    force_shard = (
                        getattr(self, '_shard_size', None)
                        if (self.offload_config.shard_per_rank
                            and not self._shared_h2d_layer_ready(layer_idx))
                        else None)
                    placement = plan_hot_preload(
                        pairs,
                        global_num_experts=self.num_total_experts,
                        ep_size=self.ep_size,
                        num_device_experts=per_rank_slots,
                        force_shard=force_shard,
                    )
                    my_experts = placement.per_rank_experts[self.ep_rank]
                    for slot, eid in enumerate(my_experts):
                        if eid >= 0:
                            self._load_expert_weights_into_slot(
                                layer, layer_idx, eid, slot)
                    self.log2phy_h.copy_(placement.log2phy)
                    self._mc_prev_log2phy[layer_idx] = (
                        placement.log2phy.clone())
                    self._mc_resident[layer_idx] = {
                        slot: int(eid)
                        for slot, eid in enumerate(my_experts) if eid >= 0
                    }
                    pair_weights = {int(eid): float(weight)
                                    for eid, weight in pairs}
                    weights = {
                        eid: pair_weights[eid]
                        for eid, physical_id in enumerate(
                            placement.log2phy.tolist())
                        if physical_id >= 0
                    }
                else:
                    ndev = self.num_device_experts_for_layer(layer_idx)
                    selected_pairs = pairs[:ndev]
                    self.log2phy_np[:] = -1
                    weights = {}
                    for slot, (eid, weight) in enumerate(selected_pairs):
                        self._load_expert_weights_into_slot(
                            layer, layer_idx, eid, slot)
                        self.log2phy_np[eid] = slot
                        weights[eid] = weight
                layer.log2phy.copy_(self.log2phy_h)         # H2D writeback
                if self.cache_policy is not None:
                    self.cache_policy.seed_layer_hotness(layer_idx, weights)
                if self._debug:
                    logger.info("[HOT-PRELOAD] l=%d loaded %d hot experts, "
                                "ids=%s",
                                layer_idx, len(weights),
                                list(weights)[:20])
            self._synchronize_h2d()

    def predict_next_layer_experts_npu(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Predict which experts layer layer_idx+1 will need, on NPU.

        Runs entirely on the NPU so it can be captured in a CUDA/NPU graph.
        The returned tensors live on NPU.

        Two paths:
        - Hash-routed layers (first num_hash_layers): experts come from the
          tid2eid table indexed by token id (deterministic, 100% accurate).
        - Learned layers (the rest): softmax + topk on the gate logits of the
          first `expert_prefetch_tokens` token rows.

        Args:
            layer_idx: Current layer index.
            hidden_states: [num_tokens, hidden_dim] NPU tensor.

        Returns:
            (topk_weights, topk_ids), both [n_tok, topk] NPU tensors where
            n_tok = min(expert_prefetch_tokens, forward rows), or None if
            prediction is not possible. CHANGE: was "[1, topk], first token
            only" — that stopped being true when expert_prefetch_tokens landed.
        """
        next_idx = layer_idx + 1
        if next_idx >= len(self.moe_layers):
            return None  # last layer — nothing to prefetch

        if next_idx >= len(self._gate_weights_npu):
            return None
        gate_w = self._gate_weights_npu[next_idx]
        if gate_w is None:
            return None

        # Hash-routed layers (the first num_hash_layers in DeepSeek-V4):
        # experts are a deterministic function of the token id via the tid2eid
        # table, NOT of the gate logits. The learned softmax+topk path below
        # would predict the *wrong* experts for these layers, so look the
        # table up directly — perfect prediction, no matmul needed.
        # See md_analysis/2026-0711-1430-ds-v4前三层的topk计算规则.md §7.
        next_layer = self.moe_layers[next_idx]
        tid2eid = getattr(getattr(next_layer, "gate", None), "tid2eid", None)
        if tid2eid is not None:
            input_ids = get_forward_context().input_ids
            if input_ids is None:
                return None
            # n rows, was input_ids[:1]. min() against both the hidden
            # rows and the context's id count: they agree on the decode path
            n_tok = min(self.prefetch_tokens, hidden_states.shape[0],
                        input_ids.shape[0])
            if n_tok < 1:
                return None
            # [n_tok] on the tid2eid device -> index_select -> [n_tok, topk] int32.
            tok_ids = input_ids[:n_tok].to(tid2eid.device).long()
            topk_ids = tid2eid.index_select(0, tok_ids)
            # Selection does not depend on affinity for hash layers, so the
            # router weight is meaningless; uniform placeholders keep the
            # [n_tok, topk] shape the caller expects. NOTE this leaves every hash
            # candidate score-tied, so the single-card ranking in _update_weights
            # falls back to insertion order for these layers — harmless, since
            # tid2eid prediction is exact and any miss is equally worth fetching.
            topk_weights = torch.full(
                (n_tok, self.topk), 1.0 / self.topk,
                dtype=torch.float32, device=topk_ids.device,
            )
            return topk_weights, topk_ids

        # Previous behavior: Predict from the first token only — one representative token's
        # experts is enough for prefetch; others are handled reactively by
        # update_weights(). prefetch_topk (= min(topk, expert_prefetch_num))
        # caps how many experts are prefetched per layer — single-card uses 1
        # (cheap, conservative); raise expert_prefetch_num for more coverage.
        # New behavior TO BE CHECKED: first token may not be enough for MTP
        # Changed to n rows, was hidden_states[:1]. The [:1] was an identity slice
        # when written — without speculative decoding and at MAX_NUM_SEQS=1 the
        # decode path carries exactly one token — and only became a one-in-N
        # sample when MTP raised the forward to 1 + num_speculative_tokens rows.
        # Rows always fit the pinned staging buffers, which are
        # [offload_threshold, topk], because a prefetch is only triggered when
        # num_tokens <= offload_threshold.
        # Shape is fixed per captured graph, so this stays capture-safe.
        # On-device prediction: [n_tok, hidden_dim] x [n_experts, hidden_dim]
        n_tok = min(self.prefetch_tokens, hidden_states.shape[0])
        router_logits = F.linear(hidden_states[:n_tok].float(), gate_w)
        probs = router_logits.softmax(dim=-1)
        topk_weights, topk_ids = probs.topk(self.topk, dim=-1)
        return topk_weights, topk_ids

    def trigger_next_layer_prefetch(self, layer,
                        hidden_states: torch.Tensor | None = None) -> None:
        """Trigger next-layer expert prefetch after the GMM kernel submits.

        Graph-compatible (mirrors the reactive update_weights path — NO stream
        switch, which would break NPU capture_end): record ready_to_load_event
        on the compute stream, then _launch_host_func registers the prefetch as
        a host callback (re-run every replay). The callback runs the planner+H2D
        inner (load_stream inside gives overlap with subsequent compute) and
        records load_done_event for the next layer's reactive to stream-join.
        Eager mode keeps the prefetch-stream overlap path.

        The fork/callback/write-back/record block lives in _dispatch_prefetch,
        which both this driver and a trained predictor end in. Nothing below
        may reference topk_ids_h / next_layer / mc2_mask_h: those are elements
        of the tuple _stage_predicted_topk returns, and they are unpacked
        inside _dispatch_prefetch.
        """
        if not self.offload_config.expert_prefetch_enabled:
            return
        if self._skip_prefill:
            return
        try:
            layer_idx = self.moe_layers.index(layer)
        except ValueError:
            return
        # Arbitrate by TARGET COVERAGE, not by hash membership. This driver
        # keeps every target the trained head does not own: the hash-routed
        # layers (tid2eid is exact and free) and, for a layer-shifted method
        # like mode2_prevhfr, the first covered MoE layer, whose predecessor was
        # zero-filled during training. The two target sets stay disjoint, so
        # _prefetch_layer_npu_event never collides.
        if (self.expert_predictor is not None
                and self.expert_predictor.covers(layer_idx + 1)):
            return

        staged = self._stage_predicted_topk(layer_idx, hidden_states)
        if staged is None:
            return

        ready_to_load_event = torch_npu.npu.Event()
        torch_npu.npu.current_stream().record_event(ready_to_load_event)
        self._dispatch_prefetch(staged, ready_to_load_event)

    def _dispatch_prefetch(self, staged, ready_event) -> None:
        """Fork the prefetch stream, plan+H2D, publish log2phy, record the join.

        The tail of trigger_next_layer_prefetch, extracted so every prefetch
        method ends in one execution path. The heuristic method stages on the
        compute stream and passes an event to fork behind; a trained predictor
        stages on the prefetch stream itself and passes ready_event=None for
        layer_delta=0 (same-stream FIFO already orders the callback), or the
        on-demand-done event for layer_delta=1.

        `staged` is exactly _stage_predicted_topk's return tuple.
        """
        (topk_ids_h, topk_weights_h, log2phy_h, log2phy_np,
         next_layer, next_idx, mc2_mask_h) = staged
        with torch_npu.npu.stream(self._prefetch_stream):
            if ready_event is not None:
                self._prefetch_stream.wait_event(ready_event)
            current_compute_stream = torch_npu.npu.current_stream()
            subscribed_compute_streams = get_subscribed_compute_streams()
            if current_compute_stream not in subscribed_compute_streams:
                torch_npu.npu._subscribe_report(current_compute_stream)
                subscribed_compute_streams.add(current_compute_stream)

            prefetch_fn, prefetch_args = self._build_prefetch_call(
                topk_ids_h, topk_weights_h, log2phy_h, log2phy_np, next_layer, next_idx, mc2_mask_h)
            nxt = next_idx

            def _prefetch_host_cb(_args):
                # snapshot the staged mapping first, and restore it if
                # the callback dies. log2phy_h is mutated in place, and the H2D
                # write-back below is a recorded node that replays regardless —
                # so a partial failure would publish residency for experts whose
                # H2D never happened (log2phy applied then clamped -> slot 0,
                # silent misroute). Restoring makes the write-back a no-op.
                saved = log2phy_h.clone()
                try:
                    prefetch_fn(_args)
                except Exception:
                    log2phy_h.copy_(saved)
                    self._note_cb_failure("prefetch")

            if _EXTRA_CTX.capturing:
                if self.enable_multi_card:
                    self._log_mc_debug_schedule(
                        next_idx, is_prefetch=True)
                torch_npu.npu._launch_host_func(
                    current_compute_stream, _prefetch_host_cb, prefetch_args)
            else:
                _prefetch_host_cb(prefetch_args)

            next_layer.log2phy.copy_(log2phy_h, non_blocking=_EXTRA_CTX.capturing)
            # 记录一个传输流完成的事件，用于后续主流和它汇聚
            load_done_event = torch_npu.npu.Event()
            self._prefetch_stream.record_event(load_done_event)
            with self._prefetch_state_lock:
                self._prefetch_layer_npu_event[nxt] = load_done_event

    def _stage_predicted_topk(self, layer_idx, hidden_states):
        """Resolve the next layer, predict its experts on-device, and D2H-stage
        them into pinned buffers. Returns (topk_ids_h, topk_weights_h, log2phy_h,
        log2phy_np, next_layer, next_idx), or None if prefetch isn't possible
        (last layer / missing gate weights / prediction failed)."""
        next_idx = layer_idx + 1
        # TO BE CHECKED: was `>= len(self.moe_layers) - 1`, which excluded the LAST MoE
        # layer from being prefetched for.
        if next_idx >= len(self.moe_layers):
            return None
        predicted = self.predict_next_layer_experts_npu(layer_idx, hidden_states)
        if predicted is None:
            return None
        topk_weights, topk_ids = predicted
        next_layer = self.moe_layers[next_idx]
        num_tokens = topk_ids.size(0)
        # Prediction shape for this call. Derived per tensor rather than shared,
        # so the two branches cannot drift apart silently.
        # Changed to num_tokens is min(expert_prefetch_tokens, forward rows), no
        # longer always 1. The slice stays contiguous because k_ids == topk ==
        # the buffer's full width, and the rows fit because a prefetch only
        # triggers when num_tokens <= offload_threshold — exactly how these
        # buffers are sized. Multi-card needs nothing further: it bincounts these
        # rows into global_counts, so extra rows simply make the placement better
        # informed, with multiplicity carrying the weight.
        k_ids = topk_ids.size(1)
        topk_ids_h = self.topk_ids_h[:num_tokens, :k_ids]
        topk_ids_h.copy_(topk_ids.to(torch.int32), non_blocking=_EXTRA_CTX.capturing)
        # staged whenever the predictor produced weights, was gated on
        # the LRC policy plus a non-zero cache_router_weight. The single-card
        # load selection below now ranks candidates by router score across token
        # rows, so the scores are needed whatever the eviction policy is. This
        # copy was previously staged and never read on the single-card prefetch
        # path. _build_prefetch_call's multi-card branch builds a 7-tuple that
        # omits topk_weights_h entirely, so multi-card cannot observe this.
        topk_weights_h = None
        if topk_weights is not None:
            k_w = topk_weights.size(1)
            topk_weights_h = self.topk_weights_h[:num_tokens, :k_w]
            topk_weights_h.copy_(topk_weights.to(dtype=torch.float32),
                                 non_blocking=_EXTRA_CTX.capturing)
        log2phy_h = self._prefetch_log2phy_h
        log2phy_h.copy_(next_layer.log2phy, non_blocking=_EXTRA_CTX.capturing)
        # (multi-card): mirror the active-token mask for the rows we predicted from
        mc2_mask_h = None
        if self.enable_multi_card:
            mc2_mask = getattr(get_forward_context(), "mc2_mask", None)
            if mc2_mask is not None:
                mc2_mask_h = self.mc2_mask_h[:num_tokens]
                # int32, not bool: Ascend has no async bool D2H, and a sync here
                # would break graph capture. Same path topk_ids_h already uses.
                mc2_mask_h.copy_(mc2_mask[:num_tokens].to(torch.int32), non_blocking=_EXTRA_CTX.capturing)

        return (topk_ids_h, topk_weights_h, log2phy_h, self._prefetch_log2phy_np,
                next_layer, next_idx, mc2_mask_h)

    def _build_prefetch_call(self, topk_ids_h, topk_weights_h, log2phy_h,
                             log2phy_np, next_layer, next_idx, mc2_mask_h=None):
        """Pick the single-card vs multi-card planner+H2D inner and its args."""
        self._is_prefetch = True
        if self.enable_multi_card:
            per_rank_slots = self.offload_config.num_device_experts_for_rank(
                next_idx, self.ep_size)
            # (merge): pass the real mc2_mask, not None.
            args = (
                topk_ids_h, log2phy_h, next_layer, next_idx, per_rank_slots,
                True, mc2_mask_h)
            if getattr(self, "_debug", False):
                args += (_EXTRA_CTX.capturing,)
            return self._update_weights_multi_card, args
        # SINGLE-CARD. Unchanged
        return self._update_weights, (
            topk_ids_h, log2phy_np, next_layer, next_idx, topk_weights_h,
            self._is_prefetch)

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _record_cache_stats(
        self,
        layer_idx: int,
        hit_experts: set[int],
        miss_experts: set[int],
        needed: set[int],
        on_device: set[int],
    ):
        self.cache_calls[layer_idx] += 1
        self.cache_requests[layer_idx] += len(needed)
        self.cache_hits[layer_idx] += len(hit_experts)
        self.cache_misses[layer_idx] += len(miss_experts)

        interval = self.offload_config.cache_stats_log_interval
        if interval == 0 or self.cache_calls[layer_idx] % interval != 0:
            return

        requests = self.cache_requests[layer_idx]
        hit_rate = self.cache_hits[layer_idx] / requests if requests else 0.0
        policy_step = -1
        if self.cache_policy is not None:
            policy_step = self.cache_policy.layer_step(layer_idx)
        # (merge): the two sorted() calls sit here rather than at the top of the function.
        self.last_hit_experts[layer_idx] = sorted(hit_experts)
        self.last_miss_experts[layer_idx] = sorted(miss_experts)
        if self._debug:
            logger.info(
                "[EXPERT-OFFLOAD-CACHE] layer=%d cache_step=%d calls=%d policy_step=%d "
                "hit_rate=%.4f hits=%d misses=%d last_hit=%s last_miss=%s resident=%s",
                layer_idx,
                self.cache_calls[layer_idx],
                self.cache_calls[layer_idx],
                policy_step,
                hit_rate,
                self.cache_hits[layer_idx],
                self.cache_misses[layer_idx],
                self.last_hit_experts[layer_idx],
                self.last_miss_experts[layer_idx],
                sorted(on_device),
            )



_EXPERT_OFFLOAD_MANAGER: ExpertOffloadManager = None


def maybe_init_expert_offload_manager(vllm_config: VllmConfig):
    # if no need to init offload manager:
    #     return
    global _EXPERT_OFFLOAD_MANAGER
    if _EXPERT_OFFLOAD_MANAGER is None:
        _EXPERT_OFFLOAD_MANAGER = ExpertOffloadManager(vllm_config)


def has_expert_offload_manager():
    return _EXPERT_OFFLOAD_MANAGER is not None


def get_expert_offload_manager():
    assert _EXPERT_OFFLOAD_MANAGER is not None, (
        "Expert Offload Manager is not initialized"
    )
    return _EXPERT_OFFLOAD_MANAGER
