"""Expert Offload Manager — manages CPU-side expert weights and NPU paging."""

import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import torch
import torch_npu
import torch.nn.functional as F
from vllm.config import VllmConfig
from vllm.logger import logger

from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.expert_offload.lrc_policy import LRCExpertCachePolicy
from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ


_SUBSCRIBED_COMPUTE_STREAMS = set()
def get_subscribed_compute_streams() -> set:
    return _SUBSCRIBED_COMPUTE_STREAMS


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
    _LOAD_POOL_DRAIN_EVERY = 2048

    @classmethod
    def get_instance(cls) -> "ExpertOffloadManager":
        assert cls._instance is not None, "ExpertOffloadManager not initialized"
        return cls._instance

    def __init__(self, vllm_config: VllmConfig):
        from vllm_ascend.ascend_config import get_ascend_config

        self.offload_config = get_ascend_config().expert_offload_config
        self.num_device_experts = self.offload_config.num_device_experts
        self.topk = vllm_config.model_config.hf_config.num_experts_per_tok
        self.offload_threshold = self.num_device_experts // self.topk

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
        # Master debug switch for expert-offload diagnostics — UPDATE-W cache
        # trace, per-prefill-load logs, prefetch/update slot shortfalls.
        # Flipping it on surfaces them at info level (no need for global
        # VLLM_LOGGING_LEVEL=DEBUG).
        self._debug = self.offload_config.moe_offload_debug

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

        # Prefill pool: ndl layers × all experts on NPU, shared round-robin
        self._prefill_w13: list[torch.Tensor] = []
        self._prefill_w2: list[torch.Tensor] = []
        self._prefill_w13_scale: list[torch.Tensor] = []       # W8A8
        self._prefill_w13_scale_fp32: list[torch.Tensor] = []   # W8A8
        self._prefill_w13_offset: list[torch.Tensor] = []       # W8A8
        self._prefill_w2_scale: list[torch.Tensor] = []         # W8A8
        self._prefill_w2_offset: list[torch.Tensor] = []        # W8A8
        self._prefill_log2phy: torch.Tensor = None              # identity [0..127]
        self._prefill_initialized: bool = False
        self._skip_prefill: bool = False  # set during profile runs

        # Next-layer expert prefetch infrastructure
        self._prefetch_stream = torch_npu.npu.Stream()
        self._gate_weights_cpu: list[torch.Tensor | None] = []
        # NPU copy of gate weights for graph-capturable on-device prediction.
        # Kept in fp32 to match the CPU prediction path.
        self._gate_weights_npu: list[torch.Tensor | None] = []

        # Threaded prefetch: daemon thread processes prefetch requests
        # so the main forward-pass thread is never blocked by H2D copies.
        self._prefetch_queue: queue.Queue = queue.Queue()
        self._prefetch_thread_ready: threading.Event = threading.Event()
        self._prefetch_state_lock = threading.Lock()
        self._prefetch_layer_done: dict[int, threading.Event] = {}
        self._prefetch_layer_npu_event: dict[int, torch_npu.npu.Event] = {}
        self._prefetch_thread: threading.Thread | None = None
        self._npu_device: torch.device | None = None

        # Pinned CPU staging buffers for graph-mode prefetch prediction.
        # Allocated lazily in _finalize_offload (hidden_dim / num_total_experts
        # are only known after MoE layers are registered).  Mirror
        # update_weights' non_blocking D2H + stream-order-ready pattern: the
        # graph host callback reads these (already-ready) buffers instead of
        # doing a blocking .cpu() on a live graph tensor.
        self._prefetch_hs_h: torch.Tensor | None = None
        self._prefetch_log2phy_h: torch.Tensor | None = None
        self._prefetch_log2phy_np = None

    # ------------------------------------------------------------------ #
    #  Lifecycle: called during model init and after weight loading       #
    # ------------------------------------------------------------------ #

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

        params_dtype = layer.w13_weight.dtype
        # Use logical dimensions (layer.hidden_size / intermediate_size)
        # rather than device tensor shapes.  Device tensor layout may be
        # transposed before process_weights_after_loading (e.g. W8A8
        # stores [intermediate, hidden] pre-transpose), confusing the
        # per-expert shape derivation.
        w13_shape = (layer.hidden_size, 2 * layer.intermediate_size_per_partition)
        w2_shape = (layer.intermediate_size_per_partition, layer.hidden_size)

        w13_list = [
            torch.empty(w13_shape, dtype=params_dtype, device="cpu", pin_memory=True)
            for _ in range(ntotal)
        ]
        w2_list = [
            torch.empty(w2_shape, dtype=params_dtype, device="cpu", pin_memory=True)
            for _ in range(ntotal)
        ]
        self.w13_weights_cpu.append(w13_list)
        self.w2_weights_cpu.append(w2_list)

        # Per-expert storage size in bytes. The expert shape is uniform across
        # layers (asserted above), so this is set unconditionally on the first
        # layer and reused. Used for raw-storage slicing during NZ paging.
        self.w13_expert_size_bytes = w13_list[0].nelement() * w13_list[0].element_size()
        self.w2_expert_size_bytes = w2_list[0].nelement() * w2_list[0].element_size()

        # Scale / offset CPU buffers (W8A8)
        self._init_layer_scale_buffers(layer, layer_moe_idx, ntotal)

        self.moe_layers.append(layer)

    def _init_layer_scale_buffers(self, layer, layer_moe_idx: int,
                                   ntotal: int):
        """Allocate CPU scale/offset buffers for a single MoE layer."""
        attr_specs = [
            ("scale_cpu_buffers", "w13_weight_scale"),
            ("scale_cpu_buffers", "w2_weight_scale"),
            ("offset_cpu_buffers", "w13_weight_offset"),
            ("offset_cpu_buffers", "w2_weight_offset"),
        ]
        for buffer_dict_name, attr_name in attr_specs:
            if not hasattr(layer, attr_name):
                continue
            dev_tensor = getattr(layer, attr_name)
            # Match the device slot shape the buffer is paged into. The W8A8
            # path flattens each expert's scale/offset to 1D (.view(E, -1)) in
            # its process_weights_after_loading, which runs AFTER this alloc
            # (pre-flatten, shape [.., 1]). Allocate 1D so the buffer already
            # matches the post-flatten device slot — no reshape at copy time.
            per_expert_shape = (dev_tensor[0].numel(),)
            dtype = dev_tensor.dtype
            buffer_dict: dict = getattr(self, buffer_dict_name)
            if attr_name not in buffer_dict:
                buffer_dict[attr_name] = []
            buffers = buffer_dict[attr_name]
            while len(buffers) <= layer_moe_idx:
                buffers.append([])
            for _ in range(ntotal):
                buffers[layer_moe_idx].append(
                    torch.empty(per_expert_shape, dtype=dtype,
                                device="cpu", pin_memory=True))

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
        t2 = time.perf_counter()

        num_moe_layers = len(self.moe_layers)
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
                cache_size=self.num_device_experts,
                topk=self.topk,
                recent_window=self.offload_config.cache_recent_window,
                ema_beta=self.offload_config.cache_ema_beta,
                recent_weight=self.offload_config.cache_recent_weight,
                ema_weight=self.offload_config.cache_ema_weight,
                router_weight=self.offload_config.cache_router_weight,
                age_weight=self.offload_config.cache_age_weight,
            )
        t3 = time.perf_counter()

        ntotal = self.num_total_experts
        self.topk_ids_h = torch.zeros(
            [self.offload_threshold, self.topk],
            dtype=torch.int32, device="cpu", pin_memory=True)
        self.topk_weights_h = torch.zeros(
            [self.offload_threshold, self.topk],
            dtype=torch.float32, device="cpu", pin_memory=True)
        self.log2phy_h = torch.zeros(ntotal, dtype=torch.int32,
                                     device='cpu', pin_memory=True)
        self.log2phy_np = self.log2phy_h.numpy()
        t4 = time.perf_counter()

        self.refresh_fp32_scales()
        t5 = time.perf_counter()
        self.create_prefill_pool()
        t6 = time.perf_counter()
        if self.offload_config.expert_prefetch_enabled:
            self.register_gate_weights(model)
            # Pinned staging buffers for graph-mode host-callback prefetch.
            # The callback runs on the compute stream's host thread during
            # graph replay, where blocking .cpu() on a live graph tensor would
            # deadlock (see md_anlysis/2026-0615-1416-...).  These buffers let
            # trigger_next_layer_prefetch stage the data with non_blocking D2H
            # *before* launching the callback, exactly like update_weights.
            hidden_dim = self.moe_layers[0].hidden_size
            self._prefetch_hs_h = torch.zeros(
                [self.offload_threshold, hidden_dim],
                dtype=torch.float32, device='cpu', pin_memory=True)
            self._prefetch_log2phy_h = torch.zeros(
                self.num_total_experts, dtype=torch.int32,
                device='cpu', pin_memory=True)
            self._prefetch_log2phy_np = self._prefetch_log2phy_h.numpy()
        t7 = time.perf_counter()
        logger.info(
            "[OFFLOAD] finalize breakdown: process_weights=%.1fs "
            "cache_policy=%.1fs buffers=%.1fs init_device=%.1fs "
            "prefill_pool=%.1fs gate=%.1fs | total=%.1fs",
            t2 - t1, t3 - t2, t4 - t3, t5 - t4, t6 - t5, t7 - t6, t7 - t0)

    def process_weights_after_loading(self):
        """Convert resident CPU expert buffers to fractal NZ format (W8A8).

        For W8A8 the device weight lives in NZ format, so we mirror that on
        the CPU side and page experts to the device with a raw copy_ on the
        underlying storage (avoids an implicit format cast on every H2D).

        After this runs each w13/w2 CPU tensor still reports its original
        [hidden, ...] shape, but its storage holds NZ-format bytes — a "liar
        tensor". Touch it only via untyped_storage() slicing, never through
        the tensor view. No-op for non-int8 models.
        """
        first_w13 = self.w13_weights_cpu[0][0]
        if first_w13.dtype != torch.int8:
            return
        num_moe_layers = len(self.w13_weights_cpu)
        num_experts = len(self.w13_weights_cpu[0])
        for layer_id in range(num_moe_layers):
            w13 = torch.stack(self.w13_weights_cpu[layer_id]).to('npu')
            w13_nz = torch_npu.npu_format_cast(w13, ACL_FORMAT_FRACTAL_NZ)
            w13_nz_storage = w13_nz.untyped_storage()
            w2 = torch.stack(self.w2_weights_cpu[layer_id]).to('npu')
            w2_nz = torch_npu.npu_format_cast(w2, ACL_FORMAT_FRACTAL_NZ)
            w2_nz_storage = w2_nz.untyped_storage()
            for expert_id in range(num_experts):
                self.w13_weights_cpu[layer_id][expert_id].untyped_storage().copy_(
                    w13_nz_storage[expert_id * self.w13_expert_size_bytes : (expert_id + 1) * self.w13_expert_size_bytes]
                )
                self.w2_weights_cpu[layer_id][expert_id].untyped_storage().copy_(
                    w2_nz_storage[expert_id * self.w2_expert_size_bytes : (expert_id + 1) * self.w2_expert_size_bytes]
                )

    def register_gate_weights(self, model):
        """Store fp32 CPU copies of gate.weight for each MoE layer.

        Called from _finalize_offload() after all MoE layers are
        registered.  The gate weights are used by predict_next_layer_experts()
        to predict which experts the next layer will need.
        """
        from vllm_ascend.models.deepseek_v4 import DeepseekV4MoE
        moe_wrappers = [m for m in model.modules()
                        if isinstance(m, DeepseekV4MoE)]
        for wrapper in moe_wrappers:
            gate_cpu = wrapper.gate.weight.data.cpu().float().clone()
            self._gate_weights_cpu.append(gate_cpu)
        logger.info("[PREFETCH] registered gate weights for %d MoE layers",
                    len(self._gate_weights_cpu))

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
        for f in self._load_futures:
            f.result()
        self._load_futures.clear()

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
        assembled = torch.cat([w1, w3], dim=0).reshape(target.shape)
        target.copy_(assembled)

    @staticmethod
    def _copy_scale_direct(target: torch.Tensor, owned: torch.Tensor) -> None:
        target.copy_(owned.reshape(target.shape))

    # ------------------------------------------------------------------ #
    #  Weight-load entry points (called by the safetensors loader)        #
    # ------------------------------------------------------------------ #

    def register_gate_weights(self, model):
        """Store fp32 CPU and NPU copies of gate.weight for each MoE layer.

        Called from _register_offload_layers() after all MoE layers are
        registered.  The CPU gate weights are used by the legacy
        predict_next_layer_experts(); the NPU copies are used by
        predict_next_layer_experts_npu() so prediction can run on-device
        and be captured in a CUDA/NPU graph.
        """
        from vllm_ascend.models.deepseek_v4 import DeepseekV4MoE
        moe_wrappers = [m for m in model.modules()
                        if isinstance(m, DeepseekV4MoE)]
        for wrapper in moe_wrappers:
            gate_param = wrapper.gate.weight.data
            gate_cpu = gate_param.cpu().float().clone()
            self._gate_weights_cpu.append(gate_cpu)
            # Place on the same NPU device as the parameter for on-device
            # graph-capturable prediction.
            self._gate_weights_npu.append(gate_cpu.to(gate_param.device))
        logger.info("[PREFETCH] registered gate weights for %d MoE layers",
                    len(self._gate_weights_cpu))

    def load_w13(self, layer_moe_idx: int, expert_id: int,
                 loaded_weight: torch.Tensor, shard_id: str):
        """Store w1/w3 shard to CPU buffer (transposed) via the load pool."""
        self._weight_load_calls += 1
        cpu = self.w13_weights_cpu[layer_moe_idx][expert_id]
        intermed = cpu.shape[1] // 2
        # Own the bytes now (the mmap may unmap before the worker runs).
        owned = loaded_weight.cpu().clone()
        fut = self._get_load_pool().submit(
            self._copy_w13_shard, cpu, owned, shard_id, intermed)
        self._track_load_future(fut)

    def load_w2(self, layer_moe_idx: int, expert_id: int,
                loaded_weight: torch.Tensor):
        """Store w2 weight to CPU buffer (transposed) via the load pool."""
        self._weight_load_calls += 1
        dst = self.w2_weights_cpu[layer_moe_idx][expert_id]
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
        target_dict = (self.scale_cpu_buffers if "scale" in attr_name
                       else self.offset_cpu_buffers)
        target = target_dict[attr_name][layer_moe_idx][expert_id]
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
            ndev = min(self.num_device_experts, layer.w13_weight.shape[0])
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
        dev = pool_layer.w13_weight.device
        dt = pool_layer.w13_weight.dtype
        ntotal = self.num_total_experts

        for _ in range(ndl):
            # w13: [ntotal, hidden_size, w13_up_dim] — match decode layer shape
            w13_shape = (ntotal,) + tuple(pool_layer.w13_weight.shape[1:])
            self._prefill_w13.append(
                torch.empty(w13_shape, dtype=dt, device=dev))

            # w2: [ntotal, hidden_size, intermediate_size_per_partition]
            w2_shape = (ntotal,) + tuple(pool_layer.w2_weight.shape[1:])
            self._prefill_w2.append(
                torch.empty(w2_shape, dtype=dt, device=dev))

            # W8A8 scale/offset (optional)
            if hasattr(pool_layer, 'w13_weight_scale'):
                s13_shape = (ntotal,) + tuple(pool_layer.w13_weight_scale.shape[1:])
                self._prefill_w13_scale.append(
                    torch.empty(s13_shape, dtype=pool_layer.w13_weight_scale.dtype, device=dev))
            if hasattr(pool_layer, 'w13_weight_scale_fp32'):
                fp32_13_shape = (ntotal,) + tuple(pool_layer.w13_weight_scale_fp32.shape[1:])
                self._prefill_w13_scale_fp32.append(
                    torch.empty(fp32_13_shape, dtype=torch.float32, device=dev))
            if hasattr(pool_layer, 'w13_weight_offset'):
                o13_shape = (ntotal,) + tuple(pool_layer.w13_weight_offset.shape[1:])
                self._prefill_w13_offset.append(
                    torch.empty(o13_shape, dtype=pool_layer.w13_weight_offset.dtype, device=dev))
            if hasattr(pool_layer, 'w2_weight_scale'):
                s2_shape = (ntotal,) + tuple(pool_layer.w2_weight_scale.shape[1:])
                self._prefill_w2_scale.append(
                    torch.empty(s2_shape, dtype=pool_layer.w2_weight_scale.dtype, device=dev))
            if hasattr(pool_layer, 'w2_weight_offset'):
                o2_shape = (ntotal,) + tuple(pool_layer.w2_weight_offset.shape[1:])
                self._prefill_w2_offset.append(
                    torch.empty(o2_shape, dtype=pool_layer.w2_weight_offset.dtype, device=dev))

        # Cast prefill pool weight tensors to NZ format (W8A8 kernel requires it).
        # Must happen BEFORE loading data — same order as decode path:
        # create → NZ-cast → copy_(cpu → npu)
        if dt == torch.int8:
            from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ
            for i in range(ndl):
                self._prefill_w13[i] = torch_npu.npu_format_cast(
                    self._prefill_w13[i], ACL_FORMAT_FRACTAL_NZ)
                self._prefill_w2[i] = torch_npu.npu_format_cast(
                    self._prefill_w2[i], ACL_FORMAT_FRACTAL_NZ)

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

    def _init_prefill_pool_data(self, dev, ntotal: int, ndl: int):
        """Load layer 0 weights into all prefill pool slots.

        Prefill pool tensors are already NZ-cast at this point (done in
        create_prefill_pool). Use simple per-expert copy_() — same pattern
        as the decode path's _update_weights.
        """
        has_scales = bool(self._prefill_w13_scale)
        has_offsets = bool(self._prefill_w13_offset)

        for slot in range(ndl):
            for eid in range(min(ntotal, len(self.w13_weights_cpu[0]))):
                self._prefill_w13[slot].untyped_storage()[eid * self.w13_expert_size_bytes : (eid + 1) * self.w13_expert_size_bytes].copy_(
                    self.w13_weights_cpu[0][eid].untyped_storage()
                )
                self._prefill_w2[slot].untyped_storage()[eid * self.w2_expert_size_bytes : (eid + 1) * self.w2_expert_size_bytes].copy_(
                    self.w2_weights_cpu[0][eid].untyped_storage()
                )

            # Initialize scale/offset buffers with layer 0 data (W8A8)
            if has_scales:
                for scale_name, prefill_list, cpu_buffers in [
                    ("w13_weight_scale", self._prefill_w13_scale, self.scale_cpu_buffers),
                    ("w2_weight_scale", self._prefill_w2_scale, self.scale_cpu_buffers),
                ]:
                    if (scale_name in cpu_buffers and
                            0 < len(cpu_buffers[scale_name])):
                        for eid in range(min(ntotal, len(cpu_buffers[scale_name][0]))):
                            prefill_list[slot][eid].copy_(
                                cpu_buffers[scale_name][0][eid])
            if has_offsets:
                for offset_name, prefill_list, cpu_buffers in [
                    ("w13_weight_offset", self._prefill_w13_offset, self.offset_cpu_buffers),
                    ("w2_weight_offset", self._prefill_w2_offset, self.offset_cpu_buffers),
                ]:
                    if (offset_name in cpu_buffers and
                            0 < len(cpu_buffers[offset_name])):
                        for eid in range(min(ntotal, len(cpu_buffers[offset_name][0]))):
                            prefill_list[slot][eid].copy_(
                                cpu_buffers[offset_name][0][eid])
            # Initialize fp32 scale (convert from scale)
            if has_scales and slot < len(self._prefill_w13_scale_fp32):
                for eid in range(min(ntotal, self._prefill_w13_scale[slot].shape[0])):
                    self._prefill_w13_scale_fp32[slot][eid].copy_(
                        self._prefill_w13_scale[slot][eid].to(torch.float32))

    def _prefill_load_layer(self, layer_idx: int, log2phy: torch.Tensor):
        """Load ALL experts for model layer layer_idx into the prefill pool.

        For W8A8: loads into normal-format scratch, then casts to NZ.
        For unquantized: loads directly into pool tensors via copy_().
        Full-overwrite into pool_slot = layer_idx % ndl.  No slot_owner
        tracking needed — log2phy is set to identity for prefill.
        """
        ndl = self.num_device_layers
        pool_slot = layer_idx % ndl
        dev = self._prefill_w13[pool_slot].device
        ntotal = self.num_total_experts
        is_w8a8 = self._prefill_w13[pool_slot].dtype == torch.int8

        if self._debug:
            logger.info("[PREFILL_LOAD] layer=%d pool_slot=%d ntotal=%d is_w8a8=%s",
                        layer_idx, pool_slot, ntotal, is_w8a8)

        from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ

        with torch_npu.npu.stream(self.load_stream):
            for eid in range(ntotal):
                self._prefill_w13[pool_slot].untyped_storage()[eid * self.w13_expert_size_bytes : (eid + 1) * self.w13_expert_size_bytes].copy_(
                    self.w13_weights_cpu[layer_idx][eid].untyped_storage()
                )
                self._prefill_w2[pool_slot].untyped_storage()[eid * self.w2_expert_size_bytes : (eid + 1) * self.w2_expert_size_bytes].copy_(
                    self.w2_weights_cpu[layer_idx][eid].untyped_storage()
                )

            # W8A8 scale/offset — load into prefill buffers
            for scale_name, prefill_list, cpu_buffers in [
                ("w13_weight_scale", self._prefill_w13_scale, self.scale_cpu_buffers),
                ("w2_weight_scale", self._prefill_w2_scale, self.scale_cpu_buffers),
            ]:
                if pool_slot < len(prefill_list):
                    if (scale_name in cpu_buffers and
                            layer_idx < len(cpu_buffers[scale_name])):
                        for eid in range(min(ntotal, len(cpu_buffers[scale_name][layer_idx]))):
                            prefill_list[pool_slot][eid].copy_(
                                cpu_buffers[scale_name][layer_idx][eid])
            for offset_name, prefill_list, cpu_buffers in [
                ("w13_weight_offset", self._prefill_w13_offset, self.offset_cpu_buffers),
                ("w2_weight_offset", self._prefill_w2_offset, self.offset_cpu_buffers),
            ]:
                if pool_slot < len(prefill_list):
                    if (offset_name in cpu_buffers and
                            layer_idx < len(cpu_buffers[offset_name])):
                        for eid in range(min(ntotal, len(cpu_buffers[offset_name][layer_idx]))):
                            prefill_list[pool_slot][eid].copy_(
                                cpu_buffers[offset_name][layer_idx][eid])

            # Refresh fp32 scale for prefill pool
            if (pool_slot < len(self._prefill_w13_scale_fp32) and
                    pool_slot < len(self._prefill_w13_scale)):
                # Copy scale data from freshly loaded scale to fp32
                for eid in range(min(ntotal, self._prefill_w13_scale[pool_slot].shape[0])):
                    self._prefill_w13_scale_fp32[pool_slot][eid].copy_(
                        self._prefill_w13_scale[pool_slot][eid].to(torch.float32))

            self.load_stream.synchronize()

        # NOTE: Do NOT modify the layer's own log2phy here — decode path
        # relies on it staying with 32-expert mapping.  Prefill path in
        # apply() explicitly uses self._prefill_log2phy instead.

    # ------------------------------------------------------------------ #
    #  Forward path: page in experts based on topk_ids                    #
    # ------------------------------------------------------------------ #

    def update_weights(self, layer, topk_ids: torch.Tensor,
                        log2phy: torch.Tensor,
                        topk_weights: torch.Tensor | None = None,
                        hidden_states: torch.Tensor | None = None) -> int:
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
        num_tokens = topk_ids.size(0)
        if num_tokens > self.offload_threshold:
            # Prefill: layerwise reuse + full-overwrite of all experts
            if (self._prefill_initialized
                    and not self._skip_prefill):
                try:
                    layer_idx = self.moe_layers.index(layer)
                except ValueError:
                    return 0
                self._prefill_load_layer(layer_idx, log2phy)
                return 0
            else:
                # Profile run or pool not ready — bail out gracefully
                return 0

        try:
            layer_idx = self.moe_layers.index(layer)
        except ValueError:
            return 0

        # Wait for any pending threaded prefetch for this layer to complete.
        # In graph mode there is no background thread; only the NPU event
        # stored by the host callback needs to be waited on.
        layer_done = self._prefetch_layer_done.get(layer_idx)
        if layer_done is not None:
            layer_done.wait()           # Block until prefetch thread finishes
            layer_done.clear()
            del self._prefetch_layer_done[layer_idx]

        # Wait for prefetch NPU copies to complete before using the weights.
        # Use stream wait (graphable) instead of host synchronize.
        with self._prefetch_state_lock:
            npu_event = self._prefetch_layer_npu_event.pop(layer_idx, None)
        if npu_event is not None:
            torch_npu.npu.current_stream().wait_event(npu_event)

        topk_ids_h = self.topk_ids_h[:num_tokens]
        topk_weights_h = None
        if (self.cache_policy is not None and topk_weights is not None 
                and self.offload_config.cache_router_weight != 0):
            topk_weights_h = self.topk_weights_h[:num_tokens]
            topk_weights_h.copy_(topk_weights.to(dtype=torch.float32), non_blocking=_EXTRA_CTX.capturing)
        log2phy_h = self.log2phy_h
        log2phy_np = self.log2phy_np
        topk_ids_h.copy_(topk_ids, non_blocking=_EXTRA_CTX.capturing)
        log2phy_h.copy_(log2phy, non_blocking=_EXTRA_CTX.capturing)

        current_compute_stream = torch_npu.npu.current_stream()
        subscribed_compute_streams = get_subscribed_compute_streams()
        if current_compute_stream not in subscribed_compute_streams:
            torch_npu.npu._subscribe_report(current_compute_stream)
            subscribed_compute_streams.add(current_compute_stream)

        args = (
            topk_ids_h,
            log2phy_np,
            layer,
            layer_idx,
            topk_weights_h,
        )
        if _EXTRA_CTX.capturing:
            torch_npu.npu._launch_host_func(
                current_compute_stream,
                self._update_weights,
                args,
            )
        else:
            self._update_weights(args)

        log2phy.copy_(log2phy_h, non_blocking=_EXTRA_CTX.capturing)

    def _update_weights(self, args):
        (
            topk_ids_h,
            log2phy_np,
            layer,
            layer_idx,
            topk_weights_h,
        ) = args
        with torch_npu.npu.stream(self.load_stream):
            if self.cache_policy is not None:
                router_scores = topk_weights_h.tolist() if topk_weights_h is not None else None
                needed = self.cache_policy.observe(
                    layer_idx,
                    topk_ids_h.tolist(),
                    router_scores=router_scores,
                )
            else:
                needed = set(topk_ids_h.unique().tolist())

            # Build reverse map: slot → expert_id currently occupying it
            slot_owner: dict[int, int] = {}
            for eid, slot in enumerate(log2phy_np):
                if slot >= 0:
                    slot_owner[slot] = eid

            on_device = set(slot_owner.values())
            already_there = needed & on_device           # no-op
            need_to_load = needed - already_there          # CPU→NPU copy
            if self.cache_policy is not None:
                self._record_cache_stats(layer_idx, already_there, need_to_load, needed, on_device)
            reusable_slots = [s for s, e in slot_owner.items()
                            if e not in needed]          # slots to recycle

            if self._debug:
                logger.info("[UPDATE-W] l=%d expert_hit=%s expert_miss=%s hit_rate=%.2f",
                            layer_idx, sorted(already_there),
                            sorted(need_to_load), len(already_there) / 6)
                if need_to_load and len(need_to_load) > len(reusable_slots):
                    logger.info("[UPDATE-W] l=%d SHORTFALL: need %d load but only %d slots, "
                                "to_load=%s",
                                layer_idx, len(need_to_load), len(reusable_slots),
                                sorted(need_to_load)[:20])

            dev = layer.w13_weight.device
            n_copies = 0
            for eid in need_to_load:
                if self.cache_policy is not None:
                    victim = self.cache_policy.choose_victim(
                        layer_idx,
                        slot_owner,
                        protected=needed,
                    )
                    slot = int(log2phy_np[victim]) if victim is not None else -1
                elif reusable_slots:
                    slot = reusable_slots.pop()
                    victim = slot_owner[slot]
                else:
                    slot = -1
                    victim = None

                if slot < 0:
                    if self._debug:
                        logger.info(
                            "[UPDATE-W] l=%d NO SLOTS: %d experts could not be loaded, "
                            "missed=%s",
                            layer_idx, len(need_to_load) - n_copies,
                            sorted(list(need_to_load))[n_copies:][:20])
                    break  # no free slots — should not happen in normal usage
                # Copy weights from CPU to NPU
                layer.w13_weight.data.untyped_storage()[slot * self.w13_expert_size_bytes : (slot + 1) * self.w13_expert_size_bytes].copy_(
                    self.w13_weights_cpu[layer_idx][eid].untyped_storage()
                )
                layer.w2_weight.data.untyped_storage()[slot * self.w2_expert_size_bytes : (slot + 1) * self.w2_expert_size_bytes].copy_(
                    self.w2_weights_cpu[layer_idx][eid].untyped_storage()
                )
                # Copy scales/offsets from CPU to NPU
                for attr_name, buffers in self.scale_cpu_buffers.items():
                    if layer_idx >= len(buffers) or eid >= len(buffers[layer_idx]):
                        continue
                    dev_tensor = getattr(layer, attr_name, None)
                    if dev_tensor is None:
                        continue
                    dev_tensor.data[slot].copy_(buffers[layer_idx][eid])
                for attr_name, buffers in self.offset_cpu_buffers.items():
                    if layer_idx >= len(buffers) or eid >= len(buffers[layer_idx]):
                        continue
                    dev_tensor = getattr(layer, attr_name, None)
                    if dev_tensor is None:
                        continue
                    dev_tensor.data[slot].copy_(buffers[layer_idx][eid])
                # Refresh derived fp32 scale if present (W8A8_DYNAMIC)
                if hasattr(layer, 'w13_weight_scale_fp32'):
                    layer.w13_weight_scale_fp32[slot].copy_(
                        layer.w13_weight_scale.data[slot].to(torch.float32))
                # Update mapping
                if victim is None:
                    victim = slot_owner[slot]
                log2phy_np[victim] = -1             # evict old occupant
                log2phy_np[eid] = slot               # assign slot to new expert
                slot_owner[slot] = eid
                if slot in reusable_slots:
                    reusable_slots.remove(slot)
                n_copies += 1

            self.load_stream.synchronize()

    # ------------------------------------------------------------------ #
    #  Next-layer expert prefetch                                          #
    # ------------------------------------------------------------------ #

    def predict_next_layer_experts(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
    ) -> set[int] | None:
        """Predict which experts layer layer_idx+1 will need.

        Uses the current layer's hidden_states as an approximation of
        the next layer's input, multiplied by the next layer's gate weight
        to get predicted router logits.  A simplified softmax + topk is
        used instead of the full grouped_topk for speed; misses are
        handled by the reactive fallback in update_weights().

        Args:
            layer_idx: Current layer index.
            hidden_states: [num_tokens, hidden_dim] NPU tensor.

        Returns:
            Set of predicted expert IDs, or None if prediction is not
            possible (e.g. last layer, no gate weight).
        """
        next_idx = layer_idx + 1
        if next_idx >= len(self.moe_layers):
            return None  # last layer — nothing to prefetch

        if next_idx >= len(self._gate_weights_cpu):
            return None
        if self._gate_weights_cpu[next_idx] is None:
            return None

        # Move hidden_states to CPU for prediction (tiny in decode: 1-8 tokens).
        # This runs only on the background prefetch thread (_prefetch_worker),
        # a standalone thread after compute_event.synchronize(), so the .cpu()
        # is safe.  The graph-mode path uses _predict_next_layer_experts_cpu
        # over pre-staged pinned memory and never touches a live graph tensor.
        hs_cpu = hidden_states.float().cpu()
        return self._predict_next_layer_experts_cpu(layer_idx, hs_cpu)

    def _predict_next_layer_experts_cpu(
        self,
        layer_idx: int,
        hs_cpu: torch.Tensor,
    ) -> set[int] | None:
        """Pure-CPU prediction from an already-CPU hidden_states tensor.

        Same approximation as predict_next_layer_experts (simplified softmax +
        topk), but takes a CPU tensor directly — no .cpu() / blocking D2H.
        Used by the graph-mode host callback over the pre-staged pinned
        buffer (_prefetch_hs_h).
        """
        next_idx = layer_idx + 1
        if next_idx >= len(self.moe_layers):
            return None  # last layer — nothing to prefetch

        if next_idx >= len(self._gate_weights_cpu):
            return None
        gate_w = self._gate_weights_cpu[next_idx]
        if gate_w is None:
            return None

        # hs_cpu is already fp32 (staged that way); gate_w is fp32 on CPU.
        router_logits = F.linear(hs_cpu, gate_w)  # [num_tokens, n_experts]

        # Simplified routing: softmax + topk (approximation)
        probs = router_logits.softmax(dim=-1)
        _, topk_ids = probs.topk(self.topk, dim=-1)
        return set(topk_ids.flatten().tolist())

    def predict_next_layer_experts_npu(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        """Predict which experts layer layer_idx+1 will need, on NPU.

        Same approximation as predict_next_layer_experts (simplified
        softmax + topk), but runs entirely on the NPU so it can be
        captured in a CUDA/NPU graph.  The returned tensor lives on NPU.

        Args:
            layer_idx: Current layer index.
            hidden_states: [num_tokens, hidden_dim] NPU tensor.

        Returns:
            [num_tokens * topk] NPU int64 tensor of predicted expert IDs,
            or None if prediction is not possible.
        """
        next_idx = layer_idx + 1
        if next_idx >= len(self.moe_layers):
            return None  # last layer — nothing to prefetch

        if next_idx >= len(self._gate_weights_npu):
            return None
        gate_w = self._gate_weights_npu[next_idx]
        if gate_w is None:
            return None

        # On-device prediction: [num_tokens, hidden_dim] x [n_experts, hidden_dim]^T
        router_logits = F.linear(hidden_states.float(), gate_w)
        probs = router_logits.softmax(dim=-1)
        _, topk_ids = probs.topk(self.topk, dim=-1)  # [num_tokens, topk]
        predicted = topk_ids.flatten()               # [num_tokens * topk]
        return predicted

    # ------------------------------------------------------------------ #
    #  Threaded prefetch: background thread for expert weight loading     #
    # ------------------------------------------------------------------ #

    def _start_prefetch_thread(self):
        """Start the background prefetch thread (called once, lazily)."""
        self._npu_device = torch_npu.npu.current_stream().device
        self._prefetch_thread = threading.Thread(
            target=self._prefetch_worker,
            daemon=True,
            name="ExpertPrefetch",
        )
        self._prefetch_thread.start()
        self._prefetch_thread_ready.wait()

    def _prefetch_worker(self):
        """Background thread: processes prefetch requests from the queue.

        Each request is a tuple of (layer_idx, hidden_states, compute_event).
        The thread waits for the current layer's GMM kernel to complete
        (via compute_event), then predicts the next layer's needed experts
        and loads them from CPU to NPU.
        """
        torch.npu.set_device(self._npu_device)
        self._prefetch_thread_ready.set()

        while True:
            request = self._prefetch_queue.get()
            if request is None:
                break

            layer_idx, hidden_states, compute_event = request
            next_idx = layer_idx + 1

            # Wait for current layer's GMM kernel to complete before
            # reading hidden_states and touching next layer's weights.
            compute_event.synchronize()

            # Predict which experts the next layer needs
            predicted = self.predict_next_layer_experts(layer_idx,
                                                        hidden_states)
            if predicted is None or next_idx >= len(self.moe_layers):
                # No prediction possible — signal done immediately
                done = self._prefetch_layer_done.get(next_idx)
                if done is not None:
                    done.set()
                continue

            # Snapshot next layer's current log2phy to CPU.  This runs on the
            # standalone prefetch thread (after compute_event.synchronize()),
            # so the blocking .cpu() is safe here — unlike the graph host
            # callback, which stages this into a pinned buffer instead.
            next_layer = self.moe_layers[next_idx]
            log2phy_np = next_layer.log2phy.cpu().numpy().copy()

            # Execute the prefetch (H2D copies)
            completion_event = self._do_prefetch(next_idx, predicted,
                                                 log2phy_np)

            # Store NPU completion event and signal threading.Event
            with self._prefetch_state_lock:
                if completion_event is not None:
                    self._prefetch_layer_npu_event[next_idx] = completion_event
            done = self._prefetch_layer_done.get(next_idx)
            if done is not None:
                done.set()

    def _do_prefetch(
        self,
        next_idx: int,
        predicted_experts: set[int],
        log2phy_np,
        use_sleep: bool = True,
    ) -> torch_npu.npu.Event | None:
        """Load predicted experts for layer next_idx from CPU to NPU.

        Runs on the prefetch thread (non-graph) or from the graph host
        callback (graph mode), using self._prefetch_stream for the H2D
        copies.  Returns an NPU Event that signals when all copies are
        complete, or None if no copies were needed.

        Args:
            next_idx: Index of the layer to prefetch experts for.
            predicted_experts: Set of expert IDs predicted to be needed.
            log2phy_np: numpy view of next_idx's current log2phy (CPU,
                int32).  Provided by the caller — the graph path stages it
                into a pinned buffer before the callback so this function
                never does a blocking .cpu() on a live graph tensor.
            use_sleep: If True, insert a small delay between w13 and w2 copies
                to avoid burst traffic.  Should be False when called from a
                graph host callback to avoid blocking graph execution.
        """
        next_layer = self.moe_layers[next_idx]

        # Copy out so the async H2D write-back (next_layer.log2phy.copy_)
        # reads an independent buffer.  In graph mode log2phy_np is the shared
        # _prefetch_log2phy_np view; without this copy the async H2D would
        # race the next layer's staging D2H that overwrites it.  Tiny:
        # num_total_experts * 4 bytes.
        log2phy_np = log2phy_np.copy()

        # Determine which predicted experts are not already on device
        slot_owner: dict[int, int] = {}
        for eid, slot in enumerate(log2phy_np):
            if slot >= 0:
                slot_owner[slot] = eid
        on_device = set(slot_owner.values())
        need_to_load = predicted_experts - on_device
        already_there = on_device & predicted_experts

        if not need_to_load:
            return None

        # Protected set: experts we must not evict
        protected = set(predicted_experts)
        reusable_slots = [s for s, e in slot_owner.items()
                          if e not in protected]
        if not reusable_slots:
            return None  # all resident experts are protected — skip
        
        if self._debug:
                logger.info("[PREFETCH-W] l=%d prefetch_expert_hit=%s prefetch_expert_miss=%s hit_rate=%.2f",
                            next_idx, sorted(already_there),
                            sorted(need_to_load), len(already_there) / 6)
                if need_to_load and len(need_to_load) > len(reusable_slots):
                    logger.info("[PREFETCH-W] l=%d SHORTFALL: need %d load but only %d slots, "
                                "to_load=%s",
                                next_idx, len(need_to_load), len(reusable_slots),
                                sorted(need_to_load)[:20])

        with torch_npu.npu.stream(self._prefetch_stream):
            n_copies = 0
            for eid in need_to_load:
                # Use cache_policy.choose_victim() for eviction, same as
                # _update_weights.  Pass next_idx so it queries L+1's
                # hotness statistics.
                if self.cache_policy is not None:
                    victim = self.cache_policy.choose_victim(
                        next_idx,
                        slot_owner,
                        protected=protected,
                        loading=need_to_load,
                    )
                    slot = (int(log2phy_np[victim])
                            if victim is not None else -1)
                elif reusable_slots:
                    slot = reusable_slots.pop()
                    victim = slot_owner[slot]
                else:
                    slot = -1
                    victim = None

                if slot < 0:
                    if self._debug:
                        logger.info(
                            "[PREFETCH] l=%d NO SLOTS: %d experts could not be prefetched, missed= %s", 
                        next_idx , len(need_to_load)-n_copies,sorted(list(need_to_load))[n_copies:][:20]
                        )
                    break

                # CPU → NPU async copy w13 weights
                next_layer.w13_weight.data.untyped_storage()[
                    slot * self.w13_expert_size_bytes
                    : (slot + 1) * self.w13_expert_size_bytes
                ].copy_(
                    self.w13_weights_cpu[next_idx][eid].untyped_storage())

                if use_sleep:
                    time.sleep(0.00025)  # 0.25ms delay, avoid burst in thread mode

                # CPU → NPU async copy w2 weights
                next_layer.w2_weight.data.untyped_storage()[
                    slot * self.w2_expert_size_bytes
                    : (slot + 1) * self.w2_expert_size_bytes
                ].copy_(
                    self.w2_weights_cpu[next_idx][eid].untyped_storage())

                # Copy scales/offsets from CPU to NPU (W8A8)
                for attr_name, buffers in self.scale_cpu_buffers.items():
                    if next_idx >= len(buffers):
                        continue
                    if eid >= len(buffers[next_idx]):
                        continue
                    dev_tensor = getattr(next_layer, attr_name, None)
                    if dev_tensor is not None:
                        dev_tensor.data[slot].copy_(
                            buffers[next_idx][eid])
                for attr_name, buffers in self.offset_cpu_buffers.items():
                    if next_idx >= len(buffers):
                        continue
                    if eid >= len(buffers[next_idx]):
                        continue
                    dev_tensor = getattr(next_layer, attr_name, None)
                    if dev_tensor is not None:
                        dev_tensor.data[slot].copy_(
                            buffers[next_idx][eid])

                # Refresh fp32 scale (W8A8_DYNAMIC)
                if hasattr(next_layer, 'w13_weight_scale_fp32'):
                    next_layer.w13_weight_scale_fp32[slot].copy_(
                        next_layer.w13_weight_scale.data[slot].to(
                            torch.float32))

                # Update log2phy mapping
                if victim is None:
                    victim = slot_owner[slot]
                log2phy_np[victim] = -1
                log2phy_np[eid] = slot
                slot_owner[slot] = eid
                if slot in reusable_slots:
                    reusable_slots.remove(slot)
                n_copies += 1

            # Write modified log2phy back to next layer's NPU tensor
            next_layer.log2phy.copy_(
                torch.from_numpy(log2phy_np).to(
                    device=next_layer.log2phy.device))

            # Record prefetch completion event
            completion_event = torch_npu.npu.Event()
            self._prefetch_stream.record_event(completion_event)
            return completion_event

    def trigger_next_layer_prefetch(self, layer,
                                    hidden_states: torch.Tensor):
        """在 GMM kernel 提交后触发下一层专家预加载。

        必须在 fused_experts() 之后调用，使 compute_event 捕获
        GMM kernel 的 NPU 工作，实现预加载与计算的真正并行。

        图模式下通过 _launch_host_func 提交到 host callback 执行；
        非图模式下提交到后台线程，主线程立即返回，不被 aclrtMemcpy 阻塞。

        Args:
            layer: 当前 MoE 层的 AscendFusedMoE 实例。
            hidden_states: [num_tokens, hidden_dim] NPU tensor。
        """
        if not self.offload_config.expert_prefetch_enabled:
            return

        try:
            layer_idx = self.moe_layers.index(layer)
        except ValueError:
            return

        next_idx = layer_idx + 1
        if next_idx >= len(self.moe_layers):
            return

        # Record compute event on main thread's compute stream
        # (captures GMM kernel progress so prefetch waits for it)
        compute_event = torch_npu.npu.Event()
        torch_npu.npu.current_stream().record_event(compute_event)

        if _EXTRA_CTX.capturing:
            # Graph mode: launch a host callback that runs prediction and
            # H2D copies on the prefetch stream.  The callback itself is
            # synchronous, but the copies it enqueues are asynchronous.
            #
            # Stage the data the callback needs into pinned CPU *before*
            # launching it, with non_blocking D2H on the compute stream.
            # _launch_host_func guarantees the callback runs only after the
            # stream ops queued before it complete, so the callback reads
            # already-ready memory and never does a blocking .cpu() on a live
            # graph tensor (which would deadlock the stream's host thread).
            # Mirrors update_weights' staging pattern exactly.
            next_layer = self.moe_layers[next_idx]
            num_tokens = hidden_states.size(0)
            hs_h = self._prefetch_hs_h[:num_tokens]
            # Cast to fp32 on-device first, then D2H into the fp32 pinned
            # buffer — mirrors update_weights (topk_weights.to(float32)),
            # keeping the captured op identical to the proven path rather
            # than relying on a cross-dtype copy_.
            hs_h.copy_(hidden_states.to(torch.float32),
                       non_blocking=_EXTRA_CTX.capturing)
            self._prefetch_log2phy_h.copy_(next_layer.log2phy,
                                           non_blocking=_EXTRA_CTX.capturing)
            args = (layer_idx, hs_h, compute_event)
            torch_npu.npu._launch_host_func(
                torch_npu.npu.current_stream(),
                self._do_prefetch_host_callback,
                args,
            )
        else:
            # Non-graph mode: use the background prefetch thread.
            if self._prefetch_thread is None:
                self._start_prefetch_thread()

            # Create per-layer threading.Event for completion signaling
            self._prefetch_layer_done[next_idx] = threading.Event()

            # Submit to background thread — returns immediately!
            self._prefetch_queue.put((layer_idx, hidden_states, compute_event))

    def _do_prefetch_host_callback(self, args):
        """Host callback for graph-mode prefetch.

        Runs on the CPU during graph replay (the compute stream's host
        thread).  It predicts the next layer's experts, waits for the
        current layer's GMM on the prefetch stream, and enqueues async H2D
        copies.  The returned completion event is stored for
        update_weights() to wait on.

        Must NOT block or do any synchronous D2H: the data it needs
        (hidden_states and next layer's log2phy) is pre-staged into pinned
        buffers by trigger_next_layer_prefetch on the compute stream before
        this callback is launched, so reads here are always ready.
        """
        layer_idx, hs_cpu, compute_event = args
        next_idx = layer_idx + 1
        if next_idx >= len(self.moe_layers):
            return

        # Pure-CPU prediction over the already-staged hidden_states — no .cpu().
        predicted = self._predict_next_layer_experts_cpu(layer_idx, hs_cpu)
        if predicted is None:
            return

        # Enqueue prefetch work on the prefetch stream after the GMM event.
        with torch_npu.npu.stream(self._prefetch_stream):
            self._prefetch_stream.wait_event(compute_event)
            completion_event = self._do_prefetch(
                next_idx, predicted, self._prefetch_log2phy_np,
                use_sleep=False)

        # Store completion event for update_weights() to wait on.
        if completion_event is not None:
            with self._prefetch_state_lock:
                self._prefetch_layer_npu_event[next_idx] = completion_event

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
        self.last_hit_experts[layer_idx] = sorted(hit_experts)
        self.last_miss_experts[layer_idx] = sorted(miss_experts)

        interval = self.offload_config.cache_stats_log_interval
        if interval == 0 or self.cache_calls[layer_idx] % interval != 0:
            return

        requests = self.cache_requests[layer_idx]
        hit_rate = self.cache_hits[layer_idx] / requests if requests else 0.0
        policy_step = -1
        if self.cache_policy is not None:
            policy_step = self.cache_policy.layer_step(layer_idx)
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
