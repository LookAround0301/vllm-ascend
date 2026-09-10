"""Trained expert predictors for the MoE expert-offload prefetch path.

Prefetch METHODS are selected by `expert_predictor` in expert_offload_config:

    fate            the engine's built-in hash/gate next-layer predictor,
                    driven from the four MoE apply() sites
    mode2_har       trained head, predicts layer L from L's PRE-attention
                    residual (LAYER_DELTA=0)
    mode2_prevhfr   trained head, predicts layer L+1 from L's POST-attention
                    residual (LAYER_DELTA=1)

fate always keeps the targets a trained head does not cover: the hash-routed
layers (tid2eid[token_id] is exact and free), plus — for a layer-shifted atom —
the first covered MoE layer, whose predecessor was not in the training dump.

Head, normalisation and selection are transcribed from
moe_predictor_study_mode2_pa.py and must stay literal:
    LowRankProbe.forward                -> _LowRankHead.forward
    train_probe's x_n = (x - mu) / sd   -> _TrainedPredictor.predict
    eval_full's _biased(sc) = softplus(sc).sqrt() + bias
"""

import torch
import torch.nn.functional as F
import torch_npu
from vllm.logger import logger

from vllm_ascend.ascend_forward_context import _EXTRA_CTX, MoECommType

_REGISTRY: dict[str, type] = {}

# The heuristic family. Not a registry entry: no checkpoint, and it is driven
# from the apply() sites rather than the decoder layer.
HEURISTIC = "fate"

# Set by maybe_create_driver(); read by the launch custom op and the decoder
# layer's build-time site resolution.
_DRIVER = None


def register_predictor(name: str):
    """Class decorator registering a predictor under its config string."""

    def _wrap(cls):
        if name in _REGISTRY or name == HEURISTIC:
            raise ValueError(f"duplicate expert predictor name {name!r}")
        _REGISTRY[name] = cls
        cls.predictor_name = name
        return cls

    return _wrap


def valid_predictor_names() -> list[str]:
    """Single source of truth for config validation AND its error message."""
    return [HEURISTIC] + sorted(_REGISTRY)


def create_predictor(name: str, ckpt_path):
    """Resolve the configured name to one predictor object, or None for fate.

    Resolved once at config time: the forward path must never branch on the
    name, because graph capture bakes whichever branch ran at capture.
    """
    if name == HEURISTIC:
        return None
    cls = _REGISTRY.get(name)
    if cls is None:
        raise ValueError(
            f"unknown expert predictor {name!r}; valid names: "
            f"{valid_predictor_names()}")
    return cls(ckpt_path)


# ---------------------------------------------------------------------- #
#  Heads                                                                   #
# ---------------------------------------------------------------------- #
class _LowRankHead:
    """Inference-only mirror of the study's LowRankProbe (arch="lowrank").

    Training batched all L layers via einsum("nlh,lhr->nlr", x, Wd); at decode
    we evaluate one layer, so that becomes x @ Wd[h]. Dropout is omitted (the
    study gates it on self.training and we are always in eval).

    Params are kept in bf16, the dtype the study saves them in, so an fp32
    upcast would add no information — only matmul input precision. At L=40,
    in_dim=16384, r=320 that is ~460 MiB resident instead of ~920 MiB, out of
    the same HBM budget the expert cache lives in.
    """

    PARAM_KEYS = ("Wd", "W1", "b1", "Wo", "bo")

    def __init__(self, params, device):
        missing = [k for k in self.PARAM_KEYS if k not in params]
        if missing:
            raise ValueError(
                f"lowrank checkpoint is missing params {missing}; "
                f"found {sorted(params)}")
        for key in self.PARAM_KEYS:
            setattr(self, key, params[key].to(
                device=device, dtype=torch.bfloat16).contiguous())
        self.width = int(self.b1.shape[-1])
        self.rank = int(self.Wd.shape[-1])

    def forward(self, head_idx: int, x_n: torch.Tensor) -> torch.Tensor:
        """[n, in_dim] fp32 -> [n, E] fp32 predicted router LOGITS.

        Logits, not scores: the scoring activation and the correction bias are
        applied by predict(), matching the study's eval_full.
        """
        h = x_n.to(torch.bfloat16) @ self.Wd[head_idx]
        h = h @ self.W1[head_idx] + self.b1[head_idx]
        # Parameter-free LayerNorm, exactly as trained. Done in fp32 explicitly
        # so the numerics do not depend on a torch_npu implementation detail;
        # n <= 6 here so it is free.
        h = F.layer_norm(h.to(torch.float32), (self.width,)).to(torch.bfloat16)
        h = F.gelu(h)
        h = h @ self.Wo[head_idx] + self.bo[head_idx]
        return h.to(torch.float32)


_ARCH_HEADS = {"lowrank": _LowRankHead}


# ---------------------------------------------------------------------- #
#  Predictors                                                              #
# ---------------------------------------------------------------------- #
class _TrainedPredictor:
    """Shared machinery. A concrete predictor declares only its composition."""

    # Hidden-size (H) blocks, expert-dim (E) blocks and residual blocks (each
    # res_streams * H) in the input. Solves the checkpoint's in_dim against the
    # model geometry — the only guard against a checkpoint from another model.
    IN_FEATURES = 0
    E_FEATURES = 0
    RES_FEATURES = 0

    # Layers of lead: the atom captured at source layer S predicts target
    # S + LAYER_DELTA. Also the number of leading covered layers that must be
    # dropped, because a layer-shifted atom has no predecessor at the first
    # dumped layer (the study zero-fills it during training, so the first head
    # never saw a real input).
    LAYER_DELTA = 0

    def __init__(self, ckpt_path: str):
        self.ckpt_path = ckpt_path
        self.head = None
        self.mu = None
        self.sd = None
        self.in_dim = 0
        self.num_heads = 0
        self.layer_offset = 0
        self.top_k = 0
        self.num_hash_layers = 0
        self.arch = ""

    def finalize(self, device, num_moe_layers: int, num_hash_layers: int,
                 hidden_size: int, hc_mult, num_total_experts: int,
                 model_top_k: int) -> None:
        """Load the checkpoint onto `device`. Called once, at offload finalize.

        Deferred to finalize because the device and the model geometry are only
        known once the model exists, and the head must be resident before graph
        capture.
        """
        state = torch.load(self.ckpt_path, map_location="cpu",
                           weights_only=False)
        meta = state["meta"]
        self.arch = str(meta["arch"])
        head_cls = _ARCH_HEADS.get(self.arch)
        if head_cls is None:
            raise ValueError(
                f"{self.ckpt_path}: unsupported predictor arch {self.arch!r}; "
                f"supported: {sorted(_ARCH_HEADS)}")
        self.in_dim = int(meta["in_dim"])
        self.top_k = int(meta["top_k"])
        self.num_heads = int(meta["L"])
        ckpt_experts = int(meta["E"])
        self.num_hash_layers = num_hash_layers

        self._solve_dimensions(hidden_size, hc_mult, num_total_experts)

        if ckpt_experts != num_total_experts:
            # Out-of-range ids are dropped in predict(). Ids that come out huge
            # or negative at runtime mean a pinned buffer was read before the
            # predict finished — a stream-ordering bug, not this.
            logger.warning(
                "[PREFETCH-AI] checkpoint expert space E=%d != model "
                "num_total_experts=%d; predicted ids outside [0, %d) are "
                "dropped", ckpt_experts, num_total_experts, num_total_experts)
        if self.top_k != model_top_k:
            logger.warning(
                "[PREFETCH-AI] checkpoint top_k=%d != model top_k=%d; the "
                "prediction width differs from the router's", self.top_k,
                model_top_k)
        if not meta.get("bias_aware_select", True):
            logger.warning(
                "[PREFETCH-AI] checkpoint was selected on the NO-BIAS cov@k "
                "but inference ranks with e_score_correction_bias, as the "
                "model's router does")

        # The checkpoint covers the LAST `L` MoE layers: the study's dump drops
        # the leading table-routed layers. A mismatch here maps every head onto
        # the wrong layer, with no error anywhere.
        self.layer_offset = num_moe_layers - self.num_heads
        if self.layer_offset < 0:
            raise ValueError(
                f"{self.ckpt_path}: checkpoint covers L={self.num_heads} "
                f"layers but the model has only {num_moe_layers} MoE layers")
        if self.layer_offset != num_hash_layers:
            logger.warning(
                "[PREFETCH-AI] layer_offset=%d != num_hash_layers=%d; prefetch "
                "coverage may be shifted by %d layers", self.layer_offset,
                num_hash_layers, self.layer_offset - num_hash_layers)

        self.head = head_cls(state["params"], device)
        # mu/sd are [1, L, in_dim] when the study standardized inputs and
        # [1, 1, 1] when it did not; expand covers both. Kept fp32.
        self.mu = self._prepare_stat(state["mu"], device)
        self.sd = self._prepare_stat(state["sd"], device)

        logger.info(
            "[PREFETCH-AI] predictor=%s arch=%s L=%d layer_offset=%d "
            "layer_delta=%d in_dim=%d width=%d rank=%d E=%d top_k=%d ckpt=%s",
            self.predictor_name, self.arch, self.num_heads, self.layer_offset,
            self.LAYER_DELTA, self.in_dim, self.head.width, self.head.rank,
            ckpt_experts, self.top_k, self.ckpt_path)

    def _prepare_stat(self, t: torch.Tensor, device) -> torch.Tensor:
        t = t.to(device=device, dtype=torch.float32)
        if t.dim() == 3:
            t = t[0]
        return t.expand(self.num_heads, self.in_dim).contiguous()

    def _solve_dimensions(self, hidden_size: int, hc_mult,
                          num_total_experts: int) -> None:
        expect = (self.IN_FEATURES * hidden_size
                  + self.E_FEATURES * num_total_experts)
        if self.RES_FEATURES:
            per = self.RES_FEATURES * hidden_size
            rem = self.in_dim - expect
            if rem <= 0 or rem % per != 0:
                raise ValueError(
                    f"{self.ckpt_path}: in_dim={self.in_dim} is not "
                    f"{self.IN_FEATURES}*H + {self.E_FEATURES}*E + "
                    f"k*{self.RES_FEATURES}*H for H={hidden_size}, "
                    f"E={num_total_experts}")
            res_streams = rem // per
            if hc_mult is not None and res_streams != hc_mult:
                raise ValueError(
                    f"{self.ckpt_path}: in_dim={self.in_dim} implies "
                    f"{res_streams} hyper-connection streams but the model has "
                    f"hc_mult={hc_mult} (H={hidden_size})")
        elif self.in_dim != expect:
            raise ValueError(
                f"{self.ckpt_path}: in_dim={self.in_dim} != "
                f"{self.IN_FEATURES}*H + {self.E_FEATURES}*E = {expect} "
                f"for H={hidden_size}, E={num_total_experts}")

    def head_index(self, target_idx: int):
        """TARGET MoE-layer index -> head index, or None when not covered.

        Two exclusions, both leaving the target to fate:
          * hash-routed layers — fate predicts them exactly via tid2eid;
          * the first LAYER_DELTA covered layers — a layer-shifted atom has no
            predecessor there, and the study zero-filled it during training, so
            that head never saw a real input.
        """
        if target_idx < self.num_hash_layers + self.LAYER_DELTA:
            return None
        h = target_idx - self.layer_offset
        return h if 0 <= h < self.num_heads else None

    def predict(self, head_idx: int, feat: torch.Tensor,
                e_score_correction_bias, num_total_experts: int = 0):
        """[n, in_dim] -> (ids [n, top_k] int32, scores [n, top_k] fp32).

        The head emits router LOGITS. Ranking then reproduces the study's
        deployment space (eval_full._biased): the sqrt-softplus scoring
        activation, THEN the per-expert correction bias. The bias is not
        optional — topk_method is noaux_tc, so the bias is what re-orders the
        top-k — and it must be the TARGET layer's bias, which is why the driver
        keys everything on the target index.

        The returned scores are the key the single-card prefetch pass uses to
        pick which missing experts to transfer, so they must live in the same
        space the ids were selected in.
        """
        x_n = (feat.to(torch.float32) - self.mu[head_idx]) / self.sd[head_idx]
        logits = self.head.forward(head_idx, x_n)
        scores = F.softplus(logits).sqrt()
        if e_score_correction_bias is not None:
            scores = scores + e_score_correction_bias.to(
                torch.float32).unsqueeze(0)
        if 0 < num_total_experts < scores.shape[-1]:
            # A head wider than the model's expert space would index past the
            # per-layer CPU buffers inside the copy loop. Truncate on device so
            # the output shape stays static for capture.
            scores = scores[:, :num_total_experts]
        # sorted=False matches the model's own selection call.
        values, ids = scores.topk(self.top_k, dim=-1, sorted=False)
        return ids.to(torch.int32), values.to(torch.float32)


@register_predictor("mode2_har")
class Mode2HarPredictor(_TrainedPredictor):
    """Predict layer L's OWN experts from layer L's `har` residual.

    `har` is the study's hc_attn_residual atom: the raw decoder-layer input, all
    hc_mult hyper-connection streams, before hc_pre collapses it and
    input_layernorm normalizes it. ATOM_SPEC["har"] = ("har", False, False,
    False) — no layer shift, no token shift, no zero-fill, no edge-pad.

    Captured at the FIRST residual clone of layer L.
    """

    IN_FEATURES = 0
    E_FEATURES = 0
    RES_FEATURES = 1
    LAYER_DELTA = 0


@register_predictor("mode2_prevhfr")
class Mode2PrevHfrPredictor(_TrainedPredictor):
    """Predict layer L+1's experts from layer L's `hfr` residual.

    `hfr` is the study's hc_ffn_residual atom: the post-attention block input,
    before hc_pre collapses it and post_attention_layernorm normalizes it.
    ATOM_SPEC["prevhfr"] = ("hfr", True, False, False) — layer-shifted, so
    prevhfr[L] == hfr[L-1] and _ATOM_LEAD is 1.

    Captured at the SECOND residual clone of layer L, which buys a full extra
    layer of lead over har: the transfer window becomes GMM(L) plus
    attention(L+1) instead of attention alone. That window now contains layer
    L's own on-demand load, so the dispatch is ordered behind it — see
    ExpertPredictorDriver.finish and the manager's _finish_next_layer_predict.
    """

    IN_FEATURES = 0
    E_FEATURES = 0
    RES_FEATURES = 1
    LAYER_DELTA = 1


# ---------------------------------------------------------------------- #
#  Driver                                                                  #
# ---------------------------------------------------------------------- #
class ExpertPredictorDriver:
    """Owns the capture site, the buffers and the launch/finish split.

    Everything is keyed on the TARGET MoE-layer index, so the correction bias,
    the log2phy snapshot and the join event all belong to the layer being
    predicted. The decoder layer resolves target = source + layer_delta once,
    at build time.

    launch() stages the atom and enqueues the head on the prefetch stream;
    finish() consumes it from ExpertOffloadManager.update_weights[_multi_card].
    Both run on the forward thread, and both are reached only from code OUTSIDE
    torch.compile's traced region — launch via the custom op at the bottom of
    this file, finish via the manager's apply()-side path.
    """

    def __init__(self, manager, predictor):
        self.mgr = manager
        self.predictor = predictor
        self.layer_delta = predictor.LAYER_DELTA
        self.ready = False
        # [2, mt, in_dim] bf16 NPU. TWO SLOTS, indexed target & 1. One is
        # provably safe only for layer_delta=0, where the compute stream must
        # pass wait(load_done[L]) — recorded after the head read the buffer —
        # before it can overwrite at layer L+1. With layer_delta=1 the chain
        # launched at layer L joins inside mlp(L+1), which is AFTER layer L+1's
        # own staging copy, so nothing orders the overwrite against the read.
        # Two slots restore the guarantee: the chain for target T is joined
        # inside mlp(T) and the slot is next reused at target T+2, whose staging
        # copy is after that join.
        self._feat = None
        self._ids_h = None           # [mt, k] int32 pinned  — result landing
        self._scores_h = None        # [mt, k] fp32 pinned
        self._log2phy_h = None       # [L, E] int32 pinned, per-layer rows
        self._log2phy_np = None
        self._mask_h = None          # [mt] int32 pinned     — multi card only
        # target_idx -> (n_tok, k, has_mask). Single-threaded by construction
        # and deliberately unlocked: both writers are the forward thread, and
        # the host callbacks read the pinned buffers, never this dict. A lock
        # here is also un-traceable by Dynamo if a caller ever moves back inside
        # the compiled region.
        self._pending = {}
        self._dp_size = 1
        self._stale_warned = False

    # -- build time ------------------------------------------------------ #

    def register_site(self, gate):
        """Resolve the SOURCE MoE index of the layer whose router is `gate`.

        Called once per decoder layer at build time, right after that layer's
        MoE registered. Identity on the gate object rather than the decoder
        layer's layer_idx, because MTP and DSpark reuse DeepseekV2DecoderLayer
        with prefix="mtp.N" and would alias target layer N. Draft MoE layers
        never register with the manager, so their gate matches nothing.

        The caller adds layer_delta to get the target. Coverage of that target
        is NOT checked here — the checkpoint has not loaded yet — so launch()
        rechecks head_index() per call.
        """
        if gate is None:
            return None
        for idx in range(len(self.mgr.moe_layers) - 1, -1, -1):
            if getattr(self.mgr.moe_layers[idx], "gate", None) is gate:
                return idx
        return None

    # -- finalize -------------------------------------------------------- #

    def finalize(self, model) -> None:
        """Load the head and allocate buffers. Called from _finalize_offload."""
        mgr = self.mgr
        config = model.config
        num_moe_layers = len(mgr.moe_layers)
        scoring_func = getattr(config, "scoring_func", "softmax")
        if scoring_func != "sqrtsoftplus":
            # The head's selection is evaluated in the sqrt-softplus score
            # space; another scoring function would rank the same logits in a
            # different space than the checkpoint was selected on.
            raise ValueError(
                f"expert_predictor expects scoring_func='sqrtsoftplus'; the "
                f"model declares {scoring_func!r}")
        if num_moe_layers != int(config.num_hidden_layers):
            raise ValueError(
                f"expert_predictor needs every MoE layer registered: "
                f"{num_moe_layers} registered vs num_hidden_layers="
                f"{config.num_hidden_layers}")
        if not hasattr(torch.ops.vllm_ascend, "expert_predictor_launch"):
            # The decoder layer reaches launch() only through that op; without
            # it the capture site silently never fires.
            raise RuntimeError(
                "torch.ops.vllm_ascend.expert_predictor_launch is not "
                "registered — expert_predictor.py did not import cleanly")

        # AllGather concatenates every DP rank's rows before the MoE, so the
        # count compared against offload_threshold is dp_size x the rows the
        # decoder layer sees. Multi-card gates on the comm type instead.
        from vllm.distributed.parallel_state import get_dp_group
        self._dp_size = get_dp_group().world_size

        device = next(model.parameters()).device
        self.predictor.finalize(
            device=device,
            num_moe_layers=num_moe_layers,
            num_hash_layers=int(getattr(config, "num_hash_layers", 0)),
            hidden_size=int(config.hidden_size),
            hc_mult=getattr(config, "hc_mult", None),
            num_total_experts=mgr.num_total_experts,
            model_top_k=mgr.topk,
        )

        # Sized for the decode/prefill threshold but never below
        # expert_prefetch_tokens, so raising that knob cannot silently clamp
        # the prediction width.
        mt = max(mgr.offload_threshold, mgr.prefetch_tokens)
        k_land = max(self.predictor.top_k, mgr.topk)
        # bf16 matches the layer input, so the staging copy is not a cast;
        # predict() upcasts to fp32 for the normalisation. Two slots — see
        # __init__.
        self._feat = torch.zeros([2, mt, self.predictor.in_dim],
                                 dtype=torch.bfloat16, device=device)
        # Deliberately NOT the manager's shared topk_ids_h / topk_weights_h:
        # those are written by the compute stream while these are written by
        # the prefetch stream. One slot suffices here — same-stream FIFO puts
        # each chain's host callback ahead of the next chain's D2H.
        self._ids_h = torch.zeros([mt, k_land], dtype=torch.int32,
                                  device="cpu", pin_memory=True)
        self._scores_h = torch.zeros([mt, k_land], dtype=torch.float32,
                                     device="cpu", pin_memory=True)
        # Per-layer log2phy rows, same reason. A row of a contiguous pinned 2-D
        # tensor is itself pinned and contiguous.
        self._log2phy_h = torch.zeros([num_moe_layers, mgr.num_total_experts],
                                      dtype=torch.int32, device="cpu",
                                      pin_memory=True)
        self._log2phy_np = self._log2phy_h.numpy()
        self._mask_h = torch.zeros(mt, dtype=torch.int32, device="cpu",
                                   pin_memory=True)
        self._pending = {}
        self.ready = True

        covered = [i for i in range(num_moe_layers)
                   if self.predictor.head_index(i) is not None]
        logger.info(
            "[PREFETCH-AI] armed: targets=%d layers=%s..%s layer_delta=%d "
            "multi_card=%s dp_size=%d rows_buffer=%d prefetch_tokens=%d "
            "prefetch_topk=%d (uncovered targets stay on the fate driver)",
            len(covered), covered[0] if covered else None,
            covered[-1] if covered else None, self.layer_delta,
            mgr.enable_multi_card, self._dp_size, mt, mgr.prefetch_tokens,
            mgr.prefetch_topk)

    # -- runtime --------------------------------------------------------- #

    def covers(self, target_idx: int) -> bool:
        """Does this driver own the prefetch for `target_idx`?

        The manager's fate driver takes every target this returns False for, so
        hash layers and a layer-shifted head's uncovered leading layer keep the
        exact tid2eid / gate prediction instead of losing prefetch entirely.
        """
        return (self.ready
                and self.predictor.head_index(target_idx) is not None)

    def has_pending(self, target_idx: int) -> bool:
        return target_idx in self._pending

    def _in_decode_regime(self) -> bool:
        """Would this step page rather than take the prefill pool?

        Multi card decides on the comm TYPE, as apply() does; MC2 admission
        already implies num_tokens <= offload_threshold. Single card decides on
        MoE ROWS, which AllGather inflates by dp_size.
        """
        if self.mgr.enable_multi_card:
            return _EXTRA_CTX.moe_comm_type == MoECommType.MC2
        rows = _EXTRA_CTX.max_tokens_across_dp
        if rows is None:
            return False
        return rows * self._dp_size <= self.mgr.offload_threshold

    def launch(self, target_idx: int, layer_input: torch.Tensor) -> None:
        """Stage the atom and start the predict on the prefetch stream.

        `target_idx` is the layer being PREDICTED (source + layer_delta).
        `layer_input` is the source layer's residual clone, [n, hc_mult, hidden]
        — the first clone for har, the second for prevhfr. Reached only through
        the launch custom op, so the guards below are evaluated per call against
        the live forward context.
        """
        if not self.ready:
            return
        mgr = self.mgr
        if mgr._skip_prefill:
            return
        # FlashComm-v1 / sequence parallelism partitions tokens across TP ranks
        # between the decoder layer and the MoE, so these rows would describe a
        # different token set than the layer routes.
        if _EXTRA_CTX.flash_comm_v1_enabled:
            return
        head_idx = self.predictor.head_index(target_idx)
        if head_idx is None:
            return
        if not self._in_decode_regime():
            return
        rows = layer_input.shape[0]
        # Same expression predict_next_layer_experts_npu uses, so every method
        # samples the same number of rows. The head is batched over them.
        n_tok = min(mgr.prefetch_tokens, rows)
        if n_tok < 1:
            return
        if self._pending:
            # launch and finish are paired within one layer's forward, so a
            # leftover entry means a finish call site is missing.
            if not self._stale_warned:
                self._stale_warned = True
                logger.warning(
                    "[PREFETCH-AI] launch found an unfinished predict for "
                    "target(s) %s — a finish call site is missing",
                    sorted(self._pending))
            # clear on BOTH paths.
            if not _EXTRA_CTX.capturing:
                mgr._prefetch_stream.synchronize()
            self._pending.clear()
        layer = mgr.moe_layers[target_idx]
        try:
            # Stage on the COMPUTE stream: static shape for capture, and it
            # decouples the head from a compute-stream allocation's lifetime.
            stage = self._feat[target_idx & 1, :n_tok]
            stage.copy_(layer_input[:n_tok].flatten(1))
            feat_ready = torch_npu.npu.Event()
            torch_npu.npu.current_stream().record_event(feat_ready)
            # TARGET layer's bias: noaux_tc ranks with it, and it is per-layer.
            bias = getattr(getattr(layer, "gate", None),
                           "e_score_correction_bias", None)
            mc2_mask = _EXTRA_CTX.mc2_mask if mgr.enable_multi_card else None
            with torch_npu.npu.stream(mgr._prefetch_stream):
                mgr._prefetch_stream.wait_event(feat_ready)
                ids, scores = self.predictor.predict(
                    head_idx, stage, bias, mgr.num_total_experts)
                k = ids.shape[1]
                # non_blocking into PINNED memory on the prefetch stream. In
                # eager, finish() synchronizes this stream before any host read;
                # under capture these are nodes the host-func node on the same
                # stream is ordered behind.
                self._ids_h[:n_tok, :k].copy_(ids, non_blocking=True)
                self._scores_h[:n_tok, :k].copy_(scores, non_blocking=True)
                # Residency snapshot for the TARGET layer, whose log2phy is not
                # written again until its own update_weights. Must be staged,
                # not left stale: _prefetch_host_cb restores it on failure and
                # the write-back replays regardless, so a stale row would
                # publish residency for experts that were never loaded.
                self._log2phy_h[target_idx].copy_(layer.log2phy,
                                                  non_blocking=True)
                has_mask = mc2_mask is not None
                if has_mask:
                    # int32, not bool: Ascend has no async bool D2H and a sync
                    # here would break capture. mask is [padded_num_tokens] with
                    # mask[:num_actual_tokens]=True and our rows are the
                    # pre-prepare() rows, so mask[:n_tok] aligns with them.
                    self._mask_h[:n_tok].copy_(
                        mc2_mask[:n_tok].to(torch.int32), non_blocking=True)
            self._pending[target_idx] = (n_tok, k, has_mask)
        except Exception:
            # No host Event is held on this path, so a failure cannot hang the
            # forward: no load_done event is recorded and update_weights simply
            # finds nothing to wait on and pages on demand.
            self._pending.pop(target_idx, None)
            mgr._note_cb_failure("expert predictor launch")

    def finish(self, target_idx: int, ready_event=None) -> None:
        """Consume the predict and enter the shared prefetch execution path.

        Called from ExpertOffloadManager via _finish_pending_predict (delta=0,
        top of update_weights, ready_event=None) or _finish_next_layer_predict
        (delta=1, end of update_weights, ready_event recorded on the compute
        stream after that layer's on-demand load).

        `ready_event` is the ordering edge that keeps a layer-shifted method
        from contending with the on-demand load it now overlaps: the prefetch
        stream waits on it before the planning callback, so the callback cannot
        occupy the report thread or the shared load_stream until the on-demand
        transfer has drained.
        """
        if not self.ready:
            return
        pending = self._pending.pop(target_idx, None)
        if pending is None:
            return
        n_tok, k, has_mask = pending
        mgr = self.mgr
        try:
            if not _EXTRA_CTX.capturing:
                # The host callback below reads the pinned buffers on the host,
                # and their D2H was issued non_blocking on this stream.
                mgr._prefetch_stream.synchronize()
            layer = mgr.moe_layers[target_idx]
            # Exactly the tuple _stage_predicted_topk produces, so
            # _dispatch_prefetch -> _build_prefetch_call -> _update_weights
            # [_multi_card] is reached unchanged. Multi card consumes the pinned
            # log2phy tensor plus the mask and ignores the scores and the numpy
            # view; single card is the reverse.
            staged = (self._ids_h[:n_tok, :k],
                      self._scores_h[:n_tok, :k],
                      self._log2phy_h[target_idx],
                      self._log2phy_np[target_idx],
                      layer, target_idx,
                      self._mask_h[:n_tok] if has_mask else None)
            mgr._dispatch_prefetch(staged, ready_event)
        except Exception:
            mgr._note_cb_failure("expert predictor finish")


def maybe_create_driver(manager):
    """Build the driver, or return None to leave every target to fate.

    Called from ExpertOffloadManager._init_prefetch_state, before the model is
    built, so decoder layers can resolve their capture site during construction.
    """
    global _DRIVER
    _DRIVER = None
    cfg = manager.offload_config
    if not (cfg.expert_offload and cfg.expert_prefetch_enabled):
        return None
    predictor = create_predictor(cfg.expert_predictor,
                                 cfg.expert_predictor_ckpt)
    if predictor is None:
        return None
    _DRIVER = ExpertPredictorDriver(manager, predictor)
    return _DRIVER


def get_driver():
    """The active driver, or None."""
    return _DRIVER


# ---------------------------------------------------------------------- #
#  torch.compile boundary                                                  #
# ---------------------------------------------------------------------- #
# DeepseekV4Model is @support_torch_compile with fullgraph, so the decoder
# layer's forward body is traced by Dynamo. None of what launch() does survives
# that: `with torch_npu.npu.stream(...)` is an unsupported context manager,
# Event()/record_event()/_launch_host_func are untraceable, and — worse than any
# crash — the forward-context guards inside launch() would be evaluated once at
# trace time and specialized away.
#
# A custom op is opaque to Dynamo: one node in the FX graph, body executed
# eagerly per call with the live forward context. Same boundary the MoE apply()
# path already sits behind.
#
# It returns `residual` — a clone the decoder layer makes at both capture sites
# anyway — rather than declaring mutates_args. A returned, live tensor is what
# keeps the node in the graph and pins its position; declaring a mutation
# instead routes through auto_functionalized, which clones the mutated argument
# and would send the staging write to the clone. Same pattern vLLM uses for
# unified_attention, whose KV-cache write is also absent from its schema.
@torch.library.custom_op("vllm_ascend::expert_predictor_launch",
                         mutates_args=())
def expert_predictor_launch(target_idx: int,
                            hidden_states: torch.Tensor) -> torch.Tensor:
    """Clone the block input to `residual` AND start target_idx's prediction.

    The clone IS the atom (har at the first site, hfr at the second), so there
    is no reason to pay for two.
    """
    residual = hidden_states.clone()
    driver = get_driver()
    if driver is not None:
        driver.launch(target_idx, residual)
    return residual


@expert_predictor_launch.register_fake
def _expert_predictor_launch_fake(target_idx: int,
                                  hidden_states: torch.Tensor) -> torch.Tensor:
    # Tracing only: shape, dtype, device and memory format of clone(). Must
    # never touch the driver — there is no live forward context here.
    return torch.empty_like(hidden_states)