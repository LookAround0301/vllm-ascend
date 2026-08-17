#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""In-memory MoE router-gate override (feature F1).

Runtime equivalent of ``tools/swap_gates.py``: instead of rewriting a model's
safetensors shards with fine-tuned router gates and serving the copy, replace
``gate.weight`` in the already-loaded model. Enabled by
``additional_config.moe_gate_override_path``; a no-op when that key is unset.

ORDERING — the one thing that must not move
-------------------------------------------
``apply_moe_gate_override`` must run at the tail of the target model's
``load_weights``, i.e. BEFORE vLLM's ``process_weights_after_loading``. Two copies
of the gate are derived after that point and both are what actually run:

1. ``gate.weight_fp32`` — created by
   ``AscendUnquantizedLinearMethod.process_weights_after_loading`` when
   ``precast_fp32_weight`` is set (it is, for every DSV4 gate). This is the tensor
   ``AscendMoERunner.shared_forward_impl`` feeds to ``F.linear`` to produce
   ``router_logits``; ``gate.weight`` itself is not on the inference path.
2. ``ExpertOffloadManager._gate_weights_npu`` — an fp32 clone taken in
   ``register_gate_weights`` during ``_finalize_offload``, used for prefetch
   prediction.

The same function also re-lays ``gate.weight`` out in FRACTAL_NZ when
``weight_nz_mode == 2``, so writing after it would mean writing into an NZ tensor.
Running before it makes all of that moot. ``verify_moe_gate_override`` asserts the
ordering still holds; ``_refresh_precast`` below degrades gracefully if it ever
does not.
"""

from __future__ import annotations

import glob
import hashlib
import os
import re
from dataclasses import dataclass, field

import torch
from torch import nn
from vllm.logger import logger

# Path components that mark a DRAFT router (serial MTP, DSpark, next-n). Draft
# gates are NEVER overridden: the draft is a separate module with its own
# load_weights, and target/draft routers must come from the same fine-tune or
# acceptance rate drops. Component matching (not substring) so "mtp_block" and
# "mtp_model_layer" are both caught via their "mtp" component.
_DRAFT_COMPONENTS = frozenset({"mtp", "nextn", "next_n", "draft", "spec"})

# The ONLY accepted layer-index form: "layers.7", "layers_7", "layer.7".
# Deliberately no bare-digit fallback — see _parse_layer_index.
_LAYER_RE = re.compile(r"(?:^|[._])layers?[._](\d+)")

# A router-weight key ends in "<something>.gate.weight" or ".router.weight"
# (vLLM/DeepSeek spelling and Megatron spelling respectively).
_ROUTER_KEY_RE = re.compile(r"(?:^|\.)(?:gate|router)\.weight$")

# Model-side parameter suffix. endswith() rather than '"gate" in name' because
# expert MLP projections are named "...experts.{e}.gate_proj.weight" and shared
# experts "...shared_experts.gate_up_proj.weight"; overwriting either with a
# router matrix is silent and catastrophic.
_GATE_SUFFIX = ".gate.weight"

# Bias-looking keys. Per D2 the current fine-tunes carry no router bias, so these
# are REPORTED, not applied — a new router matrix combined with a stale
# e_score_correction_bias is neither the old router nor the new one, and DSV4
# selection is sqrtsoftplus(logits) + bias inside grouped top-k
# (experts_selector.py :: _select_experts_with_fusion_ops), so half-applying is
# worse than not applying. If an artefact ever grows one, this makes it visible.
_BIAS_KEY_RE = re.compile(
    r"(?:^|\.)(?:gate\.e_score_correction_bias|gate\.bias|router\.expert_bias|expert_bias)$")

# torch.load attempts, in order. weights_only=True first so a slim gate dump never
# executes pickle; mmap=True so a full trainer checkpoint is mapped rather than
# materialised (this matters because EVERY rank loads this file independently, and
# the launch script already aborts above 78% host RAM). Legacy non-zipfile .pt
# files reject mmap, hence the third attempt.
_LOAD_ATTEMPTS = (
    {"weights_only": True, "mmap": True},
    {"weights_only": False, "mmap": True},
    {"weights_only": False, "mmap": False},
)

_FINGERPRINT_BYTES = 64 * 1024 * 1024


@dataclass
class GateCheckpoint:
    """Parsed router gates plus the diagnostics needed to trust them."""

    gates: dict[int, torch.Tensor]
    path: str
    sha16: str
    size: int
    skipped_draft: list[str] = field(default_factory=list)
    skipped_bias: list[str] = field(default_factory=list)
    ignored: list[str] = field(default_factory=list)


def _is_draft_key(key: str) -> bool:
    return any(c in _DRAFT_COMPONENTS for c in re.split(r"[._/\-]", key.lower()) if c)


def _parse_layer_index(key: str) -> int | None:
    """Layer index from a parameter/checkpoint key, or None if there isn't one.

    Stricter than tools/swap_gates.py :: _layer_num on purpose. That one is
        re.search(r"(?:layers?[._])(\\d+)", name) or re.search(r"(\\d+)", name)
    which has two live bugs:
      * re.search returns the FIRST match, so a Megatron MTP key
        "mtp.layers.0.mtp_model_layer.layers.1.mlp.router.weight" resolves to 0 and
        an MTP router lands on TARGET LAYER 0, reported as a success;
      * the bare-digit fallback turns "gate_fp32" into layer 32.
    Draft keys are rejected before this is ever called, and more than one
    "layers.N" group raises rather than guessing.
    """
    if key.isdigit():
        return int(key)
    found = _LAYER_RE.findall(key)
    if len(found) == 1:
        return int(found[0])
    if len(found) > 1:
        raise ValueError(
            f"ambiguous layer index in gate-checkpoint key '{key}': found "
            f"{found}. Refusing to guess — re-dump this checkpoint as "
            f"{{'gates': {{<layer_index>: tensor}}}}.")
    return None


def _looks_like_router_key(key: str) -> bool:
    if key.isdigit():
        return True  # a flat {layer_index: tensor} dump with no "gates" wrapper
    if _ROUTER_KEY_RE.search(key):
        return True
    # tools/swap_gates.py accepts any key containing "gate"; keep the two paths
    # behaviourally identical on real dumps. Safe because the caller additionally
    # requires ndim == 2 and every accepted tensor is shape-checked against the
    # model's gate before it is written. The exclusions are the junk a dump taken
    # from a quantized checkpoint carries alongside the router.
    low = key.lower()
    if any(token in low for token in ("gate_proj", "gate_up_proj", "scale", "offset", "bias")):
        return False
    return "gate" in low


def _resolve_ckpt_path(path: str) -> str:
    """File → itself. Directory → gate_weights.pt, else the single *.pt/*.bin."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"moe_gate_override_path does not exist: {path}")
    if not os.path.isdir(path):
        return path
    named = os.path.join(path, "gate_weights.pt")
    if os.path.exists(named):
        return named
    cands = sorted(glob.glob(os.path.join(path, "*.pt")) + glob.glob(os.path.join(path, "*.bin")))
    if not cands:
        raise FileNotFoundError(
            f"moe_gate_override_path is a directory with no gate_weights.pt and no "
            f"*.pt / *.bin inside: {path}")
    if len(cands) > 1:
        # swap_gates.py takes next(iter(glob(...))), i.e. whichever the filesystem
        # happens to return first. Which gate checkpoint a benchmark ran with is
        # not something to leave to directory order.
        raise ValueError(
            f"moe_gate_override_path is a directory with {len(cands)} candidate "
            f"files: {[os.path.basename(c) for c in cands]}. Name one "
            f"gate_weights.pt, or point the key at a file.")
    return cands[0]


def _file_fingerprint(path: str) -> tuple[str, int]:
    """(sha256 prefix of the first 64 MiB, size). Cross-rank divergence detector.

    Every rank loads this file independently; if one reads a different or partial
    file the ranks route differently, which under EP means inconsistent dispatch
    with no error anywhere. Logging this makes it a grep instead of a mystery.
    Truncated so it stays cheap on a large trainer checkpoint.
    """
    size = os.path.getsize(path)
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        digest.update(handle.read(_FINGERPRINT_BYTES))
    return digest.hexdigest()[:16], size


def _torch_load(path: str):
    last_exc: Exception | None = None
    for attempt, kwargs in enumerate(_LOAD_ATTEMPTS):
        try:
            return torch.load(path, map_location="cpu", **kwargs)
        except Exception as exc:  # noqa: BLE001 — every failure mode falls back
            last_exc = exc
            logger.warning(
                "[GATE-OVERRIDE] torch.load(%s, %s) failed: %s: %s",
                path, kwargs, type(exc).__name__, exc)
            if attempt == len(_LOAD_ATTEMPTS) - 1:
                raise
    raise AssertionError(f"unreachable: {last_exc}")  # pragma: no cover


def load_gate_checkpoint(path: str) -> GateCheckpoint:
    """Parse a fine-tuned router checkpoint into {layer_index: gate tensor}.

    Two formats, matching tools/swap_gates.py :: load_gates:
      A trainer/extracted:  {"gates": {layer_index: Tensor[E, H]}, "opt": ...}
                            keys may be int or str
      B flat dump:          {"model.layers.7.mlp.gate.weight": Tensor[E, H], ...}
                            tried only if A yields nothing
    Silence is the failure mode that costs a whole benchmark run, so anything
    unparsable is either raised on or listed in the returned diagnostics.
    """
    resolved = _resolve_ckpt_path(path)
    sha16, size = _file_fingerprint(resolved)
    state = _torch_load(resolved)
    if not isinstance(state, dict):
        raise TypeError(
            f"gate checkpoint {resolved} must be a dict, got {type(state).__name__}")

    gates: dict[int, torch.Tensor] = {}
    skipped_draft: list[str] = []
    skipped_bias: list[str] = []
    ignored: list[str] = []

    def _accept(key: str, value, idx: int) -> None:
        if idx in gates:
            raise ValueError(
                f"gate checkpoint {resolved} maps layer {idx} twice (second key "
                f"'{key}'). Refusing to pick one.")
        # clone(): drops the reference to the surrounding container so a trainer
        # checkpoint's optimizer state (or a shared legacy storage blob) is freed
        # by the `del state` below. 43 x [256, 4096] fp32 is ~180 MB, which is
        # nothing next to what it releases.
        gates[idx] = value.detach().clone()

    inner = state.get("gates")
    if isinstance(inner, dict):
        for key_obj, value in inner.items():
            key = str(key_obj)
            if _is_draft_key(key):
                skipped_draft.append(key)
                continue
            if _BIAS_KEY_RE.search(key):
                skipped_bias.append(key)
                continue
            if not torch.is_tensor(value):
                ignored.append(key)
                continue
            if isinstance(key_obj, int) and not isinstance(key_obj, bool):
                idx: int | None = key_obj
            else:
                idx = _parse_layer_index(key)
            if idx is None:
                ignored.append(key)
                continue
            _accept(key, value, idx)

    if not gates:
        for key_obj, value in state.items():
            key = str(key_obj)
            if not torch.is_tensor(value):
                continue
            # Draft check FIRST, and it is the primary defence rather than a
            # backstop: the vLLM-side MTP name "model.layers.43.mtp_block.mlp.
            # gate.weight" has exactly one "layers.N" group, so _parse_layer_index
            # would happily return 43 and nothing else would notice.
            if _is_draft_key(key):
                skipped_draft.append(key)
                continue
            if _BIAS_KEY_RE.search(key):
                skipped_bias.append(key)
                continue
            if not _looks_like_router_key(key):
                continue  # a trainer checkpoint has thousands; not worth reporting
            if value.ndim != 2:
                ignored.append(key)  # router-named but not a matrix
                continue
            idx = _parse_layer_index(key)
            if idx is None:
                ignored.append(key)
                continue
            _accept(key, value, idx)

    if not gates:
        raise ValueError(
            f"no router-gate tensors parsed from {resolved}. Top-level keys: "
            f"{list(state.keys())[:10]}; router-named but unusable: {ignored[:10]}. "
            f"Expected either {{'gates': {{<layer_index>: tensor}}}} or a flat dict "
            f"of '*.gate.weight' / '*.router.weight' entries.")
    del state  # release the trainer payload before the rest of the model loads

    return GateCheckpoint(
        gates=dict(sorted(gates.items())),
        path=resolved,
        sha16=sha16,
        size=size,
        skipped_draft=skipped_draft,
        skipped_bias=skipped_bias,
        ignored=ignored,
    )


def _is_hash_gate(gate_module: nn.Module, layer_idx: int, num_hash_layers: int) -> bool:
    """True for a hash-routed (table-driven) layer.

    DeepseekV4MoE sets gate.tid2eid to an nn.Parameter when
    `layer_idx < config.num_hash_layers and not is_draft_layer`, and to None
    otherwise (deepseek_v4.py:418-433), so the attribute is an exact test and does
    not depend on the config being read correctly. The index comparison is a second
    condition, not the primary one.
    """
    return getattr(gate_module, "tid2eid", None) is not None or layer_idx < num_hash_layers


def _describe_quant(model: nn.Module) -> str:
    """Best-effort one-line description of the quantisation in force.

    Every read is guarded: this is for a log line, and the attribute names on
    AscendQuantConfig are not part of any contract. `quant_description` keys are
    the ones w4a8.py reads (w4a8.py:95-96, 358).
    """
    quant_config = getattr(model, "quant_config", None)
    if quant_config is None:
        return "none (bf16)"
    name = type(quant_config).__name__
    description = getattr(quant_config, "quant_description", None)
    if isinstance(description, dict):
        bits = [f"{k}={description[k]}"
                for k in ("ascend_quant_method", "version", "group_size")
                if k in description]
        if bits:
            return f"{name}({', '.join(bits)})"
    return name


def _log_scope(model: nn.Module, config, num_gates: int, num_hash_layers: int) -> None:
    """State what this feature is validated for, and flag anything outside it.

    Not a hard gate — the code path is generic and the call site already confines it
    to AscendDeepseekV4ForCausalLM. The point is that serve.log records, for every
    run, exactly which model and quantisation the override was applied to.
    """
    model_type = str(getattr(config, "model_type", "?"))
    architectures = list(getattr(config, "architectures", None) or [])
    # AscendMoERunner.quant_type is resolved at construction (fused_moe.py:489), so
    # it is already correct here. getattr-guarded because moe_layers may hold the
    # FusedMoE wrapper rather than the runner depending on the vLLM version.
    moe_quant_types = sorted({
        str(getattr(layer, "quant_type", None))
        for layer in (getattr(model, "moe_layers", None) or [])
    })
    logger.info(
        "[GATE-OVERRIDE] scope: validated ONLY for the DeepSeek-V4-Flash / "
        "DeepSeek-V4-Flash-MTP family with W4A8 or W8A8 quantisation, target model "
        "only (never the MTP/DSpark draft), learned-routing layers only (hash layers "
        "are skipped). Observed: model_type=%s architectures=%s quant=%s "
        "moe_quant_type=%s moe_gates=%d num_hash_layers=%d n_routed_experts=%s "
        "hidden_size=%s",
        model_type, architectures, _describe_quant(model), moe_quant_types or "?",
        num_gates, num_hash_layers,
        getattr(config, "n_routed_experts", "?"), getattr(config, "hidden_size", "?"))
    if model_type != "deepseek_v4":
        logger.warning(
            "[GATE-OVERRIDE] model_type=%s is OUTSIDE the validated scope for this "
            "feature (deepseek_v4). Proceeding, but nothing here has been checked "
            "against this architecture's gate naming or routing.", model_type)
    if getattr(model, "quant_config", None) is None:
        logger.warning(
            "[GATE-OVERRIDE] no quant_config (bf16) — OUTSIDE the validated scope "
            "(W4A8 / W8A8). The override itself is quantisation-independent (the "
            "router gate is built with quant_config=None either way), but note that "
            "expert offload is a no-op for an unquantized MoE on this build.")


def _refresh_precast(gate_module: nn.Module, param: torch.Tensor) -> None:
    """Keep gate.weight_fp32 in step, if it exists yet.

    At the tail of load_weights it does NOT exist, so this is a no-op — it is here
    so that if the loader ordering ever changes, or this helper is called from a
    later hook, the feature degrades to CORRECT rather than to SILENTLY INERT.
    weight_fp32 is what shared_forward_impl routes with (fused_moe.py:1225), and
    it is always ND fp32 (_should_trans_nz returns False for fp32 before any other
    branch, utils.py:253-256), so a plain copy_ is safe here regardless of
    weight_nz_mode.
    """
    precast = getattr(gate_module, "weight_fp32", None)
    if precast is not None:
        precast.copy_(param.data.float())


def apply_moe_gate_override(
    model: nn.Module,
    params_dict: dict[str, torch.Tensor],
    loaded_params: set[str],
    path: str,
) -> None:
    """Overwrite the router gate of every LEARNED-ROUTING MoE layer the checkpoint
    covers. Call at the tail of the TARGET model's load_weights, before
    `return loaded_params`. Hash-routed layers and draft models are never written."""
    model_gates: dict[int, str] = {}
    for name in params_dict:
        if not name.endswith(_GATE_SUFFIX) or "gate_proj" in name:
            continue
        idx = _parse_layer_index(name)
        if idx is None:
            raise ValueError(f"cannot parse a layer index from gate parameter '{name}'")
        if idx in model_gates:
            raise ValueError(
                f"two gate parameters map to layer {idx}: "
                f"'{model_gates[idx]}' and '{name}'")
        model_gates[idx] = name
    if not model_gates:
        raise RuntimeError(
            f"moe_gate_override_path={path} is set but this model exposes no "
            f"'*{_GATE_SUFFIX}' parameters. The gate naming changed; the override "
            f"would be a silent no-op.")

    config = getattr(model, "config", None)
    num_hash_layers = int(getattr(config, "num_hash_layers", 0) or 0)
    # Before loading the checkpoint, so serve.log carries the scope line even when
    # the checkpoint itself is what fails.
    _log_scope(model, config, len(model_gates), num_hash_layers)

    ckpt = load_gate_checkpoint(path)

    matched = sorted(set(ckpt.gates) & set(model_gates))
    if not matched:
        raise ValueError(
            f"gate checkpoint {ckpt.path} covers layers "
            f"{sorted(ckpt.gates)[:8]}... but this rank's model has gates for layers "
            f"{sorted(model_gates)[:8]}... — zero overlap. Wrong checkpoint, or a "
            f"layer-index convention mismatch. (tools/swap_gates.py keys on the "
            f"model's GLOBAL decoder-layer index; the producer must too.)")

    # HASH EXCLUSION. Hash-routed layers take their expert ids from gate.tid2eid, not
    # from the gate matrix, and are deliberately left at their base weights. Delete
    # this partition (apply to `matched` directly) to restore exact tools/swap_gates.py
    # parity, which rewrites all layers — see the guide's Findings (3) for the
    # trade-off, since router_logits IS still passed to moe_gating_top_k_hash and so
    # the hash-layer gate may influence topk_weights.
    covered: list[int] = []
    hash_skipped: list[int] = []
    gate_modules: dict[int, nn.Module] = {}
    for idx in matched:
        name = model_gates[idx]
        gate_module = model.get_submodule(name[: -len(".weight")])
        gate_modules[idx] = gate_module
        if _is_hash_gate(gate_module, idx, num_hash_layers):
            hash_skipped.append(idx)
        else:
            covered.append(idx)
    if not covered:
        raise ValueError(
            f"gate checkpoint {ckpt.path} matches only hash-routed layers "
            f"{hash_skipped}, which this feature deliberately does not write. "
            f"Nothing would change. Check the checkpoint's layer indexing against "
            f"the model's (num_hash_layers={num_hash_layers}).")

    # {param_name: fp32 sum of the overridden weight}. Read back by
    # verify_moe_gate_override() after process_weights_after_loading to prove the
    # override reached gate.weight_fp32. Stored on the model because that is the
    # only object both call sites share.
    override_state: dict[str, float] = {}
    total_numel = 0
    for idx in covered:
        name = model_gates[idx]
        param = params_dict[name]
        new = ckpt.gates[idx]
        if list(new.shape) != list(param.shape):
            raise ValueError(
                f"gate shape mismatch at layer {idx} ('{name}'): checkpoint "
                f"{list(new.shape)} vs model {list(param.shape)}. Wrong "
                f"n_routed_experts or hidden_size — continuing would corrupt routing.")
        # .to(param.dtype) ON THE CPU TENSOR, before copy_. This is load-bearing for
        # bit-parity with tools/swap_gates.py, which writes
        # `new.to(sd[name].dtype).contiguous()` into the safetensors shard. Letting
        # copy_ do the fp32->bf16 cast on device instead could round differently and
        # would break the offline/runtime equivalence test.
        # .contiguous(): a saved tensor may be a non-contiguous view.
        # .data: never let autograd see this.
        param.data.copy_(new.to(param.dtype).contiguous())
        loaded_params.add(name)
        _refresh_precast(gate_modules[idx], param)
        weight_sum = float(param.data.float().sum().item())
        override_state[name] = weight_sum
        total_numel += param.numel()
        logger.debug(
            "[GATE-OVERRIDE] layer %d %s shape=%s dtype=%s sum=%.6e",
            idx, name, list(param.shape), param.dtype, weight_sum)

    model._moe_gate_override_state = override_state

    unused = sorted(set(ckpt.gates) - set(model_gates))
    logger.info(
        "[GATE-OVERRIDE] applied: swapped=%d/%d gates layers=[%d..%d] "
        "hash_skipped=%s unused_ckpt=%d%s draft_skipped=%d bias_ignored=%d "
        "unparsable=%d ckpt=%s size=%dB sha256_16=%s sum=%.6e numel=%d",
        len(covered), len(model_gates), covered[0], covered[-1],
        hash_skipped or "[]", len(unused), f" {unused[:16]}" if unused else "",
        len(ckpt.skipped_draft), len(ckpt.skipped_bias), len(ckpt.ignored),
        ckpt.path, ckpt.size, ckpt.sha16, sum(override_state.values()), total_numel)

    if hash_skipped:
        logger.warning(
            "[GATE-OVERRIDE] %d hash-routed layer(s) %s are present in the checkpoint "
            "and were NOT written, by design. tools/swap_gates.py DOES write them, so "
            "this run is not bit-identical to a swap_gates.py --out model on those "
            "layers.", len(hash_skipped), hash_skipped)
    if ckpt.skipped_draft:
        logger.info(
            "[GATE-OVERRIDE] draft (MTP/DSpark) entries in the checkpoint are NOT "
            "applied — the draft has its own load_weights and target/draft routers "
            "must come from the same fine-tune: %s", ckpt.skipped_draft[:8])
    if ckpt.skipped_bias:
        logger.warning(
            "[GATE-OVERRIDE] the checkpoint contains %d router-BIAS entries which "
            "are NOT applied (%s). DSV4 selects with sqrtsoftplus(logits) + "
            "e_score_correction_bias, so a new gate matrix with the base bias is "
            "neither router. If this fine-tune trained the bias, this needs a "
            "decision before the numbers mean anything.",
            len(ckpt.skipped_bias), ckpt.skipped_bias[:8])
    if ckpt.ignored:
        logger.info("[GATE-OVERRIDE] ignored %d unparsable checkpoint keys: %s",
                    len(ckpt.ignored), ckpt.ignored[:8])
    # The off-by-num_hash_layers trap: DSV4's learned routers are layers
    # [num_hash_layers, L), so a fine-tune that indexed only its trained layers from 0
    # produces a set that loads cleanly and writes every gate one block too early.
    # Hash exclusion does NOT protect against this — it only drops the first
    # num_hash_layers entries; the rest still land on the wrong layers.
    if (num_hash_layers
            and set(ckpt.gates) == set(range(len(ckpt.gates)))
            and len(ckpt.gates) == len(model_gates) - num_hash_layers):
        logger.warning(
            "[GATE-OVERRIDE] the checkpoint covers exactly layers [0..%d], which is "
            "the count of LEARNED-routing MoE layers (%d = %d - %d num_hash_layers). "
            "If the producer indexed only its trained layers, every gate just written "
            "went to the WRONG layer, shifted by %d. Confirm the producer keys on the "
            "model's global decoder-layer index, as tools/swap_gates.py does.",
            len(ckpt.gates) - 1, len(ckpt.gates), len(model_gates),
            num_hash_layers, num_hash_layers)


def verify_moe_gate_override(model: nn.Module) -> None:
    """Assert the override reached the tensor that actually routes.

    Call once after get_model() returns, i.e. after
    process_weights_after_loading has derived gate.weight_fp32. No-op when the
    feature is off. Compares the fp32 precast against the sum recorded at swap
    time rather than re-reading gate.weight, because gate.weight may have been
    re-laid-out to FRACTAL_NZ by then (linear.py :: maybe_trans_nz) while
    weight_fp32 is always ND fp32.
    """
    override_state = getattr(model, "_moe_gate_override_state", None)
    if not override_state:
        return

    checked = 0
    missing = 0
    for name, expected in override_state.items():
        gate_module = model.get_submodule(name[: -len(".weight")])
        precast = getattr(gate_module, "weight_fp32", None)
        if precast is None:
            missing += 1
            continue
        got = float(precast.float().sum().item())
        if abs(got - expected) > 1e-6 * max(1.0, abs(expected)):
            raise RuntimeError(
                f"[GATE-OVERRIDE] the override did not reach the tensor that routes: "
                f"{name[: -len('.weight')]}.weight_fp32 sums to {got:.6e} but the "
                f"overridden gate.weight summed to {expected:.6e}. weight_fp32 is "
                f"derived in AscendUnquantizedLinearMethod.process_weights_after_"
                f"loading (linear.py) and is what AscendMoERunner.shared_forward_impl "
                f"feeds to F.linear for router_logits. Either the swap no longer runs "
                f"before that function, and _refresh_precast did not repair it, or "
                f"weight_fp32 is no longer derived from gate.weight. Check the tail "
                f"of AscendDeepseekV4ForCausalLM.load_weights first.")
        checked += 1

    if missing == len(override_state):
        logger.info(
            "[GATE-OVERRIDE] verified: %d overridden gates, none carrying "
            "weight_fp32 — routing reads gate.weight directly on this build",
            len(override_state))
    else:
        logger.info(
            "[GATE-OVERRIDE] verified: %d/%d overridden gates match their fp32 "
            "routing copy (%d without weight_fp32)",
            checked, len(override_state), missing)