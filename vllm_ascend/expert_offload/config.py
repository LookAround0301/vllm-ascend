# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Per-engine O-MoE configuration; importing this module installs no patches."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from packaging.version import Version

if TYPE_CHECKING:
    from vllm.config import VllmConfig

    from vllm_ascend.ascend_config import ExpertOffloadConfig


def expert_residency_limits(
    limit: int, local_experts: int, layer_count: int,
) -> tuple[tuple[int, ...], int]:
    """Keep the first layer resident and at most ``limit`` misses per later layer.

    O-MoE uses TP: every worker owns a shard of every expert, so all workers
    use the same per-layer floor. Misses use separate fixed double buffers.
    """
    minimum = max(1, local_experts - limit)
    floors = (local_experts,) + (minimum,) * (layer_count - 1)
    return floors, sum(floors)


def minimum_expert_slots(vllm_config: "VllmConfig", local_experts: int) -> int:
    raw = (vllm_config.additional_config or {}).get("expert_offload_config", {})
    limit = raw.get("offload_expert_limit", 0)
    layers = vllm_config.model_config.hf_text_config.num_hidden_layers
    return expert_residency_limits(limit, local_experts, layers)[1]


@dataclass(frozen=True)
class OMoEConfig:
    enabled: bool = False
    # TND is faster but can corrupt high-offset shared KV reads on Ascend.
    attention_layout: Literal["bnsd", "tnd"] = "tnd"

    def validate(
        self,
        vllm_config: "VllmConfig",
        expert_config: "ExpertOffloadConfig",
        vllm_version: str,
    ) -> None:
        if self.attention_layout not in ("bnsd", "tnd"):
            raise ValueError("omoe_config.attention_layout must be 'bnsd' or 'tnd'.")
        if not self.enabled:
            if expert_config.offload_expert_limit:
                raise ValueError("offload_expert_limit requires omoe_config.enabled=true")
            return
        if Version(vllm_version).release[:2] != (0, 26):
            raise ValueError(f"O-MoE requires vLLM 0.26.x; got {vllm_version}.")
        model = vllm_config.model_config
        model_type = getattr(getattr(model, "hf_text_config", None), "model_type", None)
        if model_type not in ("deepseek_v4", "qwen3_moe", "qwen3_5_moe_text", "minimax_m2"):
            raise ValueError("O-MoE supports DeepSeek V4, Qwen3 MoE, Qwen3.5 MoE text and MiniMax M2 models.")
        if model_type in ("qwen3_moe", "minimax_m2") and vllm_config.cache_config.enable_prefix_caching:
            raise ValueError("O-MoE hierarchical KV currently requires prefix caching to be disabled.")
        if model_type == "qwen3_5_moe_text":
            from vllm_ascend.core.gdn_state_pages import validate_gdn_state_page_config

            validate_gdn_state_page_config(vllm_config)
        if not model.enforce_eager:
            raise ValueError("O-MoE currently requires enforce_eager=True.")
        if str(getattr(model, "dtype", None)) not in ("torch.bfloat16", "bfloat16"):
            raise ValueError("O-MoE W8A8 currently requires the model dtype to be bfloat16.")
        if not expert_config.expert_offload:
            raise ValueError("O-MoE requires expert_offload_config.expert_offload=true.")
        if expert_config.expert_substitution_enabled:
            raise ValueError("O-MoE correctness mode requires expert_substitution_enabled=false.")
        if expert_config.h2d_backend != "torch":
            raise ValueError("O-MoE shared-pool integration currently requires h2d_backend='torch'.")
        if getattr(vllm_config.scheduler_config, "async_scheduling", None) is not False:
            raise ValueError("O-MoE currently requires --no-async-scheduling for cross-worker pool ownership.")
        if getattr(vllm_config, "lora_config", None) is not None:
            raise ValueError("O-MoE currently does not support MoE LoRA.")
        if getattr(vllm_config, "weight_transfer_config", None) is not None:
            raise ValueError("O-MoE CPU checkpoint sources cannot be updated through weight transfer yet.")
        if getattr(model, "enable_sleep_mode", False):
            raise ValueError("O-MoE shared arena currently requires sleep mode to be disabled.")
        offload = getattr(vllm_config, "offload_config", None)
        if offload is not None and (
            getattr(getattr(offload, "uva", None), "cpu_offload_gb", 0)
            or getattr(getattr(offload, "prefetch", None), "offload_group_size", 0)
            or getattr(offload, "offload_backend", "auto") == "prefetch"
        ):
            raise ValueError("O-MoE cannot be combined with vLLM UVA/prefetch model-weight offloading.")
        if (vllm_config.additional_config or {}).get("mix_placement", False):
            raise ValueError("O-MoE currently does not support mixed shared/routed expert placement.")
        parallel = vllm_config.parallel_config
        if parallel.enable_expert_parallel:
            raise ValueError("O-MoE requires tensor parallelism with expert parallelism disabled.")
        if getattr(vllm_config, "use_v2_model_runner", False):
            raise ValueError("O-MoE shared-pool integration currently requires the v1 model runner.")
        for name in (
            "pipeline_parallel_size",
            "data_parallel_size",
            "prefill_context_parallel_size",
            "decode_context_parallel_size",
        ):
            if getattr(parallel, name) != 1:
                raise ValueError(f"O-MoE currently requires {name}=1.")
        if vllm_config.speculative_config is not None:
            raise ValueError("O-MoE currently requires speculative decoding to be disabled.")
        if vllm_config.kv_transfer_config is not None:
            raise ValueError("O-MoE currently does not support KV transfer.")


def resolve_omoe_config(
    vllm_config: "VllmConfig",
    expert_config: "ExpertOffloadConfig",
    *,
    enabled_by_env: bool,
    vllm_version: str,
) -> OMoEConfig:
    """Resolve and serialize the switch once for this engine.

    An explicit additional_config.omoe_config.enabled takes precedence over
    the environment, including False. Serializing the resolved value keeps
    spawned workers and vLLM's config hash consistent with the parent engine.
    The default-disabled path leaves additional_config completely untouched.
    """
    additional = vllm_config.additional_config
    if additional is None:
        additional = {}
    raw = additional.get("omoe_config", {})
    if not isinstance(raw, dict):
        raise TypeError("additional_config.omoe_config must be a dictionary.")
    unknown = set(raw) - {"enabled", "attention_layout"}
    if unknown:
        raise ValueError(f"Unknown O-MoE configuration keys: {sorted(unknown)}")
    enabled = raw.get("enabled", enabled_by_env)
    if not isinstance(enabled, bool):
        raise TypeError("omoe_config.enabled must be a boolean.")
    config = OMoEConfig(enabled=enabled, attention_layout=raw.get("attention_layout", "tnd"))
    config.validate(vllm_config, expert_config, vllm_version)
    if enabled or "omoe_config" in additional:
        # Do not mutate a caller-owned dict or publish a failed configuration.
        vllm_config.additional_config = {
            **additional,
            "omoe_config": {**raw, "enabled": enabled},
        }
    return config
