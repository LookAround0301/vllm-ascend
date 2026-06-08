"""Gate copies for one-layer-ahead expert routing prediction."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch

from vllm_ascend.ops.fused_moe.experts_selector import select_experts


@dataclass
class GateLookaheadEntry:
    """Independent gate module and routing context for a destination MoE layer."""

    gate_copy: torch.nn.Module
    dst_layer: object


class GateLookaheadRegistry:
    """Registry of per-layer gate copies used for lookahead expert prediction."""

    def __init__(self) -> None:
        self.entries: dict[int, GateLookaheadEntry] = {}

    def register_from_moe_layers(self, moe_layers: list) -> None:
        """Deep-copy gate modules for layers 1..N-1 (keyed by destination layer idx)."""
        self.entries.clear()
        for dst_idx in range(1, len(moe_layers)):
            dst_layer = moe_layers[dst_idx]
            gate = getattr(dst_layer, "_gate", None)
            if gate is None:
                continue
            gate_copy = copy.copy(gate)
            gate_copy.eval()
            self.entries[dst_idx] = GateLookaheadEntry(
                gate_copy=gate_copy,
                dst_layer=dst_layer,
            )

    def predict_expert_ids(
        self,
        dst_layer_idx: int,
        hidden_states: torch.Tensor,
    ) -> set[int]:
        """Run the destination gate copy and return unique routed expert ids."""
        entry = self.entries.get(dst_layer_idx)
        if entry is None:
            return set()

        dst_layer = entry.dst_layer
        gate_copy = entry.gate_copy

        with torch.no_grad():
            router_logits, _ = gate_copy(hidden_states)
            _, topk_ids = select_experts(
                hidden_states=hidden_states,
                router_logits=router_logits,
                top_k=dst_layer.top_k,
                use_grouped_topk=dst_layer.use_grouped_topk,
                renormalize=dst_layer.renormalize,
                topk_group=dst_layer.topk_group,
                num_expert_group=dst_layer.num_expert_group,
                custom_routing_function=dst_layer.custom_routing_function,
                scoring_func=dst_layer.scoring_func,
                routed_scaling_factor=dst_layer._original_routed_scaling_factor,
                e_score_correction_bias=dst_layer.e_score_correction_bias,
                num_experts=dst_layer.moe_config.num_experts,
                tid2eid=getattr(dst_layer, "tid2eid", None),
            )
        return set(topk_ids.reshape(-1).tolist())
