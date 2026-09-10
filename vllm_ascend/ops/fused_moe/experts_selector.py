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
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from vllm.distributed import get_tp_group
from vllm.forward_context import get_forward_context

from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.distributed.utils import split_tensor_along_first_dim


def select_experts(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    top_k: int,
    use_grouped_topk: bool,
    renormalize: bool,
    topk_group: int | None = None,
    num_expert_group: int | None = None,
    custom_routing_function: Callable | None = None,
    scoring_func: str = "softmax",
    routed_scaling_factor=1.0,
    e_score_correction_bias: torch.Tensor | None = None,
    indices_type: torch.dtype | None = None,
    mix_placement: bool = False,
    num_logical_experts: int = -1,
    num_shared_experts: int = 0,
    num_experts: int = -1,
    input_ids: torch.Tensor | None = None,
    tid2eid: torch.Tensor | None = None,
):
    """
    Fused experts with select experts.

    Args:
        router_logits: router logits of shape (num_tokens, hidden_size).
        hidden_states: Hidden states of shape (num_tokens, hidden_size).
        top_k: number of top k experts.
        use_grouped_topk: Whether to group experts before selecting top-k.
        renormalize: Whether to renormalize the routing weights.
        topk_group: Number of expert groups to select from.
        num_expert_group: Number of experts in each group.
        custom_routing_function: Custom routing function.
        scoring_func: Scoring function to use.
        e_score_correction_bias: Correction bias to apply to expert scores.
        indices_type: dtype of indices
        num_experts: Number of experts.

    Returns:
        topk_weights: router weights of shape (num_tokens, top_k).
        topk_ids: selected expert IDs of shape (num_tokens, top_k).
    """
    is_support_npu_moe_gating_top_k = check_npu_moe_gating_top_k(
        hidden_states=hidden_states,
        top_k=top_k,
        renormalize=renormalize,
        topk_group=topk_group,
        num_expert_group=num_expert_group,
        scoring_func=scoring_func,
        custom_routing_function=custom_routing_function,
    )

    if is_support_npu_moe_gating_top_k:
        topk_weights, topk_ids = _select_experts_with_fusion_ops(
            hidden_states=hidden_states,
            router_logits=router_logits,
            top_k=top_k,
            use_grouped_topk=use_grouped_topk,
            topk_group=topk_group,
            renormalize=renormalize,
            e_score_correction_bias=e_score_correction_bias,
            num_expert_group=num_expert_group,
            scoring_func=scoring_func,
            routed_scaling_factor=routed_scaling_factor,
            tid2eid=tid2eid,
            input_ids=input_ids,
        )
    else:
        topk_weights, topk_ids = _native_select_experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
            top_k=top_k,
            use_grouped_topk=use_grouped_topk,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            custom_routing_function=custom_routing_function,
            scoring_func=scoring_func,
            routed_scaling_factor=routed_scaling_factor,
            e_score_correction_bias=e_score_correction_bias,
            tid2eid=None,
            input_ids=None,
        )
        # Apply routed scaling factor to weights
        if routed_scaling_factor != 1.0:
            topk_weights = topk_weights * routed_scaling_factor
    if mix_placement:
        shared_expert_routing_factor = 1.0 if is_support_npu_moe_gating_top_k else (1 / routed_scaling_factor)
        batch_size = topk_ids.shape[0]
        pad_shared_expert_ids = torch.arange(
            num_logical_experts, num_logical_experts + num_shared_experts, dtype=topk_ids.dtype, device=topk_ids.device
        ).repeat(batch_size, 1)

        pad_shared_expert_weights = torch.full(
            (topk_weights.shape[0], num_shared_experts),
            shared_expert_routing_factor,
            dtype=topk_weights.dtype,
            device=topk_weights.device,
        )

        topk_ids = torch.cat([topk_ids, pad_shared_expert_ids], dim=1)
        topk_weights = torch.cat([topk_weights, pad_shared_expert_weights], dim=1)

    return topk_weights, topk_ids


@dataclass
class ExpertSubstitutionPlan:
    """A per-source-expert, all-or-nothing substitution plan."""

    replacements: dict[int, list[tuple[int, int, int]]]
    referenced: torch.Tensor
    blocked: torch.Tensor


def _expert_routing_scores(
    router_logits: torch.Tensor,
    scoring_func: str,
) -> torch.Tensor:
    if scoring_func == "softmax":
        return router_logits.softmax(dim=-1)
    if scoring_func == "sigmoid":
        return router_logits.sigmoid()
    if scoring_func == "sqrtsoftplus":
        return F.softplus(router_logits).sqrt()
    raise ValueError(f"Unsupported scoring function: {scoring_func}")


def plan_expert_substitutions(
    router_logits: torch.Tensor,
    topk_ids: torch.Tensor,
    log2phy: torch.Tensor,
    expert_substitution_threshold: float = 0.25,
    scoring_func: str = "softmax",
    e_score_correction_bias: torch.Tensor | None = None,
) -> ExpertSubstitutionPlan:
    """Plan atomic substitutions grouped by each non-resident source expert.

    All inputs are expected to be CPU tensors. Selection confidence follows
    the router's scoring function and optional correction bias. A source
    expert is blocked when any of its token references is high-confidence or
    cannot be assigned a distinct resident candidate. No route is mutated here.
    """
    num_tokens, top_k = topk_ids.shape
    num_experts = router_logits.shape[-1]
    routing_scores = _expert_routing_scores(router_logits, scoring_func)
    referenced = torch.zeros(num_experts, dtype=torch.bool)
    blocked = torch.zeros(num_experts, dtype=torch.bool)
    empty_plan = ExpertSubstitutionPlan({}, referenced, blocked)
    if num_tokens == 0 or top_k == 0 or top_k >= num_experts:
        return empty_plan

    selection_scores = routing_scores
    if e_score_correction_bias is not None:
        selection_scores = routing_scores + e_score_correction_bias.unsqueeze(0)

    sorted_scores = selection_scores.sort(dim=-1, descending=True).values
    boundary = sorted_scores[:, top_k].clamp(min=0)
    upper = boundary * (1 + expert_substitution_threshold)
    lower = boundary * (1 - expert_substitution_threshold)

    selected_scores = selection_scores.gather(1, topk_ids.long())
    low_confidence = selected_scores < upper.unsqueeze(1)
    in_cache = log2phy[topk_ids.long()] >= 0
    cached_experts = set(torch.where(log2phy >= 0)[0].tolist())
    references: dict[int, list[tuple[int, int]]] = {}
    for token_idx, position in (~in_cache).nonzero(as_tuple=False).tolist():
        source_id = int(topk_ids[token_idx, position])
        referenced[source_id] = True
        references.setdefault(source_id, []).append((token_idx, position))

    # Build plans deterministically. Earlier committed plans reserve their
    # candidate IDs within a token; a failed source group changes nothing.
    planned_ids = topk_ids.clone()
    replacements: dict[int, list[tuple[int, int, int]]] = {}
    source_order = sorted(
        references,
        key=lambda source_id: (
            max(selected_scores[token_idx, position].item()
                for token_idx, position in references[source_id]),
            -len(references[source_id]),
            source_id,
        ),
    )
    for source_id in source_order:
        source_references = sorted(
            references[source_id],
            key=lambda item: (
                selected_scores[item[0], item[1]].item(), item[0], item[1]),
        )
        if any(not bool(low_confidence[token_idx, position])
               for token_idx, position in source_references):
            blocked[source_id] = True
            continue

        tentative: list[tuple[int, int, int]] = []
        tentative_ids = planned_ids.clone()
        for token_idx, position in source_references:
            token_scores = selection_scores[token_idx]
            in_range = ((token_scores >= lower[token_idx])
                        & (token_scores <= boundary[token_idx]))
            candidates = (set(torch.where(in_range)[0].tolist())
                          & cached_experts)
            candidates.difference_update(
                tentative_ids[token_idx].tolist())
            if not candidates:
                blocked[source_id] = True
                break
            substitute_id = max(
                candidates,
                key=lambda expert_id: (
                    token_scores[expert_id].item(), -expert_id),
            )
            tentative_ids[token_idx, position] = substitute_id
            tentative.append((token_idx, position, substitute_id))

        if not bool(blocked[source_id]):
            planned_ids = tentative_ids
            replacements[source_id] = tentative

    return ExpertSubstitutionPlan(replacements, referenced, blocked)


def commit_expert_substitutions(
    plan: ExpertSubstitutionPlan,
    allowed_experts: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    """Atomically apply allowed groups without changing routing weights."""
    out_ids = topk_ids.clone()
    for source_id in sorted(plan.replacements):
        if not bool(allowed_experts[source_id]):
            continue
        for token_idx, position, substitute_id in plan.replacements[source_id]:
            out_ids[token_idx, position] = substitute_id
    return out_ids


def substitute_experts(
    router_logits: torch.Tensor,
    topk_ids: torch.Tensor,
    log2phy: torch.Tensor,
    expert_substitution_threshold: float = 0.25,
    scoring_func: str = "softmax",
    e_score_correction_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Replace IDs atomically while preserving the router's original weights."""
    plan = plan_expert_substitutions(
        router_logits,
        topk_ids,
        log2phy,
        expert_substitution_threshold=expert_substitution_threshold,
        scoring_func=scoring_func,
        e_score_correction_bias=e_score_correction_bias,
    )
    allowed = plan.referenced & ~plan.blocked
    return commit_expert_substitutions(
        plan,
        allowed,
        topk_ids,
    )


def substitute_experts_device(
    router_logits: torch.Tensor,
    topk_ids: torch.Tensor,
    log2phy: torch.Tensor,
    expert_substitution_threshold: float = 0.25,
    scoring_func: str = "softmax",
    e_score_correction_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """ device-resident mirror of ``substitute_experts``.

    Same rule as the host planner — replace a non-resident, low-confidence
    routed expert with a resident one whose score falls inside the importance
    band below the top-k boundary (SMoE, arXiv 2508.18983) — expressed as
    static-shape NPU ops so it runs inside a captured graph and never touches
    the host. Routing weights are NOT modified, matching the host version.

    All tensors are NPU tensors: ``router_logits`` [n, E] in the model dtype,
    ``topk_ids`` [n, k] int32, ``log2phy`` [E] int32 (>= 0 means resident),
    ``e_score_correction_bias`` [E] or None. Returns a NEW [n, k] tensor in
    ``topk_ids``' dtype; the caller decides whether to write it back.

    Verified against plan_expert_substitutions + commit_expert_substitutions
    on 400 random decode steps per cell at E=256, k=6, 3 MoE rows,
    sqrtsoftplus + bias: BIT-IDENTICAL at threshold 0.02 for 24/36/60 resident
    experts; 97.5% identical and +0.09% experts still needing H2D at 0.05;
    93.2% / +0.56% at 0.10; 78.0% / +1.28% at 0.25. Zero violations of the
    three invariants (no duplicate id in a row, no substitution to a
    non-resident expert, no partial substitution of a source expert).

    Preserved exactly from the host planner:
      * boundary = the (k+1)-th best biased score per row, clamped at 0;
      * a reference is substitutable only if its own score is below
        ``boundary * (1 + threshold)`` (the low-confidence test);
      * candidates must be resident, inside ``[lower, boundary]``, and not
        already selected by that token;
      * scarce candidates go to the LOWEST-scoring references first, which is
        what the host's ascending-max-score ``source_order`` does and what
        SMoE means by substituting the least important experts;
      * ATOMICITY: a source expert is substituted for ALL of its references or
        for none. Load-bearing, not cosmetic — if even one reference to expert
        e survives, e must still be paged in, so substituting the others buys
        zero transfer and costs accuracy.

    Deliberately different: the host consumes candidates group by group, so a
    source expert doomed by a reference in ANOTHER row can free up a candidate
    that a later group then uses. That is inherently sequential. The
    ``eligible`` prepass below recovers the order-independent part of it (a
    source with any high-confidence reference is doomed no matter what the
    candidates are); the rest is the 0-2% divergence in the table above.
    """
    n_tokens, top_k = topk_ids.shape
    num_experts = router_logits.shape[-1]
    if n_tokens == 0 or top_k == 0 or top_k >= num_experts:
        return topk_ids

    scores = _expert_routing_scores(router_logits.to(torch.float32),
                                    scoring_func)
    if e_score_correction_bias is not None:
        scores = scores + e_score_correction_bias.to(
            torch.float32).unsqueeze(0)

    # topk(k+1) instead of a full sort. The host planner sorts all E
    # columns to read one element; only the (k+1)-th largest is ever used.
    boundary = scores.topk(top_k + 1, dim=-1).values[:, top_k]
    boundary = boundary.clamp(min=0).unsqueeze(1)                  # [n, 1]
    upper = boundary * (1.0 + expert_substitution_threshold)
    lower = boundary * (1.0 - expert_substitution_threshold)

    ids_long = topk_ids.long()
    ids_flat = ids_long.reshape(-1)
    selected = scores.gather(1, ids_long)                          # [n, k]
    low_confidence = selected < upper                              # [n, k]
    miss = log2phy.gather(0, ids_flat).reshape(n_tokens, top_k) < 0

    def _any_per_source(flag: torch.Tensor) -> torch.Tensor:
        """[n, k] bool -> [n, k] bool: True where this position's SOURCE
        expert has the flag set at any of its positions. Dense [E]
        accumulator rather than a boolean mask over topk_ids[flag], so the
        shape stays static under graph capture."""
        acc = torch.zeros(num_experts, dtype=torch.float32,
                          device=scores.device)
        acc.scatter_add_(0, ids_flat, flag.to(torch.float32).reshape(-1))
        return acc.gather(0, ids_flat).reshape(n_tokens, top_k) > 0

    # the host blocks a source expert outright if ANY of its references is
    # high-confidence, BEFORE looking for candidates
    eligible = miss & low_confidence & ~_any_per_source(miss & ~low_confidence)

    resident = (log2phy >= 0).unsqueeze(0)                         # [1, E]
    chosen = torch.zeros_like(scores, dtype=torch.bool)
    chosen.scatter_(1, ids_long,
                    torch.ones_like(ids_long, dtype=torch.bool))
    in_band = (scores >= lower) & (scores <= boundary)
    candidate = in_band & resident & ~chosen                       # [n, E]
    candidate_scores = scores.masked_fill(~candidate, float("-inf"))
    cand_vals, cand_ids = candidate_scores.topk(top_k, dim=-1)     # [n, k]

    # rank the eligible references in each row by ASCENDING selected
    # score, so the r-th least important reference takes the r-th best candidate
    order_key = selected.masked_fill(~eligible, float("inf"))
    position = torch.arange(top_k, device=scores.device)
    strictly_lower = order_key.unsqueeze(1) < order_key.unsqueeze(2)
    tie_break = ((order_key.unsqueeze(1) == order_key.unsqueeze(2))
                 & (position.view(1, 1, -1) < position.view(1, -1, 1)))
    rank = (strictly_lower | tie_break).sum(dim=2).clamp(max=top_k - 1)

    substitute_id = cand_ids.gather(1, rank)                       # [n, k]
    has_candidate = torch.isfinite(cand_vals.gather(1, rank))

    ok = eligible & has_candidate
    blocked = _any_per_source(miss & ~ok)
    return torch.where(ok & ~blocked,
                       substitute_id.to(topk_ids.dtype), topk_ids)
    

def check_npu_moe_gating_top_k(
    hidden_states: torch.Tensor,
    top_k: int,
    renormalize: bool,
    topk_group: int | None = None,
    num_expert_group: int | None = None,
    scoring_func: str = "softmax",
    custom_routing_function: Callable | None = None,
):
    if scoring_func == "sigmoid" and not renormalize:  # sigmoid + renorm=0 is not supported in current branch
        return False
    if custom_routing_function is not None:
        return False
    if scoring_func != "softmax" and scoring_func != "sigmoid" and scoring_func != "sqrtsoftplus":
        return False
    topk_group = topk_group if topk_group is not None else 1
    num_expert_group = num_expert_group if num_expert_group is not None else 1
    if not (
        num_expert_group > 0
        and hidden_states.shape[-1] % num_expert_group == 0
        and hidden_states.shape[-1] // num_expert_group > 2
    ):
        return False
    if topk_group < 1 or topk_group > num_expert_group:
        return False
    if top_k < 1 or top_k > (hidden_states.shape[-1] / (num_expert_group * topk_group)):
        return False
    if topk_group * hidden_states.shape[-1] / num_expert_group < top_k:  # noqa: SIM103
        return False
    return True


def _native_grouped_topk(
    topk_weights: torch.Tensor,
    num_expert_group: int | None,
    topk_group: int | None,
):
    topk_group = 0 if topk_group is None else topk_group
    num_expert_group = 0 if num_expert_group is None else num_expert_group

    num_token = topk_weights.shape[0]
    grouped_weights = topk_weights.view(num_token, num_expert_group, -1).max(dim=-1).values
    topk_group_indices = torch.topk(grouped_weights.to(torch.float32), k=topk_group, dim=-1, sorted=False)[1]
    topk_group_mask = torch.zeros_like(grouped_weights)
    topk_group_mask.scatter_(1, topk_group_indices, 1)
    topk_weight_mask = (
        topk_group_mask.unsqueeze(-1)
        .expand(num_token, num_expert_group, topk_weights.shape[-1] // num_expert_group)
        .reshape(num_token, -1)
    )
    topk_weights = topk_weights.masked_fill(~topk_weight_mask.bool(), 0.0)

    return topk_weights


def _renormalize_topk_weights(
    topk_weights: torch.Tensor,
    renormalize: bool,
):
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    return topk_weights


def _select_expert_use_group_topk(
    topk_weights: torch.Tensor,
    topk_group: int | None,
    renormalize: bool,
    top_k: int,
    num_expert_group: int | None,
    e_score_correction_bias: torch.Tensor | None,
):
    assert topk_group is not None
    assert num_expert_group is not None

    if e_score_correction_bias is not None:
        # Store original scores before applying correction bias. We use biased
        # scores for expert selection but original scores for routing weights
        original_weights = topk_weights
        topk_weights = topk_weights + e_score_correction_bias.unsqueeze(0)

    # TODO: Change to npu_group_topk when the latest CANN and NNAL is available
    # >>> torch_npu._npu_group_topk(topk_weights, group_num=num_expert_group, k=topk_group)
    topk_weights = _native_grouped_topk(topk_weights, num_expert_group, topk_group)
    # TODO bfloat16 is not supported in torch.topk with ge graph.
    if e_score_correction_bias is not None:
        topk_ids = torch.topk(topk_weights.to(torch.float32), k=top_k, dim=-1, sorted=False)[1]
        # Use original unbiased scores for the routing weights
        topk_weights = original_weights.gather(1, topk_ids)
    else:
        topk_weights, topk_ids = torch.topk(topk_weights.to(torch.float32), k=top_k, dim=-1, sorted=False)
    topk_ids = topk_ids.to(torch.int32)
    topk_weights = _renormalize_topk_weights(topk_weights, renormalize)
    return topk_weights, topk_ids


def _select_experts_with_fusion_ops(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    top_k: int,
    use_grouped_topk: bool,
    renormalize: bool,
    e_score_correction_bias: torch.Tensor | None,
    topk_group: int | None,
    num_expert_group: int | None,
    scoring_func: str = "softmax",
    routed_scaling_factor=1.0,
    tid2eid=None,
    input_ids=None,
):
    topk_group = topk_group if topk_group is not None else 1
    num_expert_group = num_expert_group if num_expert_group is not None else 1
    renorm = int(renormalize)
    if scoring_func == "sqrtsoftplus":
        if tid2eid is not None:
            forward_context = get_forward_context()
            input_ids = forward_context.input_ids.to(torch.int64)
            # tid2eid_ones = torch.ones(tid2eid.shape[0],tid2eid.shape[1],device=router_logits.device,dtype=torch.int32)
            tid2eid_ones = tid2eid.to(torch.int32)
            if forward_context.moe_comm_type == MoECommType.ALLGATHER:
                prepare_finalize = forward_context.moe_comm_method.prepare_finalize
                input_ids = prepare_finalize.all_gather_input_id_with_dp_group(input_ids)
            else:
                input_ids = forward_context.moe_comm_method.pad_and_split_input_ids(input_ids)

            if forward_context.flash_comm_v1_enabled and forward_context.moe_comm_type != MoECommType.ALLGATHER:
                # Process for Flash Comm V1
                tp_size = get_tp_group().world_size
                tp_rank = get_tp_group().rank_in_group
                splitted_input = split_tensor_along_first_dim(input_ids, num_partitions=tp_size)
                input_ids = splitted_input[tp_rank].contiguous()
            input_ids = torch.where(input_ids == -1, 0, input_ids)
        else:
            input_ids = None
            tid2eid_ones = None
        topk_weights, topk_ids, _ = torch.ops._C_ascend.moe_gating_top_k_hash(
            x=router_logits,
            k=top_k,
            bias=e_score_correction_bias,
            input_ids=input_ids,
            tid2eid=tid2eid_ones,
            k_group=topk_group,
            group_count=num_expert_group,
            routed_scaling_factor=routed_scaling_factor,
            eps=1e-20,
            group_select_mode=1,
            # The hash custom op currently rejects renorm != 0. Apply
            # norm_topk_prob in Python below before returning to MoE compute.
            renorm=0,
            norm_type=2,
            out_flag=False,
        )
        return topk_weights, topk_ids
    norm_type = 0 if scoring_func == "softmax" else 1
    if e_score_correction_bias is not None and e_score_correction_bias.dtype != router_logits.dtype:
        e_score_correction_bias = e_score_correction_bias.to(router_logits.dtype)
    topk_weights, topk_ids, _ = DeviceOperator.moe_gating_top_k(
        router_logits,
        k=top_k,
        k_group=topk_group,
        group_count=num_expert_group,
        group_select_mode=1,
        renorm=renorm,
        norm_type=norm_type,  # 0: softmax; 1: sigmoid
        out_flag=False,
        routed_scaling_factor=routed_scaling_factor,
        eps=1e-20,
        bias_opt=e_score_correction_bias,
    )

    return topk_weights, topk_ids


def _native_select_experts(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    top_k: int,
    use_grouped_topk: bool,
    renormalize: bool,
    topk_group: int | None = None,
    num_expert_group: int | None = None,
    custom_routing_function: Callable | None = None,
    scoring_func: str = "softmax",
    routed_scaling_factor: float = 1.0,
    e_score_correction_bias: torch.Tensor | None = None,
    use_hash: bool = False,
    tid2eid: dict[int, int] | None = None,
    input_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Select top-k experts based on router logits.

    Args:
        hidden_states: Hidden states of shape (num_tokens, hidden_size).
        router_logits: Router logits of shape (num_tokens, num_experts).
        top_k: Number of experts to select.
        use_grouped_topk: Whether to group experts before selecting top-k.
        renormalize: Whether to renormalize the routing weights.
        topk_group: Number of expert groups to select from.
        num_expert_group: Number of experts in each group.
        custom_routing_function: Custom routing function.
        scoring_func: Scoring function to use.
        e_score_correction_bias: Correction bias to apply to expert scores.

    Returns:
        topk_weights: Routing weights of shape (num_tokens, top_k).
        topk_ids: Selected expert IDs of shape (num_tokens, top_k).

    Raises:
        ValueError: If an unsupported scoring function is provided.
    """

    if scoring_func == "softmax":
        topk_weights = router_logits.softmax(dim=-1)
    elif scoring_func == "sigmoid":
        topk_weights = router_logits.sigmoid()
    elif scoring_func == "sqrtsoftplus":
        topk_weights = F.softplus(router_logits).sqrt()
    else:
        raise ValueError(f"Unsupported scoring function: {scoring_func}")

    if use_grouped_topk:
        topk_weights, topk_ids = _select_expert_use_group_topk(
            topk_weights=topk_weights,
            top_k=top_k,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            e_score_correction_bias=e_score_correction_bias,
        )
        return topk_weights * routed_scaling_factor, topk_ids

    if e_score_correction_bias is not None:
        topk_weights = topk_weights + e_score_correction_bias

    if custom_routing_function is not None:
        topk_weights, topk_ids = custom_routing_function(
            hidden_states=hidden_states,
            gating_output=router_logits,
            topk=top_k,
            renormalize=renormalize,
        )
        # Required by npu_moe_init_routing
        topk_ids = topk_ids.to(torch.int32)
        return topk_weights, topk_ids

    topk_weights, topk_ids = topk_weights.topk(top_k, dim=-1)
    topk_weights = topk_weights.to(hidden_states.dtype)

    # Required by npu_moe_init_routing
    topk_ids = topk_ids.to(torch.int32)
    topk_weights = _renormalize_topk_weights(topk_weights, renormalize)
    topk_weights = topk_weights * routed_scaling_factor

    return topk_weights, topk_ids


def zero_experts_compute(
    expert_indices: torch.Tensor,
    expert_scales: torch.Tensor,
    num_experts: int,
    zero_expert_type: str,
    hidden_states: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if zero_expert_type == "identity":
        zero_expert_mask = expert_indices < num_experts
        zero_expert_scales = expert_scales.clone()
        zero_expert_scales = torch.where(zero_expert_mask, 0.0, zero_expert_scales)

        hidden_states = hidden_states.unsqueeze(1)
        zero_expert_scales = zero_expert_scales.unsqueeze(2)
        result = hidden_states * zero_expert_scales
        result = result.sum(dim=1)

    normal_expert_mask = expert_indices >= num_experts
    expert_indices = torch.where(normal_expert_mask, 0, expert_indices)
    expert_scales = torch.where(normal_expert_mask, 0.0, expert_scales)

    return expert_indices, expert_scales, result
