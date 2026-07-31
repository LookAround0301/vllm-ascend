from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.quantization.methods.w4a8_mxfp4 import AscendW4A8MXFPDynamicFusedMoEMethod


NUM_EXPERTS = 8
HIDDEN_SIZE = 16
TOP_K = 2


def _make_method() -> AscendW4A8MXFPDynamicFusedMoEMethod:
    method = AscendW4A8MXFPDynamicFusedMoEMethod.__new__(AscendW4A8MXFPDynamicFusedMoEMethod)
    method.dynamic_eplb = False
    return method


def _make_layer(*, multi_card: bool) -> SimpleNamespace:
    return SimpleNamespace(
        enable_expert_offload=True,
        enable_multi_card=multi_card,
        moe_config=SimpleNamespace(num_local_experts=4),
        local_num_experts=4,
        w13_weight=object(),
        w2_weight=object(),
        w13_weight_scale=object(),
        w2_weight_scale=object(),
        swiglu_limit=None,
    )


def _make_manager(layer: SimpleNamespace, *, offload_threshold: int = 4) -> Mock:
    manager = Mock()
    manager.offload_threshold = offload_threshold
    manager._prefill_initialized = True
    manager._skip_prefill = False
    manager.moe_layers = [layer]
    manager.num_total_experts = NUM_EXPERTS
    manager.mc_shard_size = NUM_EXPERTS // 2
    manager._prefill_w13 = [object()]
    manager._prefill_w2 = [object()]
    manager._prefill_w13_scale = [object()]
    manager._prefill_w2_scale = [object()]
    manager._prefill_log2phy = torch.arange(NUM_EXPERTS, dtype=torch.int32)
    return manager


def _run_apply(
    method: AscendW4A8MXFPDynamicFusedMoEMethod,
    layer: SimpleNamespace,
    manager: Mock,
    comm_method: Mock,
    comm_type: MoECommType,
    *,
    num_tokens: int,
    mc2_mask: torch.Tensor | None = None,
    build_side_effect=None,
):
    x = torch.randn(num_tokens, HIDDEN_SIZE, dtype=torch.bfloat16)
    router_logits = torch.randn(num_tokens, NUM_EXPERTS)
    topk_weights = torch.randn(num_tokens, TOP_K, dtype=x.dtype)
    topk_ids = torch.randint(0, NUM_EXPERTS, (num_tokens, TOP_K))
    log2phy = torch.arange(NUM_EXPERTS, dtype=torch.int32)
    fused_input = object()
    expected_output = torch.randn(num_tokens, HIDDEN_SIZE)
    comm_method.fused_experts.return_value = expected_output

    with (
        patch(
            "vllm_ascend.quantization.methods.w4a8_mxfp4.get_moe_num_logical_experts",
            return_value=NUM_EXPERTS,
        ),
        patch(
            "vllm_ascend.quantization.methods.w4a8_mxfp4.select_experts",
            return_value=(topk_weights, topk_ids),
        ),
        patch(
            "vllm_ascend.quantization.methods.w4a8_mxfp4.get_forward_context",
            return_value=SimpleNamespace(moe_comm_method=comm_method, moe_comm_type=comm_type),
        ),
        patch(
            "vllm_ascend.quantization.methods.w4a8_mxfp4.build_fused_experts_input",
            return_value=fused_input,
            side_effect=build_side_effect,
        ) as build_input,
        patch(
            "vllm_ascend.expert_offload.ExpertOffloadManager.get_instance",
            return_value=manager,
        ),
    ):
        output = method.apply(
            layer=layer,
            x=x,
            router_logits=router_logits,
            top_k=TOP_K,
            renormalize=True,
            num_experts=NUM_EXPERTS,
            enable_force_load_balance=False,
            log2phy=log2phy,
            mc2_mask=mc2_mask,
        )

    assert output is expected_output
    return SimpleNamespace(
        x=x,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        log2phy=log2phy,
        fused_input=fused_input,
        build_input=build_input,
    )


def test_single_card_decode_updates_weights_and_prefetches_after_gmm():
    method = _make_method()
    layer = _make_layer(multi_card=False)
    manager = _make_manager(layer)
    dispatcher = SimpleNamespace(num_experts_local=4)
    comm_method = Mock(token_dispatcher=dispatcher)
    call_order = Mock()
    call_order.attach_mock(comm_method.fused_experts, "fused_experts")
    call_order.attach_mock(manager.trigger_next_layer_prefetch, "prefetch")

    result = _run_apply(method, layer, manager, comm_method, MoECommType.MC2, num_tokens=2)

    update_call = manager.update_weights.call_args
    assert update_call.args[0] is layer
    assert update_call.args[1] is result.topk_ids
    assert update_call.args[2] is result.log2phy
    assert update_call.args[3] is result.topk_weights
    assert update_call.kwargs["hidden_states"] is result.x
    manager.update_weights_multi_card.assert_not_called()
    manager.trigger_next_layer_prefetch.assert_called_once_with(layer, result.x)
    assert [call[0] for call in call_order.method_calls] == ["fused_experts", "prefetch"]

    build_kwargs = result.build_input.call_args.kwargs
    assert build_kwargs["w1"] is layer.w13_weight
    assert build_kwargs["w2"] is layer.w2_weight
    assert build_kwargs["log2phy"] is result.log2phy


def test_multi_card_decode_uses_multi_card_weight_update():
    method = _make_method()
    layer = _make_layer(multi_card=True)
    manager = _make_manager(layer)
    dispatcher = SimpleNamespace(num_experts_local=4)
    comm_method = Mock(token_dispatcher=dispatcher)
    mc2_mask = torch.tensor([True, False])

    result = _run_apply(
        method,
        layer,
        manager,
        comm_method,
        MoECommType.MC2,
        num_tokens=2,
        mc2_mask=mc2_mask,
    )

    manager.update_weights.assert_not_called()
    update_call = manager.update_weights_multi_card.call_args
    assert update_call.args[0] is layer
    assert update_call.args[1] is result.topk_ids
    assert update_call.args[2] is result.log2phy
    assert update_call.args[3] is result.topk_weights
    assert update_call.kwargs["hidden_states"] is result.x
    assert update_call.kwargs["mc2_mask"] is mc2_mask
    manager.trigger_next_layer_prefetch.assert_called_once_with(layer, result.x)
    assert result.build_input.call_args.kwargs["log2phy"] is result.log2phy


def test_single_card_prefill_uses_full_expert_pool_and_restores_counts():
    method = _make_method()
    layer = _make_layer(multi_card=False)
    manager = _make_manager(layer)
    dispatcher = SimpleNamespace(num_experts_local=4)
    comm_method = Mock(token_dispatcher=dispatcher)
    state_during_build = {}

    def capture_state(**kwargs):
        state_during_build["layer_count"] = layer.local_num_experts
        state_during_build["dispatcher_count"] = dispatcher.num_experts_local
        return object()

    result = _run_apply(
        method,
        layer,
        manager,
        comm_method,
        MoECommType.ALLGATHER,
        num_tokens=8,
        build_side_effect=capture_state,
    )

    manager.update_weights.assert_called_once()
    manager.update_weights_multi_card.assert_not_called()
    manager.trigger_next_layer_prefetch.assert_not_called()
    assert state_during_build == {"layer_count": NUM_EXPERTS, "dispatcher_count": NUM_EXPERTS}
    assert layer.local_num_experts == 4
    assert layer.moe_config.num_local_experts == 4
    assert dispatcher.num_experts_local == 4

    build_kwargs = result.build_input.call_args.kwargs
    assert build_kwargs["w1"] is manager._prefill_w13[0]
    assert build_kwargs["w2"] is manager._prefill_w2[0]
    assert build_kwargs["w1_scale"] is manager._prefill_w13_scale[0]
    assert build_kwargs["w2_scale"] is manager._prefill_w2_scale[0]
    assert build_kwargs["log2phy"] is manager._prefill_log2phy


def test_multi_card_prefill_uses_ep_shard_and_restores_dispatcher():
    method = _make_method()
    layer = _make_layer(multi_card=True)
    manager = _make_manager(layer)
    manager._skip_prefill = True
    shard_map = torch.tensor([-1, -1, -1, -1, 0, 1, 2, 3], dtype=torch.int32)
    manager._get_shard_expert_map.return_value = shard_map
    original_expert_ids = torch.arange(NUM_EXPERTS, dtype=torch.int32)
    original_local_indices = [0, 1, 2, 3]
    dispatcher = SimpleNamespace(
        ep_rank=1,
        num_experts_local=2,
        num_local_experts=2,
        expert_ids_per_ep_rank=original_expert_ids,
        local_expert_indices=original_local_indices,
    )
    comm_method = Mock(token_dispatcher=dispatcher)
    state_during_build = {}

    def capture_state(**kwargs):
        state_during_build.update(
            layer_count=layer.local_num_experts,
            dispatcher_count=dispatcher.num_local_experts,
            local_indices=dispatcher.local_expert_indices,
            expert_ids=dispatcher.expert_ids_per_ep_rank.clone(),
        )
        return object()

    result = _run_apply(
        method,
        layer,
        manager,
        comm_method,
        MoECommType.ALLTOALL,
        num_tokens=2,
        build_side_effect=capture_state,
    )

    manager.update_weights.assert_not_called()
    manager.update_weights_multi_card.assert_called_once()
    manager.trigger_next_layer_prefetch.assert_not_called()
    assert state_during_build["layer_count"] == manager.mc_shard_size
    assert state_during_build["dispatcher_count"] == manager.mc_shard_size
    assert state_during_build["local_indices"] == [4, 5, 6, 7]
    assert torch.equal(state_during_build["expert_ids"], torch.tensor([0, 1, 2, 3, 0, 1, 2, 3]))

    assert layer.local_num_experts == 4
    assert layer.moe_config.num_local_experts == 4
    assert dispatcher.num_experts_local == 2
    assert dispatcher.num_local_experts == 2
    assert dispatcher.local_expert_indices is original_local_indices
    assert dispatcher.expert_ids_per_ep_rank is original_expert_ids

    build_kwargs = result.build_input.call_args.kwargs
    assert build_kwargs["w1"] is manager._prefill_w13[0]
    assert build_kwargs["w2"] is manager._prefill_w2[0]
    assert build_kwargs["expert_map"] is shard_map
    assert build_kwargs["log2phy"] is None


def test_prefill_restores_state_when_fused_experts_raises():
    method = _make_method()
    layer = _make_layer(multi_card=True)
    manager = _make_manager(layer)
    manager._get_shard_expert_map.return_value = torch.arange(NUM_EXPERTS, dtype=torch.int32)
    original_expert_ids = torch.arange(NUM_EXPERTS, dtype=torch.int32)
    original_local_indices = [0, 1]
    dispatcher = SimpleNamespace(
        ep_rank=0,
        num_experts_local=2,
        num_local_experts=2,
        expert_ids_per_ep_rank=original_expert_ids,
        local_expert_indices=original_local_indices,
    )
    comm_method = Mock(token_dispatcher=dispatcher)
    comm_method.fused_experts.side_effect = RuntimeError("GMM failed")

    with pytest.raises(RuntimeError, match="GMM failed"):
        _run_apply(method, layer, manager, comm_method, MoECommType.ALLTOALL, num_tokens=2)

    assert layer.local_num_experts == 4
    assert layer.moe_config.num_local_experts == 4
    assert dispatcher.num_experts_local == 2
    assert dispatcher.num_local_experts == 2
    assert dispatcher.local_expert_indices is original_local_indices
    assert dispatcher.expert_ids_per_ep_rank is original_expert_ids
