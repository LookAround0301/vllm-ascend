from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from vllm_ascend import ascend_forward_context as afc
from vllm_ascend.ascend_forward_context import MoECommType


@pytest.fixture(autouse=True)
def reset_mc2_tokens_capacity(monkeypatch):
    monkeypatch.setattr(afc, "_mc2_tokens_capacity", None)
    monkeypatch.setattr(
        afc,
        "get_ascend_config",
        lambda: SimpleNamespace(
            enable_prefill_mc2=False,
            enable_fused_mc2=0,
            expert_offload_config=SimpleNamespace(
                expert_offload=False,
                enable_multi_card=False,
            ),
        ),
    )


def _make_vllm_config(
    *,
    enable_expert_parallel: bool = True,
    world_size: int = 8,
    pipeline_parallel_size: int = 1,
    tensor_parallel_size: int = 1,
    num_experts: int = 128,
    quant_type: str | None = None,
    top_k_experts: int | None = 1,
    num_experts_per_tok: int | None = None,
    num_experts_per_token: int | None = None,
    cudagraph_capture_sizes: list[int] | None = None,
    max_cudagraph_capture_size: int = 0,
    max_num_batched_tokens: int = 0,
    hidden_size: int = 2048,
):
    hf_text_config_attrs: dict[str, object] = {}
    if top_k_experts is not None:
        hf_text_config_attrs["top_k_experts"] = top_k_experts
    if quant_type is not None:
        hf_text_config_attrs["quantize"] = quant_type
    if num_experts_per_tok is not None:
        hf_text_config_attrs["num_experts_per_tok"] = num_experts_per_tok
    if num_experts_per_token is not None:
        hf_text_config_attrs["num_experts_per_token"] = num_experts_per_token
    hf_text_config_attrs["hidden_size"] = hidden_size

    model_config = SimpleNamespace(
        hf_text_config=SimpleNamespace(**hf_text_config_attrs),
        get_num_experts=lambda: num_experts,
    )
    parallel_config = SimpleNamespace(
        enable_expert_parallel=enable_expert_parallel,
        world_size_across_dp=world_size,
        pipeline_parallel_size=pipeline_parallel_size,
        tensor_parallel_size=tensor_parallel_size,
    )
    compilation_config = SimpleNamespace(
        cudagraph_capture_sizes=cudagraph_capture_sizes or [],
        max_cudagraph_capture_size=max_cudagraph_capture_size,
    )
    scheduler_config = SimpleNamespace(max_num_batched_tokens=max_num_batched_tokens)
    return SimpleNamespace(
        model_config=model_config,
        parallel_config=parallel_config,
        compilation_config=compilation_config,
        scheduler_config=scheduler_config,
    )


def _patch_select_moe_comm_method_deps(
    monkeypatch,
    *,
    device_type,
    capacity: int = 128,
    ep_world_size: int = 8,
    enable_fused_mc2: int = 0,
    enable_prefill_mc2: int = 0,
    is_moe: bool = True,
):
    monkeypatch.setattr(afc, "is_moe_model", lambda _: is_moe)
    monkeypatch.setattr(afc, "get_mc2_tokens_capacity", lambda: capacity)
    monkeypatch.setattr(afc, "get_ascend_device_type", lambda: device_type)
    monkeypatch.setattr(afc, "get_ep_group", lambda: SimpleNamespace(world_size=ep_world_size))
    monkeypatch.setattr(
        afc,
        "get_ascend_config",
        lambda: SimpleNamespace(
            enable_fused_mc2=enable_fused_mc2,
            enable_prefill_mc2=enable_prefill_mc2,
            expert_offload_config=SimpleNamespace(
                expert_offload=False,
                enable_multi_card=False,
            ),
        ),
    )


@pytest.mark.parametrize(
    (
        "expert_offload",
        "enable_multi_card",
        "moe_comm_type",
        "expected",
    ),
    [
        (True, True, MoECommType.ALLTOALL, True),
        (True, True, MoECommType.MC2, False),
        (True, False, MoECommType.ALLTOALL, False),
        (False, True, MoECommType.ALLTOALL, False),
    ],
)
def test_should_skip_compiled_for_multi_card_offload(
    monkeypatch,
    expert_offload,
    enable_multi_card,
    moe_comm_type,
    expected,
):
    monkeypatch.setattr(
        afc,
        "get_ascend_config",
        lambda: SimpleNamespace(
            expert_offload_config=SimpleNamespace(
                expert_offload=expert_offload,
                enable_multi_card=enable_multi_card,
            )
        ),
    )

    assert afc._should_skip_compiled_for_multi_card_offload(moe_comm_type) is expected


def test_set_forward_context_skips_compiled_alltoall_for_multi_card_offload(
    monkeypatch,
):
    forward_context = SimpleNamespace(skip_compiled=False, dp_metadata=None)

    @contextmanager
    def fake_set_forward_context(**kwargs):
        forward_context.skip_compiled = kwargs["skip_compiled"]
        yield

    monkeypatch.setattr(afc, "set_forward_context", fake_set_forward_context)
    monkeypatch.setattr(afc, "get_forward_context", lambda: forward_context)
    monkeypatch.setattr(
        afc,
        "get_ascend_config",
        lambda: SimpleNamespace(
            expert_offload_config=SimpleNamespace(
                expert_offload=True,
                enable_multi_card=True,
            )
        ),
    )
    monkeypatch.setattr(
        afc,
        "select_moe_comm_method",
        lambda *_: MoECommType.ALLTOALL,
    )
    monkeypatch.setattr(afc, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(afc, "is_moe_model", lambda _: False)
    monkeypatch.setattr(afc, "enable_sp", lambda *_: False)
    monkeypatch.setattr(afc, "has_layer_idx", lambda _: False)
    monkeypatch.setattr(afc, "get_dp_group", lambda: SimpleNamespace(world_size=1))
    monkeypatch.setattr(afc, "get_mc2_mask", lambda: None)

    from vllm_ascend.ops.fused_moe import moe_comm_method

    monkeypatch.setattr(moe_comm_method, "get_moe_comm_method", lambda _: None)

    with afc.set_ascend_forward_context(
        attn_metadata=None,
        vllm_config=SimpleNamespace(),
        num_tokens=8192,
    ):
        assert forward_context.skip_compiled is True


def test_set_mc2_tokens_capacity_without_cudagraph_aligns_per_tp_rank():
    vllm_config = _make_vllm_config(tensor_parallel_size=6)

    afc.set_mc2_tokens_capacity(vllm_config, max_num_reqs=200, uniform_decode_query_len=3)

    assert afc.get_mc2_tokens_capacity() == 600


def test_set_mc2_tokens_capacity_with_cudagraph_uses_capture_size_and_aligns():
    vllm_config = _make_vllm_config(
        tensor_parallel_size=8,
        cudagraph_capture_sizes=[1, 2],
        max_cudagraph_capture_size=257,
    )

    afc.set_mc2_tokens_capacity(vllm_config, max_num_reqs=16, uniform_decode_query_len=1)

    assert afc.get_mc2_tokens_capacity() == 264


def test_set_mc2_tokens_capacity_prefill_mc2_uses_max_num_batched_tokens(monkeypatch):
    monkeypatch.setattr(
        afc,
        "get_ascend_config",
        lambda: SimpleNamespace(enable_prefill_mc2=True, enable_fused_mc2=0),
    )
    vllm_config = _make_vllm_config(tensor_parallel_size=8, max_num_batched_tokens=513)

    afc.set_mc2_tokens_capacity(vllm_config, max_num_reqs=16, uniform_decode_query_len=1)

    assert afc.get_mc2_tokens_capacity() == 520


def test_select_moe_comm_method_returns_none_for_non_moe(monkeypatch):
    _patch_select_moe_comm_method_deps(
        monkeypatch,
        device_type=afc.AscendDeviceType.A3,
        is_moe=False,
    )

    assert afc.select_moe_comm_method(16, _make_vllm_config()) is None


@pytest.mark.parametrize(
    ("enable_expert_parallel", "ep_world_size"),
    [
        (False, 8),
        (True, 1),
    ],
)
def test_select_moe_comm_method_uses_allgather_without_effective_expert_parallel(
    monkeypatch,
    enable_expert_parallel,
    ep_world_size,
):
    _patch_select_moe_comm_method_deps(
        monkeypatch,
        device_type=afc.AscendDeviceType.A3,
        ep_world_size=ep_world_size,
    )
    vllm_config = _make_vllm_config(enable_expert_parallel=enable_expert_parallel)

    assert afc.select_moe_comm_method(16, vllm_config) == MoECommType.ALLGATHER


@pytest.mark.parametrize(
    ("hf_config", "num_tokens", "expected"),
    [
        (SimpleNamespace(num_experts_per_tok=2), 2, MoECommType.MC2),
        (SimpleNamespace(num_experts_per_tok=2), 3, MoECommType.ALLTOALL),
        (
            SimpleNamespace(
                text_config=SimpleNamespace(num_experts_per_token=4)),
            1,
            MoECommType.MC2,
        ),
        (
            SimpleNamespace(
                text_config=SimpleNamespace(num_experts_per_token=4)),
            2,
            MoECommType.ALLTOALL,
        ),
    ],
)
def test_multi_card_offload_capacity_supports_deepseek_and_kimi_k3(
    monkeypatch,
    hf_config,
    num_tokens,
    expected,
):
    _patch_select_moe_comm_method_deps(
        monkeypatch,
        device_type=afc.AscendDeviceType.A3,
        capacity=128,
        ep_world_size=2,
    )
    monkeypatch.setattr(
        afc,
        "get_ascend_config",
        lambda: SimpleNamespace(
            expert_offload_config=SimpleNamespace(
                expert_offload=True,
                enable_multi_card=True,
                moe_offload_debug=True,
                shard_per_rank=False,
                num_device_experts_list=[8, 16],
                num_device_experts_for_rank=lambda _layer, ep_size: 8 // ep_size,
            ),
        ),
    )
    vllm_config = _make_vllm_config()
    vllm_config.model_config.hf_config = hf_config

    assert afc.select_moe_comm_method(num_tokens, vllm_config) == expected


@pytest.mark.parametrize(
    ("shard_per_rank", "num_tokens", "expected"),
    [
        (True, 1, MoECommType.MC2),
        (True, 2, MoECommType.MC2),
        (True, 3, MoECommType.ALLTOALL),
        (False, 2, MoECommType.MC2),
        (False, 3, MoECommType.MC2),
        (False, 5, MoECommType.ALLTOALL),
    ],
)
def test_multi_card_torch_offload_admission_uses_owner_capacity_when_sharded(
    monkeypatch,
    shard_per_rank,
    num_tokens,
    expected,
):
    _patch_select_moe_comm_method_deps(
        monkeypatch,
        device_type=afc.AscendDeviceType.A3,
        capacity=128,
        ep_world_size=2,
    )
    offload_config = SimpleNamespace(
        expert_offload=True,
        enable_multi_card=True,
        moe_offload_debug=True,
        shard_per_rank=shard_per_rank,
        h2d_backend="torch",
        num_device_experts_list=[8],
        num_device_experts_for_rank=lambda _layer, ep_size: 8 // ep_size,
    )
    monkeypatch.setattr(
        afc,
        "get_ascend_config",
        lambda: SimpleNamespace(expert_offload_config=offload_config),
    )
    vllm_config = _make_vllm_config()
    vllm_config.model_config.hf_config = SimpleNamespace(
        num_experts_per_tok=2)

    with patch.object(afc.logger, "info") as log_info:
        selected = afc.select_moe_comm_method(num_tokens, vllm_config)

    assert selected == expected
    admission = next(
        call for call in log_info.call_args_list
        if call.args[0].startswith("[MC_ADMISSION]"))
    assert admission.args[1] == (
        "sharded" if shard_per_rank else "replicated")
    assert admission.args[2:5] == (
        "torch", "owner" if shard_per_rank else "global", num_tokens)
    assert admission.args[5:11] == (
        2, 2, 8, 4 if shard_per_rank else 8,
        num_tokens * 2, 128)
    assert admission.args[12] == expected.name
    assert admission.args[13] == (
        "fits" if expected == MoECommType.MC2 else "expert_slots")


@pytest.mark.parametrize(
    ("h2d_backend", "is_draft_model", "expected", "admission_slots",
     "placement_scope"),
    [
        ("torch", False, MoECommType.ALLTOALL, 6, "owner"),
        ("memfabric", False, MoECommType.MC2, 48, "global"),
        ("memfabric", True, MoECommType.ALLTOALL, 6, "owner"),
    ],
)
def test_multi_card_memfabric_admission_uses_global_target_capacity(
    monkeypatch,
    h2d_backend,
    is_draft_model,
    expected,
    admission_slots,
    placement_scope,
):
    _patch_select_moe_comm_method_deps(
        monkeypatch,
        device_type=afc.AscendDeviceType.A3,
        capacity=128,
        ep_world_size=8,
    )
    offload_config = SimpleNamespace(
        expert_offload=True,
        enable_multi_card=True,
        moe_offload_debug=True,
        shard_per_rank=True,
        h2d_backend=h2d_backend,
        num_device_experts_list=[48],
        num_device_experts_for_rank=lambda _layer, ep_size: 48 // ep_size,
    )
    monkeypatch.setattr(
        afc,
        "get_ascend_config",
        lambda: SimpleNamespace(expert_offload_config=offload_config),
    )
    vllm_config = _make_vllm_config()
    vllm_config.model_config.hf_config = SimpleNamespace(
        num_experts_per_tok=6)

    with patch.object(afc.logger, "info") as log_info:
        selected = afc.select_moe_comm_method(
            3, vllm_config, is_draft_model=is_draft_model)

    assert selected == expected
    admission = next(
        call for call in log_info.call_args_list
        if call.args[0].startswith("[MC_ADMISSION]"))
    assert admission.args[1:11] == (
        "sharded", h2d_backend, placement_scope,
        3, 6, 8, 48, admission_slots, 18, 128)
    assert admission.args[12] == expected.name
    assert admission.args[13] == (
        "fits" if expected == MoECommType.MC2 else "expert_slots")


@pytest.mark.parametrize(
    ("num_tokens", "expected"),
    [
        (128, MoECommType.MC2),
        (129, MoECommType.ALLGATHER),
    ],
)
def test_select_moe_comm_method_a2_uses_mc2_within_capacity(monkeypatch, num_tokens, expected):
    _patch_select_moe_comm_method_deps(
        monkeypatch,
        device_type=afc.AscendDeviceType.A2,
        capacity=128,
        ep_world_size=16,
    )
    vllm_config = _make_vllm_config(world_size=16, num_experts=128)

    assert afc.select_moe_comm_method(num_tokens, vllm_config) == expected


def test_select_moe_comm_method_a2_uses_allgather_for_more_than_512_experts(monkeypatch):
    _patch_select_moe_comm_method_deps(
        monkeypatch,
        device_type=afc.AscendDeviceType.A2,
        capacity=128,
        ep_world_size=64,
    )
    vllm_config = _make_vllm_config(world_size=64, num_experts=896)

    assert afc.select_moe_comm_method(128, vllm_config) == MoECommType.ALLGATHER


@pytest.mark.parametrize(
    ("num_tokens", "ep_world_size", "expected"),
    [
        (128, 8, MoECommType.FUSED_MC2),
        (128, 128, MoECommType.MC2),
        (4097, 8, MoECommType.FUSED_MC2),
        (4097, 128, MoECommType.ALLTOALL),
    ],
)
def test_select_moe_comm_method_a3_enable_fused_mc2_mode_1(
    monkeypatch,
    num_tokens,
    ep_world_size,
    expected,
):
    _patch_select_moe_comm_method_deps(
        monkeypatch,
        device_type=afc.AscendDeviceType.A3,
        capacity=128,
        ep_world_size=ep_world_size,
        enable_fused_mc2=1,
    )

    vllm_config = _make_vllm_config(quant_type="w4a8")

    assert afc.select_moe_comm_method(num_tokens, vllm_config) == expected


@pytest.mark.parametrize(
    ("num_tokens", "expected"),
    [
        (128, MoECommType.MC2),
        (129, MoECommType.ALLTOALL),
    ],
)
def test_select_moe_comm_method_a3_without_fused_mc2(
    monkeypatch,
    num_tokens,
    expected,
):
    _patch_select_moe_comm_method_deps(
        monkeypatch,
        device_type=afc.AscendDeviceType.A3,
        capacity=128,
        enable_prefill_mc2=1,
    )
    vllm_config = _make_vllm_config()

    assert afc.select_moe_comm_method(num_tokens, vllm_config) == expected


@pytest.mark.parametrize(
    ("num_tokens", "ep_world_size", "expected"),
    [
        (128, 8, MoECommType.FUSED_MC2),
    ],
)
def test_select_moe_comm_method_a3_quant_w4a16(
    monkeypatch,
    num_tokens,
    ep_world_size,
    expected,
):
    _patch_select_moe_comm_method_deps(
        monkeypatch,
        device_type=afc.AscendDeviceType.A3,
        capacity=128,
        ep_world_size=ep_world_size,
        enable_fused_mc2=1,
        enable_prefill_mc2=1,
    )

    vllm_config = _make_vllm_config(quant_type="w4a16")

    assert afc.select_moe_comm_method(num_tokens, vllm_config) == expected


@pytest.mark.parametrize(
    ("num_tokens", "ep_world_size", "expected"),
    [
        (128, 8, MoECommType.FUSED_MC2),
    ],
)
def test_select_moe_comm_method_a3_quant_w4a8(
    monkeypatch,
    num_tokens,
    ep_world_size,
    expected,
):
    _patch_select_moe_comm_method_deps(
        monkeypatch,
        device_type=afc.AscendDeviceType.A3,
        capacity=128,
        ep_world_size=ep_world_size,
        enable_fused_mc2=1,
        enable_prefill_mc2=1,
    )

    vllm_config = _make_vllm_config(quant_type="w4a8")

    assert afc.select_moe_comm_method(num_tokens, vllm_config) == expected


@pytest.mark.parametrize(
    ("num_tokens", "ep_world_size", "expected"),
    [
        (128, 8, MoECommType.FUSED_MC2),
    ],
)
def test_select_moe_comm_method_a3_quant_w8a8(
    monkeypatch,
    num_tokens,
    ep_world_size,
    expected,
):
    _patch_select_moe_comm_method_deps(
        monkeypatch,
        device_type=afc.AscendDeviceType.A3,
        capacity=128,
        ep_world_size=ep_world_size,
        enable_fused_mc2=1,
        enable_prefill_mc2=1,
    )

    vllm_config = _make_vllm_config(quant_type="w8a8")

    assert afc.select_moe_comm_method(num_tokens, vllm_config) == expected


@pytest.mark.parametrize(
    ("quant_type", "expected"),
    [
        ("w4a8", True),
        ("w8a8", True),
        ("w8a16", False),
    ],
)
def test_cann_megamoe_supported_by_config_quant_type(
    quant_type,
    expected,
):
    vllm_config = _make_vllm_config(quant_type=quant_type)

    assert afc._cann_megamoe_supported_by_config(vllm_config) == expected


@pytest.mark.parametrize(
    ("num_tokens", "ep_world_size", "expected"),
    [
        (128, 8, MoECommType.FUSED_MC2),
    ],
)
def test_select_moe_comm_method_a3_mc2_invalid_hidden_size(
    monkeypatch,
    num_tokens,
    ep_world_size,
    expected,
):
    _patch_select_moe_comm_method_deps(
        monkeypatch,
        device_type=afc.AscendDeviceType.A3,
        capacity=128,
        ep_world_size=ep_world_size,
        enable_fused_mc2=1,
        enable_prefill_mc2=0,
    )

    vllm_config = _make_vllm_config(quant_type="w4a8", hidden_size=512)

    assert afc.select_moe_comm_method(num_tokens, vllm_config) == expected


@pytest.mark.parametrize(
    ("num_tokens", "world_size", "ep_world_size", "top_k_experts", "expected"),
    [
        (128, 4, 4, 2, MoECommType.MC2),
        (129, 2, 2, 4, MoECommType.ALLGATHER),
        (129, 8, 8, 4, MoECommType.ALLTOALL),
        (129, 32, 8, 16, MoECommType.ALLGATHER),
    ],
)
def test_select_moe_comm_method_a5(
    monkeypatch,
    num_tokens,
    world_size,
    ep_world_size,
    top_k_experts,
    expected,
):
    _patch_select_moe_comm_method_deps(
        monkeypatch,
        device_type=afc.AscendDeviceType.A5,
        capacity=128,
        ep_world_size=ep_world_size,
    )
    vllm_config = _make_vllm_config(world_size=world_size, top_k_experts=top_k_experts)

    assert afc.select_moe_comm_method(num_tokens, vllm_config) == expected


def test_select_moe_comm_method_a5_uses_kimi_k3_top_k(monkeypatch):
    _patch_select_moe_comm_method_deps(
        monkeypatch,
        device_type=afc.AscendDeviceType.A5,
        capacity=128,
        ep_world_size=8,
    )
    vllm_config = _make_vllm_config(
        world_size=32,
        top_k_experts=None,
        num_experts_per_token=16,
    )

    assert afc.select_moe_comm_method(129, vllm_config) == MoECommType.ALLGATHER


def test_select_moe_comm_method_310p_uses_allgather(monkeypatch):
    _patch_select_moe_comm_method_deps(
        monkeypatch,
        device_type=afc.AscendDeviceType._310P,
    )

    assert afc.select_moe_comm_method(128, _make_vllm_config()) == MoECommType.ALLGATHER
