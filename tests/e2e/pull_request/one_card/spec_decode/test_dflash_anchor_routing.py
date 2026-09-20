#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
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
# This file is a part of the vllm-ascend project.
#
"""Real-NPU exact routing and changing-input graph regression tests.

Run `pytest tests/e2e/pull_request/one_card/spec_decode/test_dflash_anchor_routing.py`.
"""

import json
import subprocess
from pathlib import Path

import pytest
import torch
import torch_npu  # noqa: F401

import vllm_ascend.ops  # noqa: F401 - Register dependencies before DeviceOperator import.
from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.dflash_topm_state import DFlashTopMState
from vllm_ascend.ops.triton.dflash_fused_mask_native import allocate_workspace, topm_route
from vllm_ascend.utils import AscendDeviceType, enable_custom_op, get_ascend_device_type

TOP_K = 8
REQUEST_ROWS = 8
PROTECTED_ROWS = 3
SUFFIX_POOL_TOP_K = 2


def native_route(logits):
    weights, ids, _ = DeviceOperator.moe_gating_top_k(
        logits,
        k=TOP_K,
        k_group=1,
        group_count=1,
        group_select_mode=1,
        renorm=1,
        norm_type=0,
        out_flag=False,
        routed_scaling_factor=1.0,
        eps=1e-20,
        bias_opt=None,
    )
    return weights, ids


def reference(logits, roles):
    expert_count = logits.shape[1]
    top_ids = torch.topk(logits, TOP_K, dim=-1).indices
    pool = torch.zeros(expert_count, dtype=torch.bool, device=logits.device)
    pool[top_ids[roles == 1].flatten()] = True
    pool[top_ids[roles == 2, :SUFFIX_POOL_TOP_K].flatten()] = True
    restricted = torch.where((roles[:, None] == 2) & ~pool[None, :], torch.finfo(logits.dtype).min, logits)
    return native_route(restricted)


@pytest.fixture(autouse=True)
def npu_device():
    if not torch.npu.is_available():
        pytest.skip("Ascend NPU required")
    torch.npu.set_device(0)
    if get_ascend_device_type() == AscendDeviceType.A5:
        # enable_custom_op() is force-disabled (utils.py FIXME #7157)
        # but the C extension imports fine on A5 — register its ops
        # directly instead of asserting the A3-only gate.
        import vllm_ascend.vllm_ascend_C  # noqa: F401
    else:
        assert enable_custom_op()


@pytest.mark.parametrize("batch", [1, 2, 3, 4])
@pytest.mark.parametrize("expert_count", [128, 256])
@pytest.mark.parametrize("case", ["random", "ties", "zeros", "extreme", "tiny", "strided"])
def test_exact_native_equivalence(batch, expert_count, case):
    torch.manual_seed(20260910)
    rows = batch * REQUEST_ROWS
    logits = torch.randn(rows, expert_count, dtype=torch.bfloat16, device="npu")
    if case == "ties":
        logits.copy_((torch.arange(expert_count, device="npu") % 9).expand_as(logits))
    elif case == "zeros":
        logits.zero_()
    elif case == "extreme":
        logits.mul_(100)
    elif case == "tiny":
        logits.mul_(1e-8)
    elif case == "strided":
        storage = torch.empty(rows, expert_count * 2, dtype=logits.dtype, device="npu")
        storage[:, ::2].copy_(logits)
        logits = storage[:, ::2]
    roles = torch.where(torch.arange(rows, device="npu") % REQUEST_ROWS < PROTECTED_ROWS, 1, 2).int()
    expected_weights, expected_ids = reference(logits, roles)
    weights, ids = topm_route(logits, roles, 32)
    torch.npu.synchronize()
    assert torch.equal(ids, expected_ids)
    assert torch.equal(weights, expected_weights)


@pytest.mark.parametrize("batch", [1, 2, 3, 4])
@pytest.mark.parametrize("expert_count", [128, 256])
def test_graph_replay_updates_roles_and_logits(batch, expert_count):
    rows = batch * REQUEST_ROWS
    logits = torch.randn(rows, expert_count, dtype=torch.bfloat16, device="npu")
    roles = torch.where(torch.arange(rows, device="npu") % REQUEST_ROWS < PROTECTED_ROWS, 1, 2).int()
    workspace = allocate_workspace(logits, 32)
    for _ in range(3):
        topm_route(logits, roles, 32, workspace)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        weights, ids = topm_route(logits, roles, 32, workspace)
    for active_rows in (rows, 8, 0):
        logits.copy_(torch.randn_like(logits))
        roles.zero_()
        if active_rows:
            roles[:active_rows] = 2
            roles[:PROTECTED_ROWS] = 1
        graph.replay()
        expected_weights, expected_ids = reference(logits, roles)
        torch.npu.synchronize()
        assert torch.equal(ids, expected_ids)
        assert torch.equal(weights, expected_weights)


def test_reject_unsupported_dtype():
    logits = torch.zeros(REQUEST_ROWS, 256, device="npu")
    roles = torch.ones(REQUEST_ROWS, dtype=torch.int32, device="npu")
    with pytest.raises(ValueError, match="BF16"):
        topm_route(logits, roles, 32)


def make_dspark_state(tmp_path):
    config_path = tmp_path / "dspark_activation_routing.json"
    config_path.write_text(
        json.dumps(
            {
                "backend": "anchor_union_fused_native",
                "verify_block_size": 8,
                "num_experts": 256,
                "route_top_k": 6,
                "scoring_func": "sqrtsoftplus",
                "protected_rows": 3,
                "suffix_pool_top_k": 2,
                "expected_router_layers": 40,
                "fused_rows": [8, 512],
            }
        )
    )
    return DFlashTopMState(config_path)


def dspark_expected(logits, bias, roles):
    raw_scores = torch.nn.functional.softplus(logits).sqrt()
    selection_scores = raw_scores + bias
    initial_ids = selection_scores.topk(6, dim=-1).indices
    limits = torch.where(roles == 1, 6, torch.where(roles == 2, 2, 0))
    active = torch.arange(6, device="npu").unsqueeze(0) < limits.unsqueeze(1)
    flags = torch.zeros_like(selection_scores, dtype=torch.bool)
    flags.scatter_(1, initial_ids, active)
    pool = flags.any(dim=0)
    restricted = selection_scores.masked_fill(
        (roles[:, None] == 2) & ~pool[None, :],
        torch.finfo(selection_scores.dtype).min,
    )
    ids = restricted.topk(6, dim=-1).indices
    weights = raw_scores.gather(1, ids)
    weights /= weights.sum(dim=-1, keepdim=True)
    weights *= 1.5
    return weights, ids.to(torch.int32)


def _hash_op_available():
    """Return True when the aclnnMoeGatingTopKHash native oracle is usable.

    The DSpark fused-path oracles below call ``torch.ops._C_ascend.moe_gating_top_k_hash``,
    which needs the moe_gating_top_k_hash custom op package installed under
    ``vllm_ascend/_cann_ops_custom``. Environments without the package skip these tests
    instead of erroring; build it with ``csrc/build_aclnn.sh`` to enable them.
    """
    package_dir = Path(vllm_ascend.__file__).resolve().parent
    libs = sorted(package_dir.glob("_cann_ops_custom/vendors/*/op_api/lib/libcust_opapi.so"))
    for lib in libs:
        try:
            proc = subprocess.run(["nm", "-D", str(lib)], capture_output=True, text=True, timeout=30)
        except (OSError, subprocess.SubprocessError):
            return False
        if proc.returncode == 0 and "aclnnMoeGatingTopKHash" in proc.stdout:
            return True
    return False


def native_dspark_route(logits, bias):
    weights, ids, _ = torch.ops._C_ascend.moe_gating_top_k_hash(
        x=logits,
        k=6,
        bias=bias,
        input_ids=None,
        tid2eid=None,
        k_group=1,
        group_count=1,
        routed_scaling_factor=1.5,
        eps=1e-20,
        group_select_mode=1,
        renorm=0,
        norm_type=2,
        out_flag=False,
    )
    return weights, ids


@pytest.mark.skipif(
    not _hash_op_available(),
    reason="requires moe_gating_top_k_hash custom op package (build via csrc/build_aclnn.sh)",
)
@pytest.mark.parametrize("case", ["random", "ties", "zeros", "extreme"])
def test_dspark_top6_sqrtsoftplus_fused_equivalence(tmp_path, case):
    state = make_dspark_state(tmp_path)
    state.prepare(REQUEST_ROWS, [(0, REQUEST_ROWS)], "npu")

    torch.manual_seed(20260912)
    logits = torch.randn(REQUEST_ROWS, 256, dtype=torch.float32, device="npu")
    bias = torch.linspace(8.0, 9.0, 256, dtype=torch.float32, device="npu")
    if case == "ties":
        logits.zero_()
        bias.zero_()
    elif case == "zeros":
        logits.zero_()
    elif case == "extreme":
        logits.mul_(10)
    reference_weights, reference_ids = state.route_reference(
        logits,
        state.current_roles,
        top_k=6,
        renormalize=True,
        routed_scaling_factor=1.5,
        e_score_correction_bias=bias,
    )
    weights, ids = state.route(
        logits,
        state.current_roles,
        top_k=6,
        renormalize=True,
        scoring_func="sqrtsoftplus",
        routed_scaling_factor=1.5,
        e_score_correction_bias=bias,
    )

    expected_weights, expected_ids = dspark_expected(logits, bias, state.current_roles)
    native_weights, native_ids = native_dspark_route(logits, bias)

    torch.npu.synchronize()
    abs_error = (weights - reference_weights).abs()
    max_abs_error = abs_error.max().item()
    max_rel_error = (abs_error / reference_weights.abs().clamp_min(1e-12)).max().item()
    print(
        f"case={case} max_abs_error={max_abs_error:.9g} max_rel_error={max_rel_error:.9g} output_dtype={weights.dtype}"
    )
    assert max_abs_error <= 1e-6
    assert max_rel_error <= 1e-5
    assert torch.equal(ids, reference_ids)
    assert torch.equal(weights.bfloat16(), reference_weights.bfloat16())
    assert torch.equal(ids, expected_ids)
    assert torch.equal(weights.bfloat16(), expected_weights.bfloat16())
    assert torch.equal(ids[:PROTECTED_ROWS], native_ids[:PROTECTED_ROWS])
    # last-ulp fp32 divergence between fused and native paths
    assert torch.equal(weights[:PROTECTED_ROWS].bfloat16(), native_weights[:PROTECTED_ROWS].bfloat16())


@pytest.mark.skipif(
    not _hash_op_available(),
    reason="requires moe_gating_top_k_hash custom op package (build via csrc/build_aclnn.sh)",
)
def test_dspark_fused_512_graph_replay(tmp_path):
    graph_rows = 512
    state = make_dspark_state(tmp_path)
    state.prepare(graph_rows, [(0, REQUEST_ROWS)], "npu")
    logits = torch.randn(graph_rows, 256, dtype=torch.float32, device="npu")
    bias = torch.linspace(8.0, 9.0, 256, dtype=torch.float32, device="npu")

    for _ in range(3):
        state.route(
            logits,
            state.current_roles,
            6,
            True,
            "sqrtsoftplus",
            1.5,
            bias,
        )
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        weights, ids = state.route(
            logits,
            state.current_roles,
            top_k=6,
            renormalize=True,
            scoring_func="sqrtsoftplus",
            routed_scaling_factor=1.5,
            e_score_correction_bias=bias,
        )

    for segments in ([(0, REQUEST_ROWS)], [(REQUEST_ROWS, REQUEST_ROWS)]):
        logits.copy_(torch.randn_like(logits))
        state.prepare(graph_rows, segments, "npu")
        graph.replay()
        expected_weights, expected_ids = dspark_expected(logits, bias, state.current_roles)
        native_weights, native_ids = native_dspark_route(logits, bias)
        torch.npu.synchronize()
        abs_error = (weights - expected_weights).abs()
        max_abs_error = abs_error.max().item()
        max_rel_error = (abs_error / expected_weights.abs().clamp_min(1e-12)).max().item()
        print(
            f"segments={segments} max_abs_error={max_abs_error:.9g} "
            f"max_rel_error={max_rel_error:.9g} output_dtype={weights.dtype}"
        )
        assert max_abs_error <= 1e-6
        assert max_rel_error <= 1e-5
        assert torch.equal(ids, expected_ids)
        assert torch.equal(weights.bfloat16(), expected_weights.bfloat16())
        unrestricted = state.current_roles != 2
        assert torch.equal(ids[unrestricted], native_ids[unrestricted])
        assert torch.equal(weights[unrestricted].bfloat16(), native_weights[unrestricted].bfloat16())
