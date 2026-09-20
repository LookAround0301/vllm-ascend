"""Exercise the actual pool allocator without importing vLLM or NPU packages.

The CPU regression verifies physical scale layout after a tensor copy from
contiguous host staging. NPU GMM behavior requires separate device validation.
"""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def load_pool_allocator():
    source = Path(__file__).resolve().parents[3] / "vllm_ascend/expert_offload/expert_offload_manager.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ExpertOffloadManager")
    methods = {"_init_prefill_pool_state", "_alloc_prefill_pool_slot"}
    cls.body = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in methods]
    module = ast.Module(body=[cls], type_ignores=[])
    namespace = {"torch": torch, "_expert_weight": getattr}
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)
    manager = namespace["ExpertOffloadManager"]()
    manager._init_prefill_pool_state()
    return manager


@pytest.mark.parametrize("transposed", [False, True])
def test_prefill_scale_pool_preserves_kernel_storage_order(transposed):
    manager = load_pool_allocator()
    scale = torch.arange(48, dtype=torch.uint8).reshape(2, 3, 8)
    if transposed:
        scale = scale.reshape(2, 3, 4, 2).transpose(-3, -2)
    layer = SimpleNamespace(
        w13_weight=torch.empty(2, 8, 3, dtype=torch.uint8),
        w2_weight=torch.empty(2, 3, 8, dtype=torch.uint8),
        w13_weight_scale=scale,
        w2_weight_scale=scale,
    )

    manager._alloc_prefill_pool_slot(layer, "cpu", torch.uint8, ntotal=5)

    for pool in (manager._prefill_w13_scale[0], manager._prefill_w2_scale[0]):
        assert pool.shape == (5, *scale.shape[1:])
        assert pool.dtype == scale.dtype
        # Copy to later expert slots to exercise the enlarged expert dimension.
        # Logical equality alone passes even with the old, incorrect layout.
        pool[3:5].copy_(scale.contiguous())
        assert torch.equal(pool[3:5], scale)
        expected = bytes(scale.untyped_storage())
        offset = 3 * scale.stride(0) * scale.element_size()
        assert bytes(pool.untyped_storage()[offset : offset + len(expected)]) == expected
        assert pool.stride() == scale.stride()
