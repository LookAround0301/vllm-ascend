import pytest
import torch

from vllm_ascend.ops.fused_moe.experts_selector import (
    dynamic_pruning_unsorted,
)


def test_pruning_sorts_routes_and_prunes_only_weak_cache_misses():
    topk_weights = torch.tensor(
        [[0.08, 0.62, 0.20, 0.10]],
        dtype=torch.float32,
    )
    topk_ids = torch.tensor(
        [[3, 0, 2, 1]],
        dtype=torch.int32,
    )
    # Expert 0 and 2 are resident; expert 1 and 3 are misses.
    log2phy = torch.tensor(
        [0, -1, 1, -1],
        dtype=torch.int32,
    )
    thresholds = torch.tensor(
        [0.0, 0.15, 0.15, 0.09],
        dtype=torch.float32,
    )

    weights, ids, debug = dynamic_pruning_unsorted(
        topk_weights,
        topk_ids,
        thresholds,
        log2phy=log2phy,
    )

    assert ids.dtype == torch.int32
    assert ids.tolist() == [[0, 2, -1, -1]]
    assert torch.allclose(
        weights,
        torch.tensor([[0.62, 0.20, 0.0, 0.0]]),
    )
    assert debug is None


def test_pruning_never_removes_a_resident_expert():
    topk_weights = torch.tensor([[0.70, 0.20, 0.10]])
    topk_ids = torch.tensor([[0, 1, 2]], dtype=torch.int32)
    log2phy = torch.tensor([0, 1, -1], dtype=torch.int32)
    thresholds = torch.tensor([0.0, 0.30, 0.15])

    weights, ids, _ = dynamic_pruning_unsorted(
        topk_weights,
        topk_ids,
        thresholds,
        log2phy=log2phy,
    )

    # Expert 1 is weak, but it is resident, so it must remain.
    # Expert 2 is both weak and missing, so it is pruned.
    assert ids.tolist() == [[0, 1, -1]]
    assert torch.allclose(
        weights,
        torch.tensor([[0.70, 0.20, 0.0]]),
    )


def test_pruning_without_log2phy_prunes_all_weak_routes():
    topk_weights = torch.tensor([[0.70, 0.20, 0.10]])
    topk_ids = torch.tensor([[0, 1, 2]], dtype=torch.int32)
    thresholds = torch.tensor([0.0, 0.30, 0.15])

    weights, ids, _ = dynamic_pruning_unsorted(
        topk_weights,
        topk_ids,
        thresholds,
        log2phy=None,
    )

    assert ids.tolist() == [[0, -1, -1]]
    assert torch.allclose(
        weights,
        torch.tensor([[0.70, 0.0, 0.0]]),
    )


def test_pruning_uses_strict_threshold_boundary():
    topk_weights = torch.tensor([[0.70, 0.20, 0.10]])
    topk_ids = torch.tensor([[0, 1, 2]], dtype=torch.int32)
    log2phy = torch.tensor([0, -1, -1], dtype=torch.int32)
    thresholds = torch.tensor([0.0, 0.20, 0.10])

    weights, ids, _ = dynamic_pruning_unsorted(
        topk_weights,
        topk_ids,
        thresholds,
        log2phy=log2phy,
    )

    # Implementation uses "<", not "<=".
    assert torch.equal(ids, topk_ids)
    assert torch.allclose(weights, topk_weights)


def test_pruning_debug_tensor_reports_miss_pruned_and_remaining():
    topk_weights = torch.tensor([[0.70, 0.20, 0.10]])
    topk_ids = torch.tensor([[0, 1, 2]], dtype=torch.int32)
    log2phy = torch.tensor([0, -1, -1], dtype=torch.int32)
    thresholds = torch.tensor([0.0, 0.15, 0.15])

    weights, ids, debug = dynamic_pruning_unsorted(
        topk_weights,
        topk_ids,
        thresholds,
        log2phy=log2phy,
        debug=True,
    )

    assert ids.tolist() == [[0, 1, -1]]
    assert torch.allclose(
        weights,
        torch.tensor([[0.70, 0.20, 0.0]]),
    )

    assert debug is not None
    assert debug.shape == (3, 1, 3)

    # Before pruning: experts 1 and 2 were cache misses.
    assert debug[0].tolist() == [[-1, 1, 2]]
    # Only expert 2 was pruned.
    assert debug[1].tolist() == [[-1, -1, 2]]
    # Expert 1 remains and still requires H2D.
    assert debug[2].tolist() == [[-1, 1, -1]]


def test_pruning_rejects_threshold_width_mismatch():
    topk_weights = torch.tensor([[0.70, 0.20, 0.10]])
    topk_ids = torch.tensor([[0, 1, 2]], dtype=torch.int32)
    log2phy = torch.tensor([0, -1, -1], dtype=torch.int32)

    with pytest.raises(
        ValueError,
        match="experts_pruning_threshold length",
    ):
        dynamic_pruning_unsorted(
            topk_weights,
            topk_ids,
            torch.tensor([0.0, 0.20]),
            log2phy=log2phy,
        )


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_pruning_preserves_expert_id_dtype(dtype):
    topk_weights = torch.tensor([[0.80, 0.20]])
    topk_ids = torch.tensor([[0, 1]], dtype=dtype)
    log2phy = torch.tensor([0, -1], dtype=torch.int32)

    _, ids, _ = dynamic_pruning_unsorted(
        topk_weights,
        topk_ids,
        torch.tensor([0.0, 0.30]),
        log2phy=log2phy,
    )

    assert ids.dtype == dtype
    assert ids.tolist() == [[0, -1]]