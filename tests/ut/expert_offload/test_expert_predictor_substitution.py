"""Tests for the AI expert predictors (mode2_har / mode2_prevhfr).

mode2_prevhfr is the default trained head: LAYER_DELTA=1 buys it a full extra
layer of transfer lead over mode2_har. Expert substitution is on in the same
configuration and is covered here as a side feature. The ReMoE gate override is
off, so the router these tests score against is the model's own.
"""
import threading
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm_ascend.ascend_config import ExpertOffloadConfig
from vllm_ascend.expert_offload.expert_offload_manager import ExpertOffloadManager
from vllm_ascend.expert_predictor import (
    ExpertPredictorDriver,
    Mode2HarPredictor,
    Mode2PrevHfrPredictor,
    create_predictor,
    valid_predictor_names,
)
from vllm_ascend.ops.fused_moe.experts_selector import (
    substitute_experts,
    substitute_experts_device,
)


def _predictor(cls, *, num_moe_layers=43, num_hash_layers=3):
    """A finalized-looking predictor without a checkpoint on disk."""
    predictor = cls.__new__(cls)
    predictor.num_hash_layers = num_hash_layers
    predictor.num_heads = num_moe_layers - num_hash_layers
    predictor.layer_offset = num_hash_layers
    return predictor


def _driver(predictor, *, ready=True, dp_size=1, offload_threshold=6,
            multi_card=False):
    driver = ExpertPredictorDriver.__new__(ExpertPredictorDriver)
    driver.predictor = predictor
    driver.ready = ready
    driver._dp_size = dp_size
    driver.mgr = SimpleNamespace(offload_threshold=offload_threshold,
                                 enable_multi_card=multi_card)
    return driver


def _logits(rows):
    """Router logits whose softmax is exactly `rows`, so scores read directly."""
    return torch.log(torch.tensor(rows))


def _log2phy(num_experts, resident):
    log2phy = torch.full((num_experts,), -1, dtype=torch.int32)
    for slot, expert_id in enumerate(resident):
        log2phy[expert_id] = slot
    return log2phy


def _random_decode_step(seed, num_experts=256, topk=6, rows=3, resident=36):
    """One production-shaped decode step: 3 MoE rows (MTP k=2), 36/256 resident."""
    generator = torch.Generator().manual_seed(seed)
    router_logits = torch.randn(rows, num_experts, generator=generator) * 2.0
    bias = torch.randn(num_experts, generator=generator) * 0.05
    scores = torch.nn.functional.softplus(router_logits).sqrt() + bias
    topk_ids = scores.topk(topk, dim=-1).indices.to(torch.int32)
    permutation = torch.randperm(num_experts, generator=generator)[:resident]
    return router_logits, topk_ids, _log2phy(num_experts, permutation.tolist()), bias


# --------------------------------------------------------------------- #
#  AI predictors                                                          #
# --------------------------------------------------------------------- #

def test_registry_resolves_trained_names_and_keeps_fate_as_the_fallback():
    assert valid_predictor_names() == ["fate", "mode2_har", "mode2_prevhfr"]
    assert create_predictor("fate", None) is None
    with pytest.raises(ValueError, match="unknown expert predictor"):
        create_predictor("mode2_nope", None)


def test_prevhfr_carries_one_more_layer_of_lead_than_har():
    """LAYER_DELTA is the whole difference between the two heads: har predicts
    layer L from L's own input, prevhfr predicts L+1 from L's post-attention
    residual, so its transfer window gains GMM(L)."""
    assert Mode2HarPredictor.LAYER_DELTA == 0
    assert Mode2PrevHfrPredictor.LAYER_DELTA == 1


def test_head_index_leaves_hash_layers_and_the_shifted_leading_layer_to_fate():
    """A layer-shifted head saw a zero-filled predecessor at its first covered
    layer during training, so it must hand that target back — otherwise the
    layer loses prefetch instead of falling back to the exact tid2eid path."""
    har = _predictor(Mode2HarPredictor)
    prevhfr = _predictor(Mode2PrevHfrPredictor)

    assert [har.head_index(i) for i in range(5)] == [None, None, None, 0, 1]
    assert [prevhfr.head_index(i) for i in range(5)] == [None, None, None,
                                                         None, 1]
    assert har.head_index(42) == 39           # last MoE layer is covered
    assert har.head_index(43) is None         # past the end


def test_driver_covers_only_targets_its_head_owns_and_only_once_armed():
    prevhfr = _predictor(Mode2PrevHfrPredictor)

    armed = _driver(prevhfr)
    assert armed.covers(2) is False           # hash layer -> fate
    assert armed.covers(3) is False           # shifted leading layer -> fate
    assert armed.covers(4) is True

    assert _driver(prevhfr, ready=False).covers(4) is False


def test_decode_regime_scales_rows_by_dp_size_on_a_single_card():
    """AllGather in prepare() inflates the MoE row count by dp_size, so the
    driver must compare the inflated count against offload_threshold or it will
    predict for a step that takes the prefill pool."""
    driver = _driver(_predictor(Mode2PrevHfrPredictor), dp_size=4,
                     offload_threshold=6)
    with patch("vllm_ascend.expert_predictor._EXTRA_CTX",
               SimpleNamespace(max_tokens_across_dp=3)):
        assert driver._in_decode_regime() is False      # 3 x 4 = 12 > 6
    with patch("vllm_ascend.expert_predictor._EXTRA_CTX",
               SimpleNamespace(max_tokens_across_dp=1)):
        assert driver._in_decode_regime() is True       # 1 x 4 = 4 <= 6
    with patch("vllm_ascend.expert_predictor._EXTRA_CTX",
               SimpleNamespace(max_tokens_across_dp=None)):
        assert driver._in_decode_regime() is False      # no context -> no predict


def test_predictor_config_selects_prevhfr_with_a_two_row_two_expert_budget():
    """The offload half of the reference configuration: prevhfr predicting from
    2 of the 3 MTP rows, at most 2 prefetch transfers per layer, substitution on
    at the production threshold, single card, no instrumentation."""
    config = ExpertOffloadConfig({
        "expert_offload": True,
        "num_device_experts": 36,
        "cache_policy_enabled": True,
        "expert_prefetch_enabled": True,
        "expert_predictor": "mode2_prevhfr",
        "expert_predictor_ckpt": "/path/prevhfr.pt",
        "expert_prefetch_tokens": 2,
        "expert_prefetch_num": 1,
        "expert_substitution_enabled": True,
        "expert_substitution_threshold": 0.04,
        "enable_multi_card": False,
        "moe_offload_debug": False,
        "expert_prefetch_wait_timing": False,
    })

    assert config.expert_predictor == "mode2_prevhfr"
    assert config.expert_prefetch_tokens == 2    # This can be 2 or more, depending on MTP tokens
    assert config.expert_prefetch_num == 1       # This is set to 1 on A3, 2 on A5
    assert config.expert_substitution_enabled is True
    assert config.expert_substitution_threshold == 0.04
    assert config.enable_multi_card is False
    assert config.moe_offload_debug is False
    assert config.expert_prefetch_wait_timing is False


# --------------------------------------------------------------------- #
#  Expert substitution (side feature, on in the same configuration)       #
# --------------------------------------------------------------------- #

def test_device_substitution_matches_host_planner_at_production_threshold():
    """Release gate: at threshold 0.04 the NPU path must be bit-identical to
    plan_/commit_expert_substitutions, so no cache metric moves when it lands."""
    for seed in range(64):
        router_logits, topk_ids, log2phy, bias = _random_decode_step(seed)
        host = substitute_experts(
            router_logits, topk_ids, log2phy,
            expert_substitution_threshold=0.04,
            scoring_func="sqrtsoftplus", e_score_correction_bias=bias)
        device = substitute_experts_device(
            router_logits, topk_ids.clone(), log2phy,
            expert_substitution_threshold=0.04,
            scoring_func="sqrtsoftplus", e_score_correction_bias=bias)
        assert torch.equal(host, device), seed


def test_device_substitution_gives_the_scarce_candidate_to_the_weakest_reference():
    """SMoE substitutes the LEAST important expert first. topk_ids is ordered by
    descending score, so ranking misses by column index inverts that — with one
    candidate the strong reference would take it and the weak one would be
    dropped. Expect [[0, 2]]; the inverted ranking yields [[2, 1]]."""
    router_logits = _logits([[0.30, 0.26, 0.25, 0.10, 0.09]])
    topk_ids = torch.tensor([[0, 1]], dtype=torch.int32)

    out = substitute_experts_device(
        router_logits, topk_ids, _log2phy(5, [2]),
        expert_substitution_threshold=0.30)

    assert out.tolist() == [[0, 2]]


def test_device_substitution_keeps_a_whole_group_when_one_reference_is_confident():
    """Atomicity: expert 0 is confidently routed in row 1, so it must be paged in
    anyway and its row-0 reference must NOT be substituted. Expert 1 is a
    separate group and is still free to move."""
    router_logits = _logits([[0.30, 0.26, 0.25, 0.10, 0.09],
                             [0.60, 0.05, 0.20, 0.10, 0.05]])
    topk_ids = torch.tensor([[0, 1], [0, 2]], dtype=torch.int32)

    out = substitute_experts_device(
        router_logits, topk_ids, _log2phy(5, [2]),
        expert_substitution_threshold=0.30)

    assert out.tolist() == [[0, 2], [0, 2]]


def test_device_substitution_never_repeats_an_expert_within_a_row():
    """Two misses in one row must take two DIFFERENT candidates — both would
    otherwise argmax to the same highest-scoring in-band resident expert."""
    router_logits = _logits([[0.30, 0.26, 0.24, 0.20, 0.00]])
    topk_ids = torch.tensor([[0, 1]], dtype=torch.int32)

    out = substitute_experts_device(
        router_logits, topk_ids, _log2phy(5, [2, 3]),
        expert_substitution_threshold=0.30)

    assert out.tolist() == [[3, 2]]
    assert len(set(out[0].tolist())) == 2


def test_device_substitution_keeps_confident_misses_and_resident_routes():
    """A miss above the boundary band is important enough to page in; a fully
    resident row has nothing to substitute."""
    confident = _logits([[0.60, 0.05, 0.20, 0.10, 0.05]])
    topk_ids = torch.tensor([[0, 2]], dtype=torch.int32)
    assert substitute_experts_device(
        confident, topk_ids.clone(), _log2phy(5, [2, 3]),
        expert_substitution_threshold=0.02).tolist() == [[0, 2]]

    weak = _logits([[0.30, 0.26, 0.24, 0.20, 0.00]])
    resident_ids = torch.tensor([[0, 1]], dtype=torch.int32)
    assert substitute_experts_device(
        weak, resident_ids.clone(), _log2phy(5, [0, 1, 2, 3, 4]),
        expert_substitution_threshold=0.30).tolist() == [[0, 1]]


def test_update_weights_stages_ground_truth_before_substituting_in_place():
    """Wiring: the pre-substitution ids must reach topk_ids_gt_h BEFORE the
    device write-back, topk_ids must be mutated in place so the GMM sees the
    substituted routing, and the callback must get the 8-tuple form."""
    layer = SimpleNamespace(w13_weight=torch.zeros(2, 1))
    manager = ExpertOffloadManager.__new__(ExpertOffloadManager)
    manager.offload_config = ExpertOffloadConfig({
        "expert_substitution_enabled": True,
        "expert_substitution_threshold": 0.30,
    })
    manager.moe_layers = [layer]
    manager.topk = 2
    manager.num_total_experts = 5
    manager.offload_threshold = 4
    manager.cache_policy = None
    manager._stats = None
    manager._pf_wait_timing = False
    manager._prefetch_state_lock = threading.Lock()
    manager._prefetch_layer_npu_event = {}
    manager._finish_pending_predict = MagicMock()
    manager._finish_next_layer_predict = MagicMock()
    manager._update_weights_guarded = MagicMock()
    manager.topk_ids_h = torch.zeros((4, 2), dtype=torch.int32)
    manager.topk_ids_gt_h = torch.zeros((4, 2), dtype=torch.int32)
    manager.topk_weights_h = torch.zeros((4, 2))
    manager.log2phy_h = torch.zeros(5, dtype=torch.int32)
    manager.log2phy_np = manager.log2phy_h.numpy()

    topk_ids = torch.tensor([[0, 1]], dtype=torch.int32)
    with (
        patch(
            "vllm_ascend.expert_offload.expert_offload_manager.torch_npu",
            MagicMock(), create=True),
        patch(
            "vllm_ascend.expert_offload.expert_offload_manager"
            ".get_subscribed_compute_streams",
            return_value=set()),
        patch(
            "vllm_ascend.expert_offload.expert_offload_manager._EXTRA_CTX",
            SimpleNamespace(capturing=False)),
    ):
        manager.update_weights(
            layer, topk_ids, _log2phy(5, [2]),
            router_logits=_logits([[0.30, 0.26, 0.25, 0.10, 0.09]]),
            scoring_func="softmax")

    assert manager.topk_ids_gt_h[:1].tolist() == [[0, 1]]   # before mutation
    assert topk_ids.tolist() == [[0, 2]]                    # mutated in place
    assert manager.topk_ids_h[:1].tolist() == [[0, 2]]      # staged after
    args = manager._update_weights_guarded.call_args.args[0]
    assert len(args) == 8 and args[5] is False and args[6] is True
    assert torch.equal(args[7], torch.tensor([[0, 1]], dtype=torch.int32))


def test_update_weights_callback_reports_substitution_against_ground_truth():
    """The callback no longer substitutes; it only reads the two staged buffers.
    hit_pre must score the ORIGINAL routing against residency while hit_post
    scores the substituted routing, which is the whole point of the split."""
    record_layer = MagicMock()
    manager = ExpertOffloadManager.__new__(ExpertOffloadManager)
    manager.topk = 2
    manager._debug = False
    manager.cache_policy = None
    manager.load_stream = MagicMock()
    manager._synchronize_h2d = MagicMock()
    manager._stats = SimpleNamespace(collecting=True, record_layer=record_layer)
    manager._prefetch_state_lock = threading.Lock()
    manager._prefetch_stats_pending = {}
    manager._pf_wait_timing = False
    layer = SimpleNamespace(w13_weight=torch.zeros(2, 1))

    # experts 0 and 3 resident; the router originally picked {0, 1} and the
    # device substituted 1 -> 3, so every routed expert is now a hit.
    log2phy_np = torch.tensor([0, -1, -1, 1, -1], dtype=torch.int32).numpy()
    args = (
        torch.tensor([[0, 3]], dtype=torch.int32),   # topk_ids_h, post-subst
        log2phy_np,
        layer,
        0,
        None,
        False,
        True,
        torch.tensor([[0, 1]], dtype=torch.int32),   # topk_ids_gt_h, pre-subst
    )
    with patch(
        "vllm_ascend.expert_offload.expert_offload_manager.torch_npu.npu.stream",
        return_value=nullcontext(), create=True,
    ):
        manager._update_weights(args)

    kwargs = record_layer.call_args.kwargs
    assert kwargs["subst"] == 1.0
    assert kwargs["gsize"] == 2.0        # |G| is the ORIGINAL routed set
    assert kwargs["hit_post"] == 1.0     # {0, 3} both resident
    assert kwargs["hit_pre"] == 0.5      # {0, 1}: only 0 was resident
    assert kwargs["loads"] == 0.0