import json
import os
import threading
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import torch

from vllm_ascend.ascend_config import ExpertOffloadConfig
from vllm_ascend.expert_offload.expert_offload_manager import (
    ExpertOffloadManager,
    _expert_weight,
    _stable_int_checksum,
)
from vllm_ascend.expert_offload.h2d_transfer import H2DCopyTask


def _manager_for_prediction(next_layer, gate_weight, *, topk=2,
                            prefetch_topk=2, prefetch_tokens=1):
    manager = ExpertOffloadManager.__new__(ExpertOffloadManager)
    manager.moe_layers = [SimpleNamespace(), next_layer]
    manager._gate_weights_npu = [None, gate_weight]
    manager.topk = topk
    manager.prefetch_topk = prefetch_topk
    # predict_next_layer_experts_npu reads prefetch_tokens on both the
    # hash (tid2eid) branch and the learned-gate branch
    manager.prefetch_tokens = prefetch_tokens
    return manager


def _manager_for_mc_debug():
    manager = ExpertOffloadManager.__new__(ExpertOffloadManager)
    manager._debug = True
    manager._ep_size = 16
    manager._ep_rank = 8
    manager._ep_info_resolved = True
    manager._mc_debug_lock = threading.Lock()
    manager._mc_debug_schedule_seq = 0
    manager._mc_debug_callback_seq = 0
    manager._mc_debug_collective_seq = 0
    manager._mc_debug_active_callbacks = 0
    manager._mc_debug_layer_calls = {}
    return manager


def test_expert_weight_supports_deepseek_and_kimi_k3_layouts():
    deepseek_weight = object()
    kimi_weight = object()
    deepseek_layer = SimpleNamespace(w13_weight=deepseek_weight)
    kimi_layer = SimpleNamespace(
        experts=SimpleNamespace(w13_weight=kimi_weight))

    assert _expert_weight(deepseek_layer, "w13_weight") is deepseek_weight
    assert _expert_weight(kimi_layer, "w13_weight") is kimi_weight


def test_cpu_tensor_storage_bytes_counts_unique_nested_storages():
    weight = torch.empty(16, dtype=torch.int8)
    scale = torch.empty(4, dtype=torch.float32)

    total = ExpertOffloadManager._cpu_tensor_storage_bytes(
        ([weight, weight.view(4, 4)], {"scale": [[scale]]}))

    assert total == weight.untyped_storage().nbytes() + \
        scale.untyped_storage().nbytes()


def test_replicated_cpu_memory_log_reports_full_copy_per_rank():
    manager = ExpertOffloadManager.__new__(ExpertOffloadManager)
    manager._debug = True
    manager.offload_config = ExpertOffloadConfig({"shard_per_rank": False})
    manager._ep_size = 2
    manager._ep_rank = 1
    manager._ep_info_resolved = True
    manager.num_total_experts = 4
    manager.moe_layers = [object()]
    manager.w13_weights_cpu = [[torch.empty(2, dtype=torch.int8)
                                for _ in range(4)]]
    manager.w2_weights_cpu = [[torch.empty(1, dtype=torch.int8)
                               for _ in range(4)]]
    manager.scale_cpu_buffers = {}
    manager.offset_cpu_buffers = {}
    manager.scale_bias_cpu_buffers = {}
    manager._host_memory_snapshot = MagicMock(return_value=(1024, 2048, 4096))

    with patch(
        "vllm_ascend.expert_offload.expert_offload_manager.logger.info"
    ) as log_info:
        manager._log_cpu_expert_memory()

    args = log_info.call_args.args
    assert args[0].startswith("[CPU_MEM]")
    assert args[1:7] == (1, 2, os.getpid(), "replicated", 4, 4)
    assert args[-1] == 2


def test_multi_card_decode_plan_log_covers_sharded_baseline():
    manager = ExpertOffloadManager.__new__(ExpertOffloadManager)
    manager._debug = True
    manager.offload_config = ExpertOffloadConfig({"shard_per_rank": True})
    manager._ep_size = 2
    manager._ep_rank = 0
    manager._ep_info_resolved = True
    global_counts = torch.tensor([3, 0, 2, 1], dtype=torch.int64)
    placement = SimpleNamespace(
        log2phy=torch.tensor([0, -1, 2, 3], dtype=torch.int32),
        per_rank_experts=[[0], [2, 3]],
        per_rank_load=[3, 3],
        unassigned=[],
    )

    with patch(
        "vllm_ascend.expert_offload.expert_offload_manager.logger.info"
    ) as log_info:
        manager._log_mc_decode_plan(
            4, global_counts, placement, per_rank_slots=2,
            counts_ms=1.25, placement_ms=0.5)

    args = log_info.call_args.args
    assert "DECODE plan" in args[0]
    assert args[1:10] == (
        0, 4, "sharded", 6, 3, 4, [1, 2], [1, 2], True)
    assert args[10] == _stable_int_checksum(global_counts)
    assert args[11] == _stable_int_checksum(placement.log2phy)
    assert args[12:15] == (0, [], [3, 3])


def test_multi_card_lrc_log_exposes_cross_rank_checksums():
    manager = ExpertOffloadManager.__new__(ExpertOffloadManager)
    manager._debug = True
    manager._ep_size = 2
    manager._ep_rank = 1
    manager._ep_info_resolved = True
    manager._mc_lrc = SimpleNamespace(layer_states=[SimpleNamespace(
        step=3,
        freq=[0, 5, 1, 4],
        ema=torch.tensor([0.0, 1.75, 0.25, 2.0]),
        router_score=[0.0, 0.4, 0.8, 0.2],
    )])
    manager._DEBUG_EXPERT_SAMPLE_LIMIT = 4
    global_counts = torch.tensor([0, 2, 0, 4], dtype=torch.int64)
    global_router_score = torch.tensor([0.0, 0.4, 0.0, 0.2])
    hotness = torch.tensor([0.0, 5.0, 1.0, 4.0])

    with patch(
        "vllm_ascend.expert_offload.expert_offload_manager.logger.info"
    ) as log_info:
        manager._log_mc_lrc_state(
            0, global_counts, global_router_score, hotness)

    args = log_info.call_args.args
    assert args[0].startswith("[MC_LRC]")
    assert args[1:8] == (1, 2, 0, 3, 6, 2,
                         _stable_int_checksum(global_counts))
    assert args[-1][0]["expert"] == 1
    assert args[-1][0]["count"] == 2
    assert args[-1][0]["freq"] == 5


def test_mc_debug_callback_log_exposes_sequence_and_reentry():
    manager = _manager_for_mc_debug()

    with patch(
        "vllm_ascend.expert_offload.expert_offload_manager.logger.info"
    ) as log_info:
        first = manager._begin_mc_debug_callback(39, False, True)
        second = manager._begin_mc_debug_callback(40, False, True)
        manager._end_mc_debug_callback(second, "ok")
        manager._end_mc_debug_callback(first, "ok")

    assert first["callback_seq"] == 1
    assert first["layer_call"] == 1
    assert second["callback_seq"] == 2
    assert manager._mc_debug_active_callbacks == 0
    events = [entry.args[1] for entry in log_info.call_args_list]
    assert events == ["CB_ENTER", "CB_ENTER", "CB_EXIT", "CB_EXIT"]
    assert "active_callbacks=2" in log_info.call_args_list[1].args[-1]


def test_mc_debug_gloo_log_pairs_collective_sequences():
    manager = _manager_for_mc_debug()
    context = manager._begin_mc_debug_callback(39, False, True)
    cpu_group = SimpleNamespace(group_name="ep_offload")
    local_values = torch.tensor([1, 2], dtype=torch.int64)

    with patch(
        "vllm_ascend.expert_offload.multi_card_planner."
        "gather_global_counts_cpu",
        side_effect=lambda values, group: values + 1,
    ), patch(
        "vllm_ascend.expert_offload.expert_offload_manager.logger.info"
    ) as log_info:
        counts = manager._gather_cpu_with_mc_debug(
            local_values, cpu_group, "count", context)
        scores = manager._gather_cpu_with_mc_debug(
            local_values.float(), cpu_group, "router_score", context)

    assert torch.equal(counts, torch.tensor([2, 3]))
    assert torch.equal(scores, torch.tensor([2.0, 3.0]))
    events = [entry.args[1] for entry in log_info.call_args_list]
    assert events == ["GLOO_ENTER", "GLOO_EXIT",
                      "GLOO_ENTER", "GLOO_EXIT"]
    assert "kind=count" in log_info.call_args_list[0].args[-1]
    assert "collective_seq=1" in log_info.call_args_list[0].args[-1]
    assert "kind=router_score" in log_info.call_args_list[2].args[-1]
    assert "collective_seq=2" in log_info.call_args_list[2].args[-1]


def test_mc_debug_callback_logs_python_error_and_exit():
    manager = _manager_for_mc_debug()
    manager._update_weights_multi_card_impl = MagicMock(
        side_effect=RuntimeError("debug failure"))
    args = (
        None, None, None, 39, 16, False, None, False, None, "softmax",
        None, None, True,
    )

    with patch(
        "vllm_ascend.expert_offload.expert_offload_manager.logger.info"
    ) as log_info:
        try:
            manager._update_weights_multi_card(args)
        except RuntimeError as exc:
            assert str(exc) == "debug failure"
        else:
            raise AssertionError("expected RuntimeError")

    events = [entry.args[1] for entry in log_info.call_args_list]
    assert events == ["CB_ENTER", "CB_ERROR", "CB_EXIT"]
    assert "status=error:RuntimeError" in log_info.call_args_list[-1].args[-1]
    assert manager._mc_debug_active_callbacks == 0


def test_mc_debug_disabled_uses_uninstrumented_fast_path():
    manager = ExpertOffloadManager.__new__(ExpertOffloadManager)
    manager._debug = False
    manager._update_weights_multi_card_impl = MagicMock(return_value="done")
    args = (
        None, None, None, 39, 16, False, None, False, None, "softmax",
        None, None,
    )

    with patch(
        "vllm_ascend.expert_offload.expert_offload_manager.logger.info"
    ) as log_info:
        result = manager._update_weights_multi_card(args)

    assert result == "done"
    manager._update_weights_multi_card_impl.assert_called_once_with(args)
    log_info.assert_not_called()


def test_multi_card_decode_plan_log_is_emitted_only_by_rank_zero():
    manager = ExpertOffloadManager.__new__(ExpertOffloadManager)
    manager._debug = True
    manager._ep_rank = 1

    with patch(
        "vllm_ascend.expert_offload.expert_offload_manager.logger.info"
    ) as log_info:
        manager._log_mc_decode_plan(0, None, None, 1, 0.0, 0.0)

    log_info.assert_not_called()


def test_multi_card_decode_plan_marks_unassigned_as_capacity_failure():
    manager = ExpertOffloadManager.__new__(ExpertOffloadManager)
    manager._debug = True
    manager.offload_config = ExpertOffloadConfig({"shard_per_rank": False})
    manager._ep_size = 2
    manager._ep_rank = 0
    manager._ep_info_resolved = True
    global_counts = torch.tensor([1, 1, 1, 1, 1], dtype=torch.int64)
    placement = SimpleNamespace(
        log2phy=torch.tensor([0, 1, 2, 3, -1], dtype=torch.int32),
        per_rank_experts=[[0, 1], [2, 3]],
        per_rank_load=[2, 2],
        unassigned=[4],
    )

    with patch(
        "vllm_ascend.expert_offload.expert_offload_manager.logger.info"
    ) as log_info:
        manager._log_mc_decode_plan(
            0, global_counts, placement, per_rank_slots=2,
            counts_ms=0.0, placement_ms=0.0)

    assert log_info.call_args.args[9] is False
    assert log_info.call_args.args[12] == 1


def test_multi_card_decode_cache_log_validates_replicated_placement():
    manager = ExpertOffloadManager.__new__(ExpertOffloadManager)
    manager._debug = True
    manager.offload_config = ExpertOffloadConfig({"shard_per_rank": False})
    manager._ep_size = 2
    manager._ep_rank = 1
    manager._ep_info_resolved = True
    log2phy = torch.tensor([-1, -1, 5, -1, -1, 3], dtype=torch.int32)

    with patch(
        "vllm_ascend.expert_offload.expert_offload_manager.logger.info"
    ) as log_info:
        manager._log_mc_decode_cache(
            layer_idx=7,
            my_experts=[5, -1, 2],
            hits=[(0, 5)],
            misses=[(2, 2)],
            resident_map={0: 5, 2: 2},
            log2phy=log2phy,
            per_rank_slots=3,
            h2d_ms=2.0,
            total_ms=3.0,
        )

    args = log_info.call_args.args
    assert "DECODE cache" in args[0]
    assert args[1:10] == (
        1, 7, "replicated", 2, 1, 1, 0.5, True, True)
    assert args[10:14] == ([(2, 2)], False, 2.0, 3.0)


def test_multi_card_decode_cache_log_skips_healthy_hit_only_step():
    manager = ExpertOffloadManager.__new__(ExpertOffloadManager)
    manager._debug = True
    manager.offload_config = ExpertOffloadConfig({"shard_per_rank": False})
    manager._ep_size = 2
    manager._ep_rank = 0
    manager._ep_info_resolved = True

    with patch(
        "vllm_ascend.expert_offload.expert_offload_manager.logger.info"
    ) as log_info:
        manager._log_mc_decode_cache(
            layer_idx=1,
            my_experts=[3],
            hits=[(0, 3)],
            misses=[],
            resident_map={0: 3},
            log2phy=torch.tensor([-1, -1, -1, 0], dtype=torch.int32),
            per_rank_slots=1,
        )

    log_info.assert_not_called()


def test_weight_load_progress_log_is_rate_limited():
    manager = ExpertOffloadManager.__new__(ExpertOffloadManager)
    manager._load_futures = [MagicMock() for _ in range(128)]
    manager._load_phase_start = 1.0
    manager._drained_shards = 0

    with patch(
        "vllm_ascend.expert_offload.expert_offload_manager.logger.info"
    ) as log_info:
        manager._drain_futures()
        log_info.assert_not_called()

        manager._load_futures = [MagicMock() for _ in range(128)]
        manager._drained_shards = manager._LOAD_PROGRESS_LOG_EVERY - 128
        with patch(
            "vllm_ascend.expert_offload.expert_offload_manager.time.perf_counter",
            return_value=2.0,
        ):
            manager._drain_futures()

    assert "weight load progress" in log_info.call_args.args[0]
    assert log_info.call_args.args[1] == manager._LOAD_PROGRESS_LOG_EVERY


def test_gate_registration_is_model_agnostic_and_index_aligned():
    deepseek_weight = torch.tensor([[1.0, 2.0]], dtype=torch.float16)
    kimi_weight = torch.tensor([[3.0, 4.0]], dtype=torch.float16)
    manager = ExpertOffloadManager.__new__(ExpertOffloadManager)
    manager.moe_layers = [
        SimpleNamespace(gate=SimpleNamespace(weight=deepseek_weight)),
        SimpleNamespace(gate=SimpleNamespace(weight=kimi_weight)),
        SimpleNamespace(),
    ]
    manager._gate_weights_npu = [torch.zeros(1)]

    manager.register_gate_weights(None)

    assert len(manager._gate_weights_npu) == 3
    assert torch.equal(manager._gate_weights_npu[0], deepseek_weight.float())
    assert torch.equal(manager._gate_weights_npu[1], kimi_weight.float())
    assert manager._gate_weights_npu[2] is None


def test_post_finalize_gate_registration_keeps_missing_gate_slot():
    manager = ExpertOffloadManager.__new__(ExpertOffloadManager)
    manager.offload_config = ExpertOffloadConfig({
        "expert_prefetch_enabled": True
    })
    manager.topk_ids_h = object()
    manager._gate_weights_npu = []
    manager.moe_layers = [SimpleNamespace()]

    manager._register_layer_gate(manager.moe_layers[0])

    assert manager._gate_weights_npu == [None]


def test_deepseek_hash_router_prefetch_uses_tid2eid():
    tid2eid = torch.tensor([[0, 1], [2, 3], [3, 1]], dtype=torch.int32)
    next_layer = SimpleNamespace(gate=SimpleNamespace(tid2eid=tid2eid))
    manager = _manager_for_prediction(
        next_layer, torch.ones((4, 3)), topk=2, prefetch_topk=1)

    with patch(
        "vllm_ascend.expert_offload.expert_offload_manager.get_forward_context",
        return_value=SimpleNamespace(input_ids=torch.tensor([2])),
    ):
        weights, expert_ids = manager.predict_next_layer_experts_npu(
            0, torch.tensor([[100.0, -100.0, 0.0]]))

    assert torch.equal(expert_ids, tid2eid[2:3])
    assert torch.equal(weights, torch.full((1, 2), 0.5))


def test_kimi_k3_learned_router_prefetch_keeps_full_topk_candidates():
    next_layer = SimpleNamespace(gate=SimpleNamespace())
    gate_weight = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    manager = _manager_for_prediction(
        next_layer, gate_weight, topk=2, prefetch_topk=1)

    weights, expert_ids = manager.predict_next_layer_experts_npu(
        0, torch.tensor([[1.0, 2.0]]))

    expected_weights = torch.softmax(torch.tensor([[1.0, 2.0, 3.0]]),
                                     dim=-1).topk(2, dim=-1).values
    assert torch.equal(expert_ids, torch.tensor([[2, 1]]))
    assert torch.allclose(weights, expected_weights)


def test_expert_substitution_log_is_controlled_by_debug():
    manager = ExpertOffloadManager.__new__(ExpertOffloadManager)
    manager.offload_config = ExpertOffloadConfig({
        "expert_substitution_enabled": True,
        "expert_substitution_threshold": 0.25,
    })
    original_ids = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)
    substituted_ids = torch.tensor([[1, 5], [6, 4]], dtype=torch.int32)

    with patch(
        "vllm_ascend.expert_offload.expert_offload_manager.logger.info"
    ) as log_info:
        manager._debug = False
        manager._log_expert_substitution(
            7, original_ids, substituted_ids)
        log_info.assert_not_called()

        manager._debug = True
        manager._log_expert_substitution(
            7, original_ids, substituted_ids)

    log_args = log_info.call_args.args
    assert log_args[1:4] == (7, 2, 0.25)
    assert log_args[4] == [
        {"token": 0, "position": 1, "original": 2, "substitute": 5},
        {"token": 1, "position": 0, "original": 3, "substitute": 6},
    ]
    assert log_args[5] == 0


def test_multi_card_substitution_updates_only_active_rows():
    manager = ExpertOffloadManager.__new__(ExpertOffloadManager)
    manager.offload_config = ExpertOffloadConfig({
        "expert_substitution_enabled": True,
        "expert_substitution_threshold": 0.30,
    })
    manager.topk = 2
    manager._debug = False
    router_logits_h = torch.log(torch.tensor([
        [0.40, 0.25, 0.20, 0.15],
        [0.40, 0.25, 0.20, 0.15],
    ]))
    topk_ids_h = torch.tensor([[0, 1], [0, 1]], dtype=torch.int32)
    # The full global placement says experts 0 and 3 are resident across EP.
    log2phy_h = torch.tensor([0, -1, -1, 3], dtype=torch.int32)
    mc2_mask_h = torch.tensor([1, 0], dtype=torch.int32)

    manager._apply_multi_card_substitution(
        5,
        topk_ids_h,
        router_logits_h,
        log2phy_h,
        mc2_mask_h,
        "softmax",
        None,
    )

    assert topk_ids_h.tolist() == [[0, 3], [0, 1]]


def test_multi_card_substitution_honors_remote_blocker():
    manager = ExpertOffloadManager.__new__(ExpertOffloadManager)
    manager.offload_config = ExpertOffloadConfig({
        "expert_substitution_enabled": True,
        "expert_substitution_threshold": 0.30,
    })
    manager.topk = 2
    manager._debug = False
    router_logits_h = torch.log(
        torch.tensor([[0.40, 0.25, 0.20, 0.15]]))
    topk_ids_h = torch.tensor([[0, 1]], dtype=torch.int32)
    log2phy_h = torch.tensor([0, -1, -1, 3], dtype=torch.int32)

    def block_expert_one(referenced, blocked, cpu_group):
        del cpu_group
        global_blocked = blocked.clone()
        global_blocked[1] = True
        return referenced, global_blocked

    with patch(
        "vllm_ascend.expert_offload.multi_card_planner."
        "gather_global_substitution_state_cpu",
        side_effect=block_expert_one,
    ):
        manager._apply_multi_card_substitution(
            5,
            topk_ids_h,
            router_logits_h,
            log2phy_h,
            None,
            "softmax",
            None,
            object(),
        )

    assert topk_ids_h.tolist() == [[0, 1]]


def test_multi_card_substitution_all_pad_still_joins_global_state():
    manager = ExpertOffloadManager.__new__(ExpertOffloadManager)
    manager.offload_config = ExpertOffloadConfig({
        "expert_substitution_enabled": True,
        "expert_substitution_threshold": 0.30,
    })
    manager.topk = 2
    manager._debug = False
    topk_ids_h = torch.tensor([[0, 1]], dtype=torch.int32)
    gather = MagicMock(side_effect=lambda referenced, blocked, cpu_group:
                       (referenced, blocked))

    with patch(
        "vllm_ascend.expert_offload.multi_card_planner."
        "gather_global_substitution_state_cpu",
        gather,
    ):
        manager._apply_multi_card_substitution(
            5,
            topk_ids_h,
            torch.log(torch.tensor([[0.40, 0.25, 0.20, 0.15]])),
            torch.tensor([0, -1, -1, 3], dtype=torch.int32),
            torch.tensor([0], dtype=torch.int32),
            "softmax",
            None,
            object(),
        )

    gather.assert_called_once()
    assert topk_ids_h.tolist() == [[0, 1]]


def test_single_card_substitution_changes_ids_but_preserves_weights():
    manager = ExpertOffloadManager.__new__(ExpertOffloadManager)
    manager.offload_config = ExpertOffloadConfig({
        "expert_substitution_enabled": True,
        "expert_substitution_threshold": 0.30,
    })
    manager.topk = 2
    manager._debug = False
    manager.cache_policy = None
    manager._stats = None      # _update_weights reads it before any guard
    manager.load_stream = MagicMock()
    topk_ids_h = torch.tensor([[0, 1]], dtype=torch.int32)
    topk_weights_h = torch.tensor([[0.40, 0.25]])
    original_weights = topk_weights_h.clone()
    log2phy_h = torch.tensor([0, -1, -1, 1], dtype=torch.int32)
    log2phy_np = log2phy_h.numpy()
    layer = SimpleNamespace(w13_weight=torch.zeros(2, 1))

    args = (
        topk_ids_h,
        log2phy_np,
        layer,
        0,
        topk_weights_h,
        False,
        True,
        torch.log(torch.tensor([[0.40, 0.25, 0.20, 0.15]])),
        "softmax",
        None,
    )
    with patch(
        "vllm_ascend.expert_offload.expert_offload_manager.torch_npu.npu.stream",
        return_value=nullcontext(),
        create=True,
    ):
        manager._update_weights(args)

    assert topk_ids_h.tolist() == [[0, 3]]
    assert torch.equal(topk_weights_h, original_weights)


def test_prefetch_dispatch_uses_single_or_per_layer_multi_card_capacity():
    manager = ExpertOffloadManager.__new__(ExpertOffloadManager)
    manager.offload_config = ExpertOffloadConfig(
        {"num_device_experts": [8, 12]})
    manager._ep_size = 2
    manager._ep_info_resolved = True
    manager._update_weights = MagicMock()
    manager._update_weights_multi_card = MagicMock()
    inputs = (object(), object(), object(), object(), object(), 1)

    manager.enable_multi_card = False
    single_fn, single_args = manager._build_prefetch_call(*inputs)
    assert single_fn is manager._update_weights
    assert single_args[-1] is True

    manager.enable_multi_card = True
    multi_fn, multi_args = manager._build_prefetch_call(*inputs)
    assert multi_fn is manager._update_weights_multi_card
    assert multi_args[4] == 6
    assert multi_args[-2:] == (True, None)

    manager._debug = True
    _, debug_multi_args = manager._build_prefetch_call(*inputs)
    assert debug_multi_args[-3:] == (True, None, False)


def test_shared_expert_copy_is_non_blocking_and_refreshes_quant_scale():
    class RecordingStorage:
        def __init__(self):
            self.slice = None
            self.copies = []

        def __getitem__(self, storage_slice):
            self.slice = storage_slice
            return self

        def copy_(self, source, non_blocking=False):
            self.copies.append((self.slice, source, non_blocking))

    class Weight:
        def __init__(self):
            self.data = self
            self.storage = RecordingStorage()

        def untyped_storage(self):
            return self.storage

    manager = ExpertOffloadManager.__new__(ExpertOffloadManager)
    manager.w13_expert_size_bytes = 4
    manager.w2_expert_size_bytes = 2
    manager._expert_src_storage = MagicMock(
        side_effect=["w13-source", "w2-source"])
    manager.scale_cpu_buffers = {}
    manager.offset_cpu_buffers = {}
    manager.scale_bias_cpu_buffers = {}
    layer = SimpleNamespace(
        w13_weight=Weight(),
        w2_weight=Weight(),
        w13_weight_scale=SimpleNamespace(data=torch.tensor([1.25, 2.5])),
        w13_weight_scale_fp32=torch.zeros(2),
    )

    manager._load_expert_weights_into_slot(layer, 0, 7, 1)

    assert layer.w13_weight.storage.copies == [
        (slice(4, 8), "w13-source", True)]
    assert layer.w2_weight.storage.copies == [
        (slice(2, 4), "w2-source", True)]
    assert layer.w13_weight_scale_fp32[1] == 2.5


def test_memfabric_shared_resolves_remote_expert_pointer():
    manager = ExpertOffloadManager.__new__(ExpertOffloadManager)
    manager.offload_config = ExpertOffloadConfig({
        "h2d_backend": "memfabric",
        "memfabric_pool_size_gib": 1,
        "enable_multi_card": True,
        "shard_per_rank": True,
    })
    manager._shard_base = 0
    manager._shard_size = 2
    manager.w13_weights_cpu = [[torch.zeros(1), torch.zeros(1)]]
    manager._shared_h2d_sources_ready = True
    manager._shared_h2d_sources = {(0, 3, "w13"): 123456}

    source = manager._expert_src_storage(0, 3, "w13")

    assert source.data_ptr() == 123456


def test_memfabric_shared_publishes_complete_peer_pointer_table():
    manager = ExpertOffloadManager.__new__(ExpertOffloadManager)
    manager.offload_config = ExpertOffloadConfig({
        "h2d_backend": "memfabric",
        "memfabric_pool_size_gib": 1,
        "enable_multi_card": True,
        "shard_per_rank": True,
    })
    manager.h2d_transport = SimpleNamespace(supports_remote_sources=True)
    manager._ep_size = 2
    manager._ep_rank = 0
    manager._ep_info_resolved = True
    manager._shard_base = 0
    manager.num_total_experts = 2
    manager.w13_weights_cpu = [[torch.zeros(1)]]
    manager.w2_weights_cpu = [[torch.zeros(1)]]
    manager.scale_cpu_buffers = {}
    manager.offset_cpu_buffers = {}
    manager.scale_bias_cpu_buffers = {}
    manager._shared_h2d_sources = {}
    manager._shared_h2d_sources_ready = False

    def gather(output, local, group):
        output[0] = local
        output[1] = {
            (0, 1, "w13"): 301,
            (0, 1, "w2"): 302,
        }

    ep_group = SimpleNamespace(cpu_group=object())
    with (
        patch("torch.distributed.all_gather_object", side_effect=gather),
        patch("vllm.distributed.parallel_state.get_ep_group",
              return_value=ep_group),
    ):
        manager._publish_shared_h2d_sources()

    assert manager._shared_h2d_sources_ready is True
    assert manager._shared_h2d_sources[(0, 1, "w13")] == 301
    assert len(manager._shared_h2d_sources) == 4


def test_expert_load_combines_multiple_experts_into_one_transport_batch():
    manager = ExpertOffloadManager.__new__(ExpertOffloadManager)
    manager.h2d_transport = MagicMock()
    first_task = H2DCopyTask(object(), object(), 4, name="first")
    second_task = H2DCopyTask(object(), object(), 8, name="second")
    manager._build_expert_h2d_tasks = MagicMock(
        side_effect=[[first_task], [second_task]])
    manager._refresh_expert_fp32_scale = MagicMock()
    layer = object()

    manager._load_expert_weights_into_slots(
        layer, 3, [(1, 7), (2, 9)])

    manager.h2d_transport.copy_batch.assert_called_once_with(
        [first_task, second_task])
    assert manager._build_expert_h2d_tasks.call_args_list == [
        call(layer, 3, 7, 1),
        call(layer, 3, 9, 2),
    ]
    assert manager._refresh_expert_fp32_scale.call_args_list == [
        call(layer, 1),
        call(layer, 2),
    ]


def test_expert_host_allocation_and_synchronization_use_transport():
    manager = ExpertOffloadManager.__new__(ExpertOffloadManager)
    manager.h2d_transport = MagicMock()
    manager.load_stream = object()
    expected = object()
    manager.h2d_transport.allocate_host_tensor.return_value = expected

    result = manager._allocate_expert_host_tensor((2, 4), torch.float32)
    manager._synchronize_h2d()

    assert result is expected
    manager.h2d_transport.allocate_host_tensor.assert_called_once_with(
        (2, 4), torch.float32)
    manager.h2d_transport.synchronize.assert_called_once_with(
        manager.load_stream)


def test_prefill_weight_and_quant_attributes_are_one_h2d_task_batch():
    manager = ExpertOffloadManager.__new__(ExpertOffloadManager)
    manager.offload_config = ExpertOffloadConfig({"shard_per_rank": False})
    manager.w13_expert_size_bytes = 4
    manager.w2_expert_size_bytes = 2
    w13_source = torch.ones(4, dtype=torch.uint8).untyped_storage()
    w2_source = torch.ones(2, dtype=torch.uint8).untyped_storage()
    manager._expert_src_storage = MagicMock(
        side_effect=[w13_source, w2_source])
    manager._prefill_w13 = [torch.zeros((2, 4), dtype=torch.uint8)]
    manager._prefill_w2 = [torch.zeros((2, 2), dtype=torch.uint8)]
    manager._prefill_w13_scale = [torch.zeros((2, 2))]
    manager._prefill_w2_scale = []
    manager._prefill_w13_offset = []
    manager._prefill_w2_offset = []
    manager._prefill_w13_scale_bias = []
    manager._prefill_w2_scale_bias = []
    scale = torch.tensor([1.0, 2.0])
    manager.scale_cpu_buffers = {"w13_weight_scale": [[scale]]}
    manager.offset_cpu_buffers = {}
    manager.scale_bias_cpu_buffers = {}

    tasks = manager._build_prefill_h2d_tasks(0, 0, 0, 1)

    assert len(tasks) == 3
    assert [task.nbytes for task in tasks] == [4, 2, 8]
    assert tasks[2].source.data_ptr() == scale.data_ptr()
    assert tasks[2].destination.data_ptr() == \
        manager._prefill_w13_scale[0][1].data_ptr()


def test_quant_attributes_are_represented_as_h2d_tasks():
    manager = ExpertOffloadManager.__new__(ExpertOffloadManager)
    manager.offload_config = ExpertOffloadConfig({"shard_per_rank": False})
    source = torch.tensor([1.25, 2.5], dtype=torch.float32)
    manager.scale_cpu_buffers = {"w13_weight_scale": [[source]]}
    manager.offset_cpu_buffers = {}
    manager.scale_bias_cpu_buffers = {}
    destination = torch.zeros((1, 2), dtype=torch.float32)
    layer = SimpleNamespace(w13_weight_scale=destination)

    tasks = manager._build_quant_attr_h2d_tasks(layer, 0, 0, 0)

    assert len(tasks) == 1
    assert tasks[0].source.data_ptr() == source.data_ptr()
    assert tasks[0].destination.data_ptr() == destination[0].data_ptr()
    assert tasks[0].nbytes == source.numel() * source.element_size()
    assert tasks[0].non_blocking is True


def test_single_card_update_submits_misses_as_one_batch():
    manager = ExpertOffloadManager.__new__(ExpertOffloadManager)
    manager.offload_config = ExpertOffloadConfig()
    manager.topk = 1
    manager._debug = False
    manager.cache_policy = None
    manager._stats = None      # _update_weights reads it before any guard
    manager.load_stream = MagicMock()
    manager._load_expert_weights_into_slots = MagicMock()
    topk_ids_h = torch.tensor([[1]], dtype=torch.int32)
    log2phy_h = torch.tensor([0, -1], dtype=torch.int32)
    layer = SimpleNamespace()

    with patch(
        "vllm_ascend.expert_offload.expert_offload_manager.torch_npu.npu.stream",
        return_value=nullcontext(),
    ):
        manager._update_weights((
            topk_ids_h,
            log2phy_h.numpy(),
            layer,
            0,
            None,
            False,
        ))

    manager._load_expert_weights_into_slots.assert_called_once_with(
        layer, 0, [(0, 1)])
    assert log2phy_h.tolist() == [-1, 0]
    manager.load_stream.synchronize.assert_called_once_with()


def test_single_card_update_ranks_all_lrc_victims_once():
    manager = ExpertOffloadManager.__new__(ExpertOffloadManager)
    manager.offload_config = ExpertOffloadConfig()
    manager.topk = 2
    manager._debug = False
    manager.cache_policy = MagicMock()
    manager.cache_policy.observe.return_value = {4, 5}
    manager.cache_policy.choose_victims.return_value = [0, 1]
    manager._stats = None      # _update_weights reads it before any guard
    manager.load_stream = MagicMock()
    manager._record_cache_stats = MagicMock()
    manager._load_expert_weights_into_slots = MagicMock()
    manager._synchronize_h2d = MagicMock()
    topk_ids_h = torch.tensor([[4, 5]], dtype=torch.int32)
    log2phy_h = torch.tensor([0, 1, -1, -1, -1, -1], dtype=torch.int32)
    layer = SimpleNamespace()

    with patch(
        "vllm_ascend.expert_offload.expert_offload_manager.torch_npu.npu.stream",
        return_value=nullcontext(),
    ):
        manager._update_weights((
            topk_ids_h,
            log2phy_h.numpy(),
            layer,
            0,
            None,
            False,
        ))

    manager.cache_policy.choose_victims.assert_called_once_with(
        0, {0: 0, 1: 1}, protected={4, 5}, count=2)
    loads = manager._load_expert_weights_into_slots.call_args.args[2]
    assert {slot for slot, _ in loads} == {0, 1}
    assert {eid for _, eid in loads} == {4, 5}
    assert log2phy_h[:4].tolist() == [-1, -1, -1, -1]
    assert set(log2phy_h[4:].tolist()) == {0, 1}
    manager._synchronize_h2d.assert_called_once_with()


def test_multi_card_misses_submit_one_batch_before_resident_commit():
    manager = ExpertOffloadManager.__new__(ExpertOffloadManager)
    manager.load_stream = MagicMock()
    manager._load_expert_weights_into_slots = MagicMock()
    layer = object()
    misses = [(0, 4), (1, 7)]
    resident_map = {}

    with patch(
        "vllm_ascend.expert_offload.expert_offload_manager.torch_npu.npu.stream",
        return_value=nullcontext(),
    ):
        manager._h2d_load_mc_misses(layer, 3, misses, resident_map)

    manager._load_expert_weights_into_slots.assert_called_once_with(
        layer, 3, misses)
    assert resident_map == {0: 4, 1: 7}
    manager.load_stream.synchronize.assert_called_once_with()


def test_single_card_hot_preload_uses_format_aware_shared_copy(tmp_path):
    ranking_path = tmp_path / "hot.json"
    ranking_path.write_text(json.dumps({"0": [[3, 0.9], [1, 0.8],
                                                   [2, 0.7]]}))
    manager = ExpertOffloadManager.__new__(ExpertOffloadManager)
    manager.offload_config = ExpertOffloadConfig({
        "num_device_experts": 2,
        "hot_expert_preload": True,
        "hot_experts_file": str(ranking_path),
    })
    manager.enable_multi_card = False
    manager.cache_policy = None
    manager._debug = False
    manager.load_stream = MagicMock()
    manager.log2phy_h = torch.full((4,), -1, dtype=torch.int32)
    manager.log2phy_np = manager.log2phy_h.numpy()
    layer = SimpleNamespace(log2phy=MagicMock())
    manager.moe_layers = [layer]
    manager._load_expert_weights_into_slot = MagicMock()

    with patch(
        "vllm_ascend.expert_offload.expert_offload_manager.torch_npu.npu.stream",
        return_value=nullcontext(),
    ):
        manager._preload_hot_experts()

    assert manager._load_expert_weights_into_slot.call_args_list == [
        call(layer, 0, 3, 0),
        call(layer, 0, 1, 1),
    ]
    assert manager.log2phy_h.tolist() == [-1, 1, -1, 0]
    layer.log2phy.copy_.assert_called_once_with(manager.log2phy_h)


def test_multi_card_hot_preload_loads_only_rank_owned_experts(tmp_path):
    ranking_path = tmp_path / "hot.json"
    ranking_path.write_text(json.dumps({
        "0": [[0, 0.9], [1, 0.8], [2, 0.7], [3, 0.6],
              [4, 0.5], [5, 0.4], [6, 0.3], [7, 0.2]],
    }))
    manager = ExpertOffloadManager.__new__(ExpertOffloadManager)
    manager.offload_config = ExpertOffloadConfig({
        "num_device_experts": 4,
        "hot_expert_preload": True,
        "hot_experts_file": str(ranking_path),
        "shard_per_rank": True,
    })
    manager.enable_multi_card = True
    manager._ep_size = 2
    manager._ep_rank = 1
    manager._ep_info_resolved = True
    manager._shard_size = 4
    manager.cache_policy = MagicMock()
    manager._mc_lrc = None
    manager._mc_prev_log2phy = {}
    manager._mc_resident = {}
    manager._debug = False
    manager.load_stream = MagicMock()
    manager.num_total_experts = 8
    manager.log2phy_h = torch.full((8,), -1, dtype=torch.int32)
    manager.log2phy_np = manager.log2phy_h.numpy()
    layer = SimpleNamespace(log2phy=MagicMock())
    manager.moe_layers = [layer]
    manager._load_expert_weights_into_slot = MagicMock()

    with patch(
        "vllm_ascend.expert_offload.expert_offload_manager.torch_npu.npu.stream",
        return_value=nullcontext(),
    ):
        manager._preload_hot_experts()

    assert manager._load_expert_weights_into_slot.call_args_list == [
        call(layer, 0, 4, 0),
        call(layer, 0, 5, 1),
    ]
    assert manager.log2phy_h.tolist() == [0, 1, -1, -1, 2, 3, -1, -1]
    assert manager._mc_resident[0] == {0: 4, 1: 5}
    assert torch.equal(manager._mc_prev_log2phy[0], manager.log2phy_h)
    assert manager._mc_lrc is manager.cache_policy
    manager.cache_policy.seed_layer_hotness.assert_called_once_with(
        0, {0: 0.9, 1: 0.8, 4: 0.5, 5: 0.4})
    layer.log2phy.copy_.assert_called_once_with(manager.log2phy_h)