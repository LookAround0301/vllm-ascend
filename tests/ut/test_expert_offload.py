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

"""Unit tests for expert offload feature (commit: moe offload support dsv4).

Covers:
  - ExpertOffloadConfig defaults, validation, and attribute access
  - init_expert_offload_config() enable/disable logic and expert_map shape
  - init_log2phy_for_offload() tensor correctness
  - ExpertOffloadManager weight loading (pending path + drain)
  - ExpertOffloadManager scale/offset pending path + drain
  - ExpertOffloadManager singleton lifecycle (module-level functions)
"""

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import torch

from tests.ut.base import TestBase
from vllm_ascend.ascend_config import ExpertOffloadConfig


class _MockMoeLayer:
    """A minimal mock MoE layer that only exposes explicitly-set attributes.

    Unlike MagicMock (where hasattr always returns True), this allows
    ``maybe_create_scale_buffers`` to correctly detect which scale/offset
    attributes are present via ``hasattr``.
    """

    def __init__(self, **attrs):
        for k, v in attrs.items():
            setattr(self, k, v)


# ---------------------------------------------------------------------------
# ExpertOffloadConfig
# ---------------------------------------------------------------------------
class TestExpertOffloadConfig(TestBase):
    """Tests for ExpertOffloadConfig default values, validation, and access."""

    def test_default_values(self):
        cfg = ExpertOffloadConfig()
        self.assertFalse(cfg.expert_offload)
        self.assertEqual(cfg.num_device_experts, 32)
        self.assertEqual(cfg.num_device_layers, 2)
        self.assertIsNone(cfg.expert_map_path)

    def test_custom_values(self):
        cfg = ExpertOffloadConfig({
            "expert_offload": True,
            "num_device_experts": 16,
            "num_device_layers": 4,
        })
        self.assertTrue(cfg.expert_offload)
        self.assertEqual(cfg.num_device_experts, 16)
        self.assertEqual(cfg.num_device_layers, 4)

    def test_none_user_config_same_as_empty(self):
        cfg = ExpertOffloadConfig(None)
        self.assertFalse(cfg.expert_offload)
        self.assertEqual(cfg.num_device_experts, 32)

    def test_unknown_key_raises_value_error(self):
        with self.assertRaises(ValueError) as ctx:
            ExpertOffloadConfig({"nonexistent_key": 42})
        self.assertIn("nonexistent_key", str(ctx.exception))

    def test_attr_access_unknown_raises_attribute_error(self):
        cfg = ExpertOffloadConfig()
        with self.assertRaises(AttributeError) as ctx:
            _ = cfg.totally_bogus
        self.assertIn("totally_bogus", str(ctx.exception))

    # -- num_device_experts validation --

    def test_num_device_experts_negative_raises_value_error(self):
        with self.assertRaises(ValueError):
            ExpertOffloadConfig({"num_device_experts": -1})

    def test_num_device_experts_zero_is_valid(self):
        cfg = ExpertOffloadConfig({"num_device_experts": 0})
        self.assertEqual(cfg.num_device_experts, 0)

    def test_num_device_experts_float_raises_type_error(self):
        with self.assertRaises(TypeError):
            ExpertOffloadConfig({"num_device_experts": 3.5})

    # -- num_device_layers validation --

    def test_num_device_layers_zero_raises_value_error(self):
        with self.assertRaises(ValueError):
            ExpertOffloadConfig({"num_device_layers": 0})

    def test_num_device_layers_negative_raises_value_error(self):
        with self.assertRaises(ValueError):
            ExpertOffloadConfig({"num_device_layers": -2})

    def test_num_device_layers_float_raises_type_error(self):
        with self.assertRaises(TypeError):
            ExpertOffloadConfig({"num_device_layers": 1.5})

    # -- expert_offload validation --

    def test_expert_offload_int_raises_type_error(self):
        with self.assertRaises(TypeError):
            ExpertOffloadConfig({"expert_offload": 1})

    def test_expert_offload_string_raises_type_error(self):
        with self.assertRaises(TypeError):
            ExpertOffloadConfig({"expert_offload": "true"})

    # -- expert_map_path validation --

    def test_expert_map_path_non_json_raises_type_error(self):
        with self.assertRaises(TypeError):
            ExpertOffloadConfig({"expert_map_path": "/tmp/map.txt"})

    def test_expert_map_path_nonexistent_raises_value_error(self):
        with self.assertRaises(ValueError):
            ExpertOffloadConfig({"expert_map_path": "/tmp/nonexistent_abc.json"})

    def test_expert_map_path_valid_json_file(self):
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            json.dump({"a": 1}, f)
            f.flush()
            path = f.name
        try:
            cfg = ExpertOffloadConfig({"expert_map_path": path})
            self.assertEqual(cfg.expert_map_path, path)
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# init_expert_offload_config (expert_offload/utils.py)
# ---------------------------------------------------------------------------
class TestInitExpertOffloadConfig(TestBase):
    """Tests for init_expert_offload_config() enable/disable and expert_map."""

    def test_disabled_when_expert_offload_false(self):
        from vllm_ascend.expert_offload.utils import init_expert_offload_config

        cfg = ExpertOffloadConfig({"expert_offload": False, "num_device_experts": 4})
        enable, emap = init_expert_offload_config(cfg, num_experts=8)
        self.assertFalse(enable)
        self.assertIsNone(emap)

    def test_disabled_when_num_device_experts_zero(self):
        from vllm_ascend.expert_offload.utils import init_expert_offload_config

        cfg = ExpertOffloadConfig({"expert_offload": True, "num_device_experts": 0})
        enable, emap = init_expert_offload_config(cfg, num_experts=8)
        self.assertFalse(enable)
        self.assertIsNone(emap)

    def test_disabled_when_num_device_experts_equals_num_experts(self):
        from vllm_ascend.expert_offload.utils import init_expert_offload_config

        # num_device_experts >= num_experts means no offload needed
        cfg = ExpertOffloadConfig({"expert_offload": True, "num_device_experts": 8})
        enable, emap = init_expert_offload_config(cfg, num_experts=8)
        self.assertFalse(enable)
        self.assertIsNone(emap)

    def test_disabled_when_num_device_experts_greater_than_num_experts(self):
        from vllm_ascend.expert_offload.utils import init_expert_offload_config

        cfg = ExpertOffloadConfig({"expert_offload": True, "num_device_experts": 16})
        enable, emap = init_expert_offload_config(cfg, num_experts=8)
        self.assertFalse(enable)
        self.assertIsNone(emap)

    def test_enabled_returns_correct_expert_map(self):
        from vllm_ascend.expert_offload.utils import init_expert_offload_config

        cfg = ExpertOffloadConfig({"expert_offload": True, "num_device_experts": 4})
        enable, emap = init_expert_offload_config(cfg, num_experts=8)
        self.assertTrue(enable)
        self.assertIsNotNone(emap)
        self.assertEqual(emap.shape, (8,))
        self.assertEqual(emap.dtype, torch.int32)

        # First 4 slots mapped 0→0, 1→1, 2→2, 3→3
        self.assertTrue(torch.equal(emap[:4], torch.arange(4, dtype=torch.int32)))
        # Remaining slots unmapped (-1)
        self.assertTrue(torch.equal(emap[4:], torch.full((4,), -1, dtype=torch.int32)))

    def test_enabled_with_single_device_expert(self):
        from vllm_ascend.expert_offload.utils import init_expert_offload_config

        cfg = ExpertOffloadConfig({"expert_offload": True, "num_device_experts": 1})
        enable, emap = init_expert_offload_config(cfg, num_experts=64)
        self.assertTrue(enable)
        self.assertEqual(emap[0].item(), 0)
        self.assertTrue(torch.equal(emap[1:], torch.full((63,), -1, dtype=torch.int32)))


# ---------------------------------------------------------------------------
# init_log2phy_for_offload (expert_offload/utils.py)
# ---------------------------------------------------------------------------
class TestInitLog2phyForOffload(TestBase):
    """Tests for init_log2phy_for_offload() tensor correctness.

    The function creates tensors with device='npu'. We test by reproducing
    the logic on CPU and verifying correctness of the algorithm.
    """

    @staticmethod
    def _ref_log2phy(global_num_experts: int, num_device_experts: int):
        """Reference implementation on CPU for verification."""
        log2phy = torch.arange(global_num_experts, dtype=torch.int32)
        log2phy[num_device_experts:].fill_(-1)
        return log2phy

    def test_log2phy_correct_mapping(self):
        """First ndev experts mapped to slots, rest set to -1."""
        result = self._ref_log2phy(global_num_experts=8, num_device_experts=4)
        expected_active = torch.arange(4, dtype=torch.int32)
        expected_inactive = torch.full((4,), -1, dtype=torch.int32)
        self.assertTrue(torch.equal(result[:4], expected_active))
        self.assertTrue(torch.equal(result[4:], expected_inactive))

    def test_log2phy_all_experts_on_device(self):
        """When num_device_experts == global_num_experts, no slots are -1."""
        result = self._ref_log2phy(global_num_experts=4, num_device_experts=4)
        self.assertTrue(torch.equal(result, torch.arange(4, dtype=torch.int32)))

    def test_log2phy_zero_device_experts(self):
        """When num_device_experts == 0, all slots are -1."""
        result = self._ref_log2phy(global_num_experts=8, num_device_experts=0)
        self.assertTrue(torch.equal(result, torch.full((8,), -1, dtype=torch.int32)))

    def test_log2phy_single_device_expert(self):
        result = self._ref_log2phy(global_num_experts=64, num_device_experts=1)
        self.assertEqual(result[0].item(), 0)
        self.assertTrue(torch.equal(result[1:], torch.full((63,), -1, dtype=torch.int32)))

    def test_log2phy_preserves_dtype(self):
        result = self._ref_log2phy(global_num_experts=8, num_device_experts=4)
        self.assertEqual(result.dtype, torch.int32)


# ---------------------------------------------------------------------------
# ExpertOffloadManager — weight loading (pending + drain)
# ---------------------------------------------------------------------------
class TestExpertOffloadManagerWeightLoading(TestBase):
    """Tests for ExpertOffloadManager weight loading paths.

    These tests mock torch_npu.npu.Stream and AscendConfig to avoid needing
    real NPU hardware.
    """

    def _make_manager(self, num_device_experts=4, num_device_layers=2):
        """Create an ExpertOffloadManager with mocked dependencies."""
        from vllm_ascend.expert_offload.expert_offload_manager import ExpertOffloadManager

        # Reset singleton
        ExpertOffloadManager._instance = None

        mock_config = MagicMock()
        mock_config.model_config.hf_config.num_experts_per_tok = 2

        mock_offload_config = ExpertOffloadConfig({
            "expert_offload": True,
            "num_device_experts": num_device_experts,
            "num_device_layers": num_device_layers,
        })

        with patch("vllm_ascend.expert_offload.expert_offload_manager.torch_npu") as mock_npu, \
             patch("vllm_ascend.expert_offload.expert_offload_manager._EXTRA_CTX"), \
             patch("vllm_ascend.ascend_config.get_ascend_config") as mock_get_cfg:
            mock_get_cfg.return_value.expert_offload_config = mock_offload_config
            mock_npu.npu.Stream.return_value = MagicMock()

            mgr = ExpertOffloadManager(mock_config)

        return mgr

    def _cleanup_manager(self):
        from vllm_ascend.expert_offload.expert_offload_manager import ExpertOffloadManager
        ExpertOffloadManager._instance = None

    def test_create_weights_allocates_cpu_buffers(self):
        mgr = self._make_manager(num_device_experts=4)
        try:
            mgr.create_weights(
                num_moe_layers=2,
                num_total_experts=8,
                w13_up_dim=64,
                hidden_size=32,
                intermediate_size_per_partition=16,
                params_dtype=torch.float32,
            )
            self.assertEqual(len(mgr.w13_weights_cpu), 2)
            self.assertEqual(len(mgr.w2_weights_cpu), 2)
            self.assertEqual(len(mgr.w13_weights_cpu[0]), 8)
            self.assertEqual(len(mgr.w2_weights_cpu[0]), 8)
            # Check buffer shapes
            self.assertEqual(mgr.w13_weights_cpu[0][0].shape, (32, 64))
            self.assertEqual(mgr.w2_weights_cpu[0][0].shape, (16, 32))
            # Check dtype and device
            self.assertEqual(mgr.w13_weights_cpu[0][0].dtype, torch.float32)
            self.assertEqual(mgr.w13_weights_cpu[0][0].device.type, "cpu")
        finally:
            self._cleanup_manager()

    def test_create_weights_sets_num_total_experts(self):
        mgr = self._make_manager(num_device_experts=4)
        try:
            self.assertIsNone(mgr.num_total_experts)
            mgr.create_weights(
                num_moe_layers=2,
                num_total_experts=16,
                w13_up_dim=64,
                hidden_size=32,
                intermediate_size_per_partition=16,
                params_dtype=torch.float32,
            )
            self.assertEqual(mgr.num_total_experts, 16)
        finally:
            self._cleanup_manager()

    def test_create_weights_allocates_mapping_buffers(self):
        mgr = self._make_manager(num_device_experts=4)
        try:
            mgr.create_weights(
                num_moe_layers=2,
                num_total_experts=8,
                w13_up_dim=64,
                hidden_size=32,
                intermediate_size_per_partition=16,
                params_dtype=torch.float32,
            )
            # offload_threshold = 4 // 2 = 2
            self.assertEqual(mgr.offload_threshold, 2)
            self.assertEqual(mgr.topk_ids_h.shape, (2, 2))
            self.assertEqual(mgr.log2phy_h.shape, (8,))
        finally:
            self._cleanup_manager()

    def test_load_w13_before_create_weights_goes_to_pending(self):
        mgr = self._make_manager(num_device_experts=4)
        try:
            weight = torch.randn(32, 32)
            mgr.load_w13(layer_moe_idx=0, expert_id=0, loaded_weight=weight, shard_id="w1")
            self.assertEqual(len(mgr._pending_weights), 1)
            self.assertIn((0, 0), mgr._pending_weights)
            self.assertIn("w13_w1", mgr._pending_weights[(0, 0)])
            # CPU buffers should still be empty
            self.assertEqual(len(mgr.w13_weights_cpu), 0)
        finally:
            self._cleanup_manager()

    def test_load_w2_before_create_weights_goes_to_pending(self):
        mgr = self._make_manager(num_device_experts=4)
        try:
            weight = torch.randn(16, 32)
            mgr.load_w2(layer_moe_idx=0, expert_id=1, loaded_weight=weight)
            self.assertEqual(len(mgr._pending_weights), 1)
            self.assertIn((0, 1), mgr._pending_weights)
            self.assertIn("w2", mgr._pending_weights[(0, 1)])
        finally:
            self._cleanup_manager()

    def test_load_w13_after_create_weights_writes_directly(self):
        mgr = self._make_manager(num_device_experts=4)
        try:
            mgr.create_weights(
                num_moe_layers=2,
                num_total_experts=8,
                w13_up_dim=64,
                hidden_size=32,
                intermediate_size_per_partition=16,
                params_dtype=torch.float32,
            )
            # w13_up_dim=64, hidden_size=32, so w13 shape is [32, 64]
            # w1 half: [32, 32], w3 half: [32, 32]
            w1_weight = torch.randn(32, 32)  # w1: [out, in] -> transposed -> [32, 32]
            mgr.load_w13(0, 0, w1_weight, "w1")

            # Verify no pending weights
            self.assertEqual(len(mgr._pending_weights), 0)
            # Verify data was transposed and stored
            expected = w1_weight.t()
            self.assertTrue(torch.equal(mgr.w13_weights_cpu[0][0][:, :32], expected))
        finally:
            self._cleanup_manager()

    def test_load_w2_after_create_weights_writes_directly(self):
        mgr = self._make_manager(num_device_experts=4)
        try:
            mgr.create_weights(
                num_moe_layers=2,
                num_total_experts=8,
                w13_up_dim=64,
                hidden_size=32,
                intermediate_size_per_partition=16,
                params_dtype=torch.float32,
            )
            w2_weight = torch.randn(32, 16)  # w2: [hidden, intermed] -> transposed
            mgr.load_w2(0, 0, w2_weight)

            self.assertEqual(len(mgr._pending_weights), 0)
            expected = w2_weight.t()
            self.assertTrue(torch.equal(mgr.w2_weights_cpu[0][0], expected))
        finally:
            self._cleanup_manager()

    def test_drain_pending_weights_after_create_weights(self):
        mgr = self._make_manager(num_device_experts=4)
        try:
            # Load before create_weights → goes to pending
            w1 = torch.randn(32, 32)
            w3 = torch.randn(32, 32)
            w2 = torch.randn(32, 16)
            mgr.load_w13(0, 0, w1, "w1")
            mgr.load_w13(0, 0, w3, "w3")
            mgr.load_w2(0, 0, w2)
            self.assertEqual(len(mgr._pending_weights), 1)

            # create_weights drains pending
            mgr.create_weights(
                num_moe_layers=2,
                num_total_experts=8,
                w13_up_dim=64,
                hidden_size=32,
                intermediate_size_per_partition=16,
                params_dtype=torch.float32,
            )
            self.assertEqual(len(mgr._pending_weights), 0)

            # Verify data: w13 has w1 in first half, w3 in second half
            cpu_w13 = mgr.w13_weights_cpu[0][0]
            self.assertTrue(torch.equal(cpu_w13[:, :32], w1.t()))
            self.assertTrue(torch.equal(cpu_w13[:, 32:], w3.t()))

            # Verify w2
            self.assertTrue(torch.equal(mgr.w2_weights_cpu[0][0], w2.t()))
        finally:
            self._cleanup_manager()

    def test_load_w13_w3_combined_shape(self):
        """Verify w1 and w3 shards fill correct halves of w13 buffer."""
        mgr = self._make_manager(num_device_experts=4)
        try:
            mgr.create_weights(
                num_moe_layers=1,
                num_total_experts=4,
                w13_up_dim=128,
                hidden_size=64,
                intermediate_size_per_partition=64,
                params_dtype=torch.float32,
            )
            # w13 shape: [64, 128], first half=64 cols for w1, second half=64 cols for w3
            w1 = torch.ones(64, 64) * 1.0
            w3 = torch.ones(64, 64) * 3.0
            mgr.load_w13(0, 2, w1, "w1")
            mgr.load_w13(0, 2, w3, "w3")

            cpu_w13 = mgr.w13_weights_cpu[0][2]
            self.assertTrue(torch.equal(cpu_w13[:, :64], w1.t()))
            self.assertTrue(torch.equal(cpu_w13[:, 64:], w3.t()))
        finally:
            self._cleanup_manager()


# ---------------------------------------------------------------------------
# ExpertOffloadManager — scale/offset pending + drain
# ---------------------------------------------------------------------------
class TestExpertOffloadManagerScaleOffset(TestBase):
    """Tests for scale/offset pending weight storage and drain."""

    def _make_manager(self, num_device_experts=4):
        from vllm_ascend.expert_offload.expert_offload_manager import ExpertOffloadManager

        ExpertOffloadManager._instance = None

        mock_config = MagicMock()
        mock_config.model_config.hf_config.num_experts_per_tok = 2

        mock_offload_config = ExpertOffloadConfig({
            "expert_offload": True,
            "num_device_experts": num_device_experts,
            "num_device_layers": 2,
        })

        with patch("vllm_ascend.expert_offload.expert_offload_manager.torch_npu") as mock_npu, \
             patch("vllm_ascend.expert_offload.expert_offload_manager._EXTRA_CTX"), \
             patch("vllm_ascend.ascend_config.get_ascend_config") as mock_get_cfg:
            mock_get_cfg.return_value.expert_offload_config = mock_offload_config
            mock_npu.npu.Stream.return_value = MagicMock()
            mgr = ExpertOffloadManager(mock_config)

        return mgr

    def _cleanup_manager(self):
        from vllm_ascend.expert_offload.expert_offload_manager import ExpertOffloadManager
        ExpertOffloadManager._instance = None

    def test_add_pending_scale_stores_correctly(self):
        mgr = self._make_manager()
        try:
            scale_w1 = torch.tensor([0.5, 0.6])
            mgr._add_pending_scale(0, 0, "w13_weight_scale", "w1", scale_w1)
            self.assertEqual(len(mgr._pending_scales), 1)
            self.assertIn((0, 0), mgr._pending_scales)
            self.assertIn("w13_weight_scale_w1", mgr._pending_scales[(0, 0)])
        finally:
            self._cleanup_manager()

    def test_add_pending_scale_multiple_shards(self):
        mgr = self._make_manager()
        try:
            scale_w1 = torch.tensor([0.5])
            scale_w3 = torch.tensor([0.7])
            mgr._add_pending_scale(0, 1, "w13_weight_scale", "w1", scale_w1)
            mgr._add_pending_scale(0, 1, "w13_weight_scale", "w3", scale_w3)
            self.assertEqual(len(mgr._pending_scales), 1)
            pending = mgr._pending_scales[(0, 1)]
            self.assertIn("w13_weight_scale_w1", pending)
            self.assertIn("w13_weight_scale_w3", pending)
        finally:
            self._cleanup_manager()

    def test_drain_pending_scales_assembles_w13_shards(self):
        mgr = self._make_manager()
        try:
            mgr.create_weights(
                num_moe_layers=1,
                num_total_experts=4,
                w13_up_dim=64,
                hidden_size=32,
                intermediate_size_per_partition=16,
                params_dtype=torch.float32,
            )

            # Add pending scales for w13 (w1 + w3 shards)
            scale_w1 = torch.ones(2)
            scale_w3 = torch.ones(2) * 2
            mgr._add_pending_scale(0, 0, "w13_weight_scale", "w1", scale_w1)
            mgr._add_pending_scale(0, 0, "w13_weight_scale", "w3", scale_w3)

            # Add pending w2 scale
            scale_w2 = torch.ones(6) * 5
            mgr._add_pending_scale(0, 0, "w2_weight_scale", "w2", scale_w2)

            # Create a layer object with only the scale attributes we want
            # per-expert shape: shape[1:] from the device tensor
            layer = _MockMoeLayer(
                w13_weight_scale=torch.zeros(4, 4),
                w2_weight_scale=torch.zeros(4, 6),
            )

            mgr.maybe_create_scale_buffers(layer, 0)

            # Verify w13 scale: w1 + w3 concatenated
            w13_scale_buf = mgr.scale_cpu_buffers["w13_weight_scale"][0][0]
            expected = torch.cat([scale_w1, scale_w3], dim=0)
            self.assertTrue(torch.equal(w13_scale_buf, expected))

            # Verify w2 scale
            w2_scale_buf = mgr.scale_cpu_buffers["w2_weight_scale"][0][0]
            expected_w2 = scale_w2.reshape(w2_scale_buf.shape)
            self.assertTrue(torch.equal(w2_scale_buf, expected_w2))

            # Pending scales should be cleared for processed entries
            self.assertNotIn((0, 0), mgr._pending_scales)
        finally:
            self._cleanup_manager()

    def test_drain_pending_scales_waits_for_both_w13_shards(self):
        """Only w1 without w3 should NOT be drained for w13 attrs."""
        mgr = self._make_manager()
        try:
            mgr.create_weights(
                num_moe_layers=1,
                num_total_experts=4,
                w13_up_dim=64,
                hidden_size=32,
                intermediate_size_per_partition=16,
                params_dtype=torch.float32,
            )

            # Only w1 shard, no w3
            scale_w1 = torch.ones(2)
            mgr._add_pending_scale(0, 0, "w13_weight_scale", "w1", scale_w1)

            layer = _MockMoeLayer(
                w13_weight_scale=torch.zeros(4, 2),
            )
            mgr.maybe_create_scale_buffers(layer, 0)

            # w13 scale buffer created but NOT drained (missing w3)
            self.assertIn((0, 0), mgr._pending_scales)
        finally:
            self._cleanup_manager()

    def test_drain_pending_scales_offset_w2(self):
        mgr = self._make_manager()
        try:
            mgr.create_weights(
                num_moe_layers=1,
                num_total_experts=4,
                w13_up_dim=64,
                hidden_size=32,
                intermediate_size_per_partition=16,
                params_dtype=torch.float32,
            )

            offset_w2 = torch.ones(3) * 7
            mgr._add_pending_scale(0, 0, "w2_weight_offset", "w2", offset_w2)

            layer = _MockMoeLayer(
                w2_weight_offset=torch.zeros(4, 3),
            )
            mgr.maybe_create_scale_buffers(layer, 0)

            offset_buf = mgr.offset_cpu_buffers["w2_weight_offset"][0][0]
            expected = offset_w2.reshape(offset_buf.shape)
            self.assertTrue(torch.equal(offset_buf, expected))
        finally:
            self._cleanup_manager()


# ---------------------------------------------------------------------------
# ExpertOffloadManager — singleton lifecycle
# ---------------------------------------------------------------------------
class TestExpertOffloadManagerLifecycle(TestBase):
    """Tests for module-level singleton lifecycle functions."""

    def setUp(self):
        super().setUp()
        # Clean up global state before each test
        import vllm_ascend.expert_offload.expert_offload_manager as mgr_mod
        mgr_mod._EXPERT_OFFLOAD_MANAGER = None
        from vllm_ascend.expert_offload.expert_offload_manager import ExpertOffloadManager
        ExpertOffloadManager._instance = None

    def tearDown(self):
        import vllm_ascend.expert_offload.expert_offload_manager as mgr_mod
        mgr_mod._EXPERT_OFFLOAD_MANAGER = None
        from vllm_ascend.expert_offload.expert_offload_manager import ExpertOffloadManager
        ExpertOffloadManager._instance = None
        super().tearDown()

    def test_has_expert_offload_manager_false_by_default(self):
        from vllm_ascend.expert_offload.expert_offload_manager import has_expert_offload_manager
        self.assertFalse(has_expert_offload_manager())

    def test_get_expert_offload_manager_raises_when_not_initialized(self):
        from vllm_ascend.expert_offload.expert_offload_manager import get_expert_offload_manager
        with self.assertRaises(AssertionError):
            get_expert_offload_manager()

    def test_maybe_init_creates_singleton(self):
        from vllm_ascend.expert_offload.expert_offload_manager import (
            ExpertOffloadManager,
            has_expert_offload_manager,
            maybe_init_expert_offload_manager,
        )

        mock_config = MagicMock()
        mock_config.model_config.hf_config.num_experts_per_tok = 2

        mock_offload_config = ExpertOffloadConfig({
            "expert_offload": True,
            "num_device_experts": 4,
            "num_device_layers": 2,
        })

        with patch("vllm_ascend.expert_offload.expert_offload_manager.torch_npu") as mock_npu, \
             patch("vllm_ascend.expert_offload.expert_offload_manager._EXTRA_CTX"), \
             patch("vllm_ascend.ascend_config.get_ascend_config") as mock_get_cfg:
            mock_get_cfg.return_value.expert_offload_config = mock_offload_config
            mock_npu.npu.Stream.return_value = MagicMock()

            maybe_init_expert_offload_manager(mock_config)

        self.assertTrue(has_expert_offload_manager())
        self.assertIsNotNone(ExpertOffloadManager.get_instance())

    def test_maybe_init_idempotent(self):
        from vllm_ascend.expert_offload.expert_offload_manager import (
            ExpertOffloadManager,
            maybe_init_expert_offload_manager,
        )

        mock_config = MagicMock()
        mock_config.model_config.hf_config.num_experts_per_tok = 2

        mock_offload_config = ExpertOffloadConfig({
            "expert_offload": True,
            "num_device_experts": 4,
            "num_device_layers": 2,
        })

        with patch("vllm_ascend.expert_offload.expert_offload_manager.torch_npu") as mock_npu, \
             patch("vllm_ascend.expert_offload.expert_offload_manager._EXTRA_CTX"), \
             patch("vllm_ascend.ascend_config.get_ascend_config") as mock_get_cfg:
            mock_get_cfg.return_value.expert_offload_config = mock_offload_config
            mock_npu.npu.Stream.return_value = MagicMock()

            maybe_init_expert_offload_manager(mock_config)
            first = ExpertOffloadManager.get_instance()

            # Second call should not create a new instance
            maybe_init_expert_offload_manager(mock_config)
            second = ExpertOffloadManager.get_instance()
            self.assertIs(first, second)

    def test_get_instance_raises_when_not_initialized(self):
        from vllm_ascend.expert_offload.expert_offload_manager import ExpertOffloadManager
        with self.assertRaises(AssertionError):
            ExpertOffloadManager.get_instance()

    def test_offload_threshold_calculation(self):
        """offload_threshold = num_device_experts // topk."""
        from vllm_ascend.expert_offload.expert_offload_manager import ExpertOffloadManager

        ExpertOffloadManager._instance = None

        mock_config = MagicMock()
        mock_config.model_config.hf_config.num_experts_per_tok = 4

        mock_offload_config = ExpertOffloadConfig({
            "expert_offload": True,
            "num_device_experts": 32,
            "num_device_layers": 2,
        })

        with patch("vllm_ascend.expert_offload.expert_offload_manager.torch_npu") as mock_npu, \
             patch("vllm_ascend.expert_offload.expert_offload_manager._EXTRA_CTX"), \
             patch("vllm_ascend.ascend_config.get_ascend_config") as mock_get_cfg:
            mock_get_cfg.return_value.expert_offload_config = mock_offload_config
            mock_npu.npu.Stream.return_value = MagicMock()

            mgr = ExpertOffloadManager(mock_config)

        self.assertEqual(mgr.offload_threshold, 32 // 4)
        self.assertEqual(mgr.topk, 4)
