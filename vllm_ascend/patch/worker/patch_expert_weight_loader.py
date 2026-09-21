# SPDX-License-Identifier: Apache-2.0
"""Keep explicitly schema-backed O-MoE expert parameters on CPU after loading."""

from contextlib import contextmanager
from functools import wraps

from vllm.model_executor.model_loader import utils as model_loader_utils

from vllm_ascend.patch.worker import patch_process_weights_after_loading as ascend_loader


def _wrap_device_loading_context(original):
    @contextmanager
    @wraps(original)
    def device_loading_context(module, target_device):
        if getattr(module, "enable_omoe", False) is True:
            # Lazy imports keep the default worker/loader path independent of
            # quantization scheme imports during plugin initialization.
            from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
            from vllm_ascend.quantization.method_adapters import validate_omoe_cpu_weights
            if isinstance(module, RoutedExperts):
                validate_omoe_cpu_weights(module)
                yield module
                return
        with original(module, target_device) as loaded_module:
            yield loaded_module

    device_loading_context._omoe_original_context = original
    return device_loading_context


# utils.process_weights_after_loading resolves its own module global, whereas
# the Ascend implementation imported a local alias. Cover both exact callsites.
# Unwrap our own adapter when re-imported, without stacking wrappers recursively.
_original = getattr(
    model_loader_utils.device_loading_context,
    "_omoe_original_context",
    model_loader_utils.device_loading_context,
)
device_loading_context = _wrap_device_loading_context(_original)
model_loader_utils.device_loading_context = device_loading_context
ascend_loader.device_loading_context = device_loading_context
