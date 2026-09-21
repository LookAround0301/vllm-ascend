# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Select the model's hierarchical planner only for explicitly enabled O-MoE."""

import functools
import sys

import vllm.v1.core.kv_cache_utils as kv_cache_utils

from vllm_ascend.core.hierarchical_memory import (
    build_hierarchical_kv_cache_configs,
    hierarchical_omoe_enabled,
)


def install():
    current = kv_cache_utils.get_kv_cache_configs
    if getattr(current, "_ascend_hierarchical_planner", False):
        wrapper = current
    else:
        @functools.wraps(current)
        def wrapper(vllm_config, kv_cache_specs, available_memory):
            if hierarchical_omoe_enabled(vllm_config):
                return build_hierarchical_kv_cache_configs(vllm_config, kv_cache_specs, available_memory)
            return current(vllm_config, kv_cache_specs, available_memory)

        wrapper._ascend_hierarchical_planner = True
        kv_cache_utils.get_kv_cache_configs = wrapper
    # EngineCore imports the callable directly. It may already be imported
    # when platform patches are installed; avoid importing EngineCore here.
    engine = sys.modules.get("vllm.v1.engine.core")
    if engine is not None:
        engine.get_kv_cache_configs = wrapper
