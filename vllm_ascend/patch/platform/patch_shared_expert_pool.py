# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Opt-in per-engine shared KV/expert ownership without replacing core.step."""

from collections import deque
from contextlib import suppress
from functools import wraps

from vllm.v1.engine.core import EngineCore


def attach_expert_manager(engine):
    """Add OMoE cache deltas to the native schedule/execute completion path."""
    from vllm import envs
    from vllm.v1.executor.multiproc_executor import FutureWrapper

    from vllm_ascend.core.hierarchical_cache_manager import HierarchicalCacheManager
    from vllm_ascend.core.hierarchical_memory import HierarchicalMemoryLayout
    from vllm_ascend.expert_offload.expert_manager import ExpertManager

    existing = getattr(engine, "_omoe_expert_manager", None)
    if existing is not None:
        return existing
    config = engine.vllm_config
    worker_count = config.parallel_config.world_size
    infos = engine.collective_rpc("get_omoe_pool_info")
    if len(infos) != worker_count:
        raise RuntimeError("O-MoE initialization requires every worker's layout")
    native_manager = engine.scheduler.kv_cache_manager
    layout = HierarchicalMemoryLayout.from_config(
        config, native_manager.kv_cache_config, cache_schemas=infos[0].get("cache_schemas")
    )
    cache_manager = HierarchicalCacheManager(native_manager, layout)
    manager = ExpertManager(config, cache_manager, infos)
    # Startup still uses blocking initialization calls, as the native engine
    # does. Bound pinned-host peak by preparing one rank's sources at a time.
    replies = engine.collective_rpc("release_omoe_window", args=(0,))
    if len(replies) != worker_count or any(
        reply.get("released") is not True or reply.get("delta_id") != 0 for reply in replies
    ):
        raise RuntimeError("O-MoE warmup ownership was not retired on every worker")
    for target_rank in range(worker_count):
        replies = engine.collective_rpc("prepare_omoe_expert_sources", args=(target_rank, 0))
        if len(replies) != worker_count or any(
            reply != {"rank": rank, "target_rank": target_rank, "delta_id": 0,
                      "status": "ready" if rank == target_rank else "skipped"}
            for rank, reply in enumerate(replies)
        ):
            raise RuntimeError("O-MoE expert source initialization did not complete")

    original_schedule = engine.scheduler.schedule
    executor = engine.model_executor

    @wraps(original_schedule)
    def schedule(*args, **kwargs):
        try:
            if manager.failed or manager._pending_delta is not None:
                raise RuntimeError("O-MoE previous execution did not complete; restart the engine")
            cache_manager.required_expert_shrink_blocks = 0
            output = original_schedule(*args, **kwargs)
            output.expert_cache_delta = manager.adjust_expert_cache_capacity(output.total_num_scheduled_tokens)
            return output
        except BaseException:
            manager.failed = True
            raise

    def execute_model(scheduler_output, non_block=False):
        delta = scheduler_output.expert_cache_delta

        def get_response():
            try:
                replies = future.result()
                manager.complete_cache_delta(delta, [snapshot for _, snapshot in replies])
                return replies[getattr(executor, "output_rank", 0)][0]
            except BaseException:
                manager.failed = True
                raise

        try:
            # Collect state with the normal execute_model result, instead of
            # sending separate snapshot/release/install RPCs each step.
            future = executor.collective_rpc(
                "execute_model", args=(scheduler_output,), non_block=True,
                timeout=envs.VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS,
            )
            result = FutureWrapper(deque(), get_response=get_response)
            return result if non_block else result.result()
        except BaseException:
            manager.failed = True
            raise

    engine.scheduler.schedule = schedule
    executor.execute_model = execute_model
    engine._omoe_expert_manager = manager
    return manager


def patch_engine_core(engine_core_class):
    original_init = engine_core_class.__init__
    if getattr(original_init, "_omoe_shared_pool_patch", False):
        return
    original_shutdown = engine_core_class.shutdown

    @wraps(original_init)
    def initialize(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        # resolve_omoe_config serializes enabled ON configuration before
        # worker creation. Do not re-read env/config singletons here: an OFF
        # engine must retain its original scheduler, cache, and RPC behavior.
        additional = self.vllm_config.additional_config or {}
        if additional.get("omoe_config", {}).get("enabled") is not True:
            return
        # Keep scheduler control torch-free and avoid loading it at all OFF.
        try:
            attach_expert_manager(self)
        except BaseException as error:
            # Normal EngineCore init completed, but no caller necessarily has
            # an engine handle yet. Release its executor, scheduler and other
            # initialized resources rather than relying on process-exit GC.
            # Use the base shutdown captured here: subclass init may not have
            # completed. Cleanup is best-effort, not a guarantee that every
            # resource retired; preserve secondary failure evidence without
            # replacing the original attach error.
            try:
                original_shutdown(self)
            except BaseException as cleanup_error:
                with suppress(BaseException):
                    error.add_note(f"O-MoE startup cleanup also failed: {cleanup_error!r}")
            raise

    initialize._omoe_shared_pool_patch = True
    engine_core_class.__init__ = initialize


patch_engine_core(EngineCore)
