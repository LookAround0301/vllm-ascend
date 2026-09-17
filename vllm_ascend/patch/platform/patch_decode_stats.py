# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Feed scheduler-authoritative speculative metrics to DECODE-STATS.

The model runner cannot faithfully reconstruct vLLM's acceptance counters
under async scheduling: the scheduler is the component that knows whether an
output was stale/discarded and how many invalid speculative tokens were
removed. ``Scheduler.make_stats`` receives the already-finalized
``SpecDecodingStats`` once per scheduler output, making it the narrowest common
hook for the default and Ascend scheduler subclasses.
"""

from functools import wraps

from vllm.logger import logger
from vllm.v1.core.sched.scheduler import Scheduler

from vllm_ascend.expert_offload.decode_stats import get_decode_stats

_PATCH_MARKER = "_vllm_ascend_decode_stats_wrapped"
_ORIGINAL_MARKER = "_vllm_ascend_decode_stats_original"
_ORIGINAL_MAKE_STATS = getattr(
    Scheduler.make_stats, _ORIGINAL_MARKER, Scheduler.make_stats)


def _record_scheduler_decode_stats(spec_decoding_stats) -> None:
    try:
        collector = get_decode_stats()
        if collector is None or not collector.collecting:
            return

        if spec_decoding_stats is None:
            accepted = proposed = n_drafts = 0
        else:
            accepted = int(spec_decoding_stats.num_accepted_tokens)
            proposed = int(spec_decoding_stats.num_draft_tokens)
            n_drafts = int(spec_decoding_stats.num_drafts)

        collector.record_spec_step_totals(
            accepted=accepted,
            proposed=proposed,
            n_drafts=n_drafts,
        )
    except Exception:
        # Statistics must never make scheduler output processing fail.
        logger.exception(
            "[DECODE-STATS] failed to record scheduler speculative metrics")


@wraps(_ORIGINAL_MAKE_STATS)
def make_stats(self, spec_decoding_stats=None, *args, **kwargs):
    result = _ORIGINAL_MAKE_STATS(
        self, spec_decoding_stats, *args, **kwargs)
    _record_scheduler_decode_stats(spec_decoding_stats)
    return result


setattr(make_stats, _PATCH_MARKER, True)
setattr(make_stats, _ORIGINAL_MARKER, _ORIGINAL_MAKE_STATS)
if not getattr(Scheduler.make_stats, _PATCH_MARKER, False):
    Scheduler.make_stats = make_stats
