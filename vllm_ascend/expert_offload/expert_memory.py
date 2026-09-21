# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Expert weight layouts and views over allocator-owned block IDs.

The shared block pool owns allocation and retirement. Workers construct views
once per installed slot and track copy/compute events on that slot.
"""

from dataclasses import dataclass, field
from typing import Protocol

import torch

from vllm.utils.math_utils import round_up

# Shared-pool storage alignment; individual expert layouts use their own value.
EXPERT_ALIGNMENT = 512




class CompletionEvent(Protocol):
    def query(self) -> bool: ...

    def synchronize(self) -> None: ...


@dataclass(frozen=True, slots=True)
class ExpertMemoryLayout:
    """Symmetric W8A8 payload; gate precedes up in fused ND W13.

    Shape restrictions of a particular compute kernel belong to its prepare
    contract. This storage layout accepts positive H/I and aligns every
    component independently, while channel scales remain resident on the model layer.
    """

    hidden_size: int
    intermediate_size: int
    alignment: int = 32
    w2_offset: int = field(init=False)
    payload_nbytes: int = field(init=False)
    extent_nbytes: int = field(init=False)

    def __post_init__(self) -> None:
        hidden_size, intermediate_size, alignment = self.hidden_size, self.intermediate_size, self.alignment
        if alignment <= 0:
            raise ValueError("alignment must be positive")
        w13_bytes, w2_bytes = 2 * hidden_size * intermediate_size, hidden_size * intermediate_size
        w2_offset = round_up(w13_bytes, alignment)
        object.__setattr__(self, "w2_offset", w2_offset)
        object.__setattr__(self, "payload_nbytes", w13_bytes + w2_bytes)
        object.__setattr__(self, "extent_nbytes", round_up(w2_offset + w2_bytes, alignment))


@dataclass(frozen=True, slots=True)
class ExpertTensorViews:
    w13: torch.Tensor
    w2: torch.Tensor


@dataclass(frozen=True, slots=True)
class ExpertProjectionTensorViews:
    gate: torch.Tensor
    up: torch.Tensor
    down: torch.Tensor


def projection_expert_views(
    arena: torch.Tensor, block_ids: tuple[int, int, int], page_bytes: int, layout: ExpertMemoryLayout,
) -> ExpertProjectionTensorViews:
    """Map gate/down/up block IDs to ND views without allocating weight storage."""
    gate_block, down_block, up_block = block_ids
    hidden_size, intermediate_size = layout.hidden_size, layout.intermediate_size
    matrix_bytes = hidden_size * intermediate_size
    return ExpertProjectionTensorViews(
        arena.narrow(0, gate_block * page_bytes, matrix_bytes).view(torch.int8).view(hidden_size, intermediate_size),
        arena.narrow(0, up_block * page_bytes, matrix_bytes).view(torch.int8).view(hidden_size, intermediate_size),
        arena.narrow(0, down_block * page_bytes, matrix_bytes).view(torch.int8).view(intermediate_size, hidden_size),
    )


def fused_expert_views(
    arena: torch.Tensor, first_block_id: int, page_bytes: int, layout: ExpertMemoryLayout,
) -> ExpertTensorViews:
    """View a consecutive warmup slot while checkpoint sources still use W13."""
    offset = first_block_id * page_bytes
    hidden_size, intermediate_size = layout.hidden_size, layout.intermediate_size
    return ExpertTensorViews(
        arena.narrow(0, offset, hidden_size * 2 * intermediate_size)
        .view(torch.int8).view(hidden_size, 2 * intermediate_size),
        arena.narrow(0, offset + layout.w2_offset, hidden_size * intermediate_size)
        .view(torch.int8).view(intermediate_size, hidden_size),
    )
