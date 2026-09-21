# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Worker views and metadata for independently allocated GDN states.

The scheduler still owns logical Mamba blocks. Its block IDs are translated
only at the Ascend metadata boundary, where convolution and recurrent attention
already consume separate index tensors. No vLLM scheduler schema is changed.
"""

from copy import copy
from math import prod

import torch

_INDEX_DTYPES = (torch.int32, torch.int64)
_STATE_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def _validate_state_shapes(shapes, dtypes) -> None:
    if not isinstance(shapes, tuple) or len(shapes) != 2:
        raise ValueError("GDN state pages require a (conv_shape, ssm_shape) tuple.")
    if not isinstance(dtypes, tuple) or len(dtypes) != 2:
        raise ValueError("GDN state pages require separate Conv and SSM dtypes.")
    for shape, ndim in zip(shapes, (2, 3)):
        if (
            not isinstance(shape, tuple)
            or len(shape) != ndim
            or any(not isinstance(dim, int) or isinstance(dim, bool) or dim <= 0 for dim in shape)
        ):
            raise ValueError("GDN native shapes must be Conv (state_len, dim) and SSM (heads, value, key).")
    if any(dtype not in _STATE_DTYPES for dtype in dtypes):
        raise ValueError("GDN state pages support float16, bfloat16, or float32 state dtypes.")


def validate_gdn_state_page_config(vllm_config, specs=(), *, conv_state_layout="SD") -> None:
    """Validate GDN cache modes and native state layouts.

    OMoEConfig.validate checks common engine execution constraints.
    The caller supplies the resolved Conv layout (from get_conv_state_layout),
    because the AscendC kernel requires state_len before the channel dimension.
    Validation belongs at initialization, before any shared state is allocated.
    """
    cache_config = vllm_config.cache_config
    if cache_config.mamba_cache_mode != "none":
        raise ValueError("O-MoE GDN state pages currently require mamba_cache_mode='none'.")
    if cache_config.enable_prefix_caching:
        raise ValueError("O-MoE GDN state pages currently require prefix caching to be disabled.")
    if conv_state_layout != "SD":
        raise ValueError("O-MoE GDN state pages require the native Ascend Conv layout 'SD'.")
    for spec in specs:
        if getattr(getattr(spec, "mamba_type", None), "name", None) != "GDN_ATTN":
            raise ValueError("O-MoE separate state pages support only GDN Mamba groups.")
        if spec.num_speculative_blocks != 0 or spec.mamba_cache_mode != "none":
            raise ValueError("O-MoE GDN MambaSpec requires no speculative blocks and cache mode 'none'.")
        _validate_state_shapes(spec.shapes, spec.dtypes)


def make_gdn_state_views(
    arena: torch.Tensor, shapes, dtypes, *, conv_stride_bytes=None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return Conv/SSM address views over one arena without allocating storage.

    Conv's leading stride encodes allocated start addresses; each state still
    consists of consecutive native rows. These views may overlap virtually,
    but only disjoint allocator grants may be read or written. SSM retains
    its native contiguous layout. Omit the stride for ordinary Conv pages.
    """
    if arena.dtype not in (torch.int8, torch.uint8) or arena.ndim != 1 or not arena.is_contiguous():
        raise ValueError("GDN state arena must be a contiguous one-dimensional byte tensor.")
    views = []
    for state_id, (shape, dtype) in enumerate(zip(shapes, dtypes)):
        element_size = torch.empty((), dtype=dtype).element_size()
        state_nbytes = prod(shape) * element_size
        stride_bytes = conv_stride_bytes if state_id == 0 and conv_stride_bytes is not None else state_nbytes
        if stride_bytes <= 0 or stride_bytes % element_size:
            raise ValueError("GDN state stride must be positive and aligned to its dtype.")
        count = (arena.numel() - state_nbytes) // stride_bytes + 1
        if count <= 0 or arena.data_ptr() % element_size:
            raise ValueError("GDN state arena must fit an aligned native state of each type.")
        elements = arena[: arena.numel() // element_size * element_size].view(dtype)
        strides = (stride_bytes // element_size, *(prod(shape[i + 1 :]) for i in range(len(shape))))
        views.append(elements.as_strided((count, *shape), strides))
    return views[0], views[1]


def _remap_indices(indices: torch.Tensor | None, index_map: torch.Tensor) -> torch.Tensor | None:
    if indices is None:
        return None
    if indices.dtype not in _INDEX_DTYPES or indices.device != index_map.device:
        raise ValueError("GDN state indices must be int32/int64 and on the index map device.")
    # PAD_SLOT_ID is negative; preserve it without ever indexing the final LUT
    # entry. Logical null block zero is resolved through the caller's null entry.
    # The scheduler/worker ownership protocol validates live logical IDs before
    # launch. This path performs no .item(), device synchronization, or D2H copy.
    mapped = torch.index_select(index_map, 0, indices.clamp_min(0).reshape(-1).long())
    mapped = mapped.reshape(indices.shape).to(indices.dtype)
    return torch.where(indices < 0, indices, mapped)


def remap_gdn_state_metadata(metadata, conv_index_map: torch.Tensor, ssm_index_map: torch.Tensor):
    """Return layer-local metadata using independently allocated state IDs.

    Maps are per-runner device tensors indexed by the unchanged logical Mamba
    block IDs. They must be refreshed from the scheduler's current allocations
    before building attention metadata. Inactive/null entries and their owned
    storage are the caller's responsibility. The input metadata and attached
    Conv structures remain untouched, including when shared by several layers.
    """
    if (
        getattr(metadata, "num_spec_decodes", 0)
        or getattr(metadata, "spec_sequence_masks", None) is not None
        or getattr(metadata, "spec_state_indices_tensor", None) is not None
    ):
        raise ValueError("O-MoE GDN separate state metadata does not support speculative decoding.")

    result = copy(metadata)
    for name in ("non_spec_state_indices_tensor", "prefill_state_indices"):
        setattr(result, name, _remap_indices(getattr(metadata, name, None), ssm_index_map))

    for name in ("non_spec_prefill_metadata", "non_spec_decode_metadata"):
        branch = getattr(metadata, name, None)
        if branch is None:
            continue
        mapped_branch = copy(branch)
        mapped_conv = copy(branch.causal_conv1d)
        mapped_conv.cache_indices = _remap_indices(branch.causal_conv1d.cache_indices, conv_index_map)
        mapped_branch.causal_conv1d = mapped_conv
        setattr(result, name, mapped_branch)
    return result
