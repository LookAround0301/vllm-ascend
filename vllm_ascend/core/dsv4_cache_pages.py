# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
DS V4 cache payload schemas and arena address views.

Allocation extents, local-L2 page ordinals, and backend address IDs are distinct.
Indexer page extents follow the native block size while backend IDs address
64-byte units. Its overlapping virtual views MUST NOT be budgeted by numel(),
zeroed wholesale, or used with int32 scatter indices. Never materialize a
complete virtual view with clone()/contiguous(): its virtual element count
greatly exceeds the arena storage. Copy only bounded, allocator-authorized
physical page slices. Only an allocator-owned extent may be cleared or written.

This module does not allocate pages, reserve null memory, or alter lifetimes.
The scheduler retains complete schemas before reducing UniformType specs to a
representative; workers validate that schema against their cache-name groups.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

INDEXER_ADDRESS_QUANTUM_BYTES = 64
BACKEND_INT32_MAX = (1 << 31) - 1


def _dtype_bytes(name: str) -> int:
    sizes = {"int8": 1, "float16": 2, "bfloat16": 2, "float32": 4}
    return sizes[name]


def _virtual_view_count(page: DSV4CachePage, arena_nbytes: int) -> int:
    """Bound native QLI's signed-INT32 shape despite int64 scatter indices."""
    if arena_nbytes < page.allocation_nbytes:
        raise ValueError("The arena cannot hold a complete cache page")
    count = (arena_nbytes - page.allocation_nbytes) // page.address_quantum_bytes + 1
    if count > BACKEND_INT32_MAX:
        raise ValueError("DS V4 virtual view leading dimension exceeds the native signed INT32 limit")
    return count


@dataclass(frozen=True)
class DSV4CachePage:
    """One cache's native payload plus its unmodified scheduler lifecycle.

    ``compress_ratio`` is the *spec's scheduler ratio*: compressor state specs
    have ratio 1, even when their semantic kind names the C4/C128 compressor.
    ``head_size`` is the actual cache tensor's final dimension, including all
    KV/score and overlap state components, not the attention head dimension.
    """

    name: str
    group_id: int
    kind: str
    allocation_nbytes: int
    block_size: int
    head_size: int
    dtype_name: str
    address_quantum_bytes: int
    scale_dim: int = 0
    scale_dtype_name: str | None = None
    sliding_window: int | None = None
    compress_ratio: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, fields: Mapping[str, Any]) -> DSV4CachePage:
        """Restore the schema captured from native cache specs."""
        return cls(**fields)


def _runtime_spec_types() -> tuple[type, type, type]:
    # Imports are lazy: byte/address validation and wire schemas are usable on
    # the controller without importing torch or initializing vLLM platforms.
    from vllm.v1.kv_cache_interface import UniformTypeKVCacheSpecs

    from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec, AscendSlidingWindowMLASpec

    return UniformTypeKVCacheSpecs, AscendMLAAttentionSpec, AscendSlidingWindowMLASpec


def _page_from_spec(name: str, group_id: int, spec: Any, spec_types: tuple[type, type, type]) -> DSV4CachePage:
    _, mla_type, sliding_type = spec_types
    if type(spec) not in (mla_type, sliding_type):
        raise ValueError(f"Unsupported DS V4 spec class for {name}: {type(spec).__name__}")
    if spec.num_kv_heads != 1:
        raise ValueError(f"DS V4 cache {name} requires one KV head")
    if getattr(spec, "kv_quant_mode", 0) != 0 or getattr(spec, "indexes_kv_by_block_stride", False):
        raise ValueError(f"Unsupported quantization or packed-block stride for {name}")
    if getattr(spec, "cache_sparse_sfa_c8", False):
        raise ValueError(f"Sparse C8 cache layout is unsupported for {name}")
    dtype_name = str(spec.dtype).removeprefix("torch.")
    window = getattr(spec, "sliding_window", None)
    ratio = getattr(spec, "compress_ratio", 1)
    scale_dim = getattr(spec, "scale_dim", 0)
    scale_dtype = str(spec.scale_dtype).removeprefix("torch.") if scale_dim else None
    model_version = getattr(spec, "model_version", None)
    if type(spec) is mla_type:
        if model_version != "deepseek_v4" or window is not None:
            raise ValueError(f"Cache {name} is not a supported DS V4 compressed KV spec")
        kind = "indexer_kv" if dtype_name == "int8" else "main_kv"
    elif dtype_name == "bfloat16":
        if model_version != "deepseek_v4":
            raise ValueError(f"Cache {name} is not a DS V4 SWA spec")
        kind = "main_kv"
    elif dtype_name == "float32" and model_version in (None, "deepseek_v4"):
        # The current state spec has no semantic role field. These payload widths
        # identify the validated DS V4 states only; they must not classify other models.
        kinds = {2048: "compressor_c4", 512: "compressor_indexer_c4", 1024: "compressor_c128"}
        kind = kinds[spec.head_size]
    else:
        raise ValueError(f"Unsupported DS V4 cache dtype/model for {name}")
    nbytes = spec.block_size * (
        spec.head_size * _dtype_bytes(dtype_name) + scale_dim * (_dtype_bytes(scale_dtype) if scale_dtype else 0)
    )
    if getattr(spec, "real_page_size_bytes", nbytes) != nbytes:
        raise ValueError(f"Unexpected native payload size for {name}")
    return DSV4CachePage(
        name=name,
        group_id=group_id,
        kind=kind,
        allocation_nbytes=nbytes,
        block_size=spec.block_size,
        head_size=spec.head_size,
        dtype_name=dtype_name,
        address_quantum_bytes=INDEXER_ADDRESS_QUANTUM_BYTES if kind == "indexer_kv" else nbytes,
        scale_dim=scale_dim,
        scale_dtype_name=scale_dtype,
        sliding_window=window,
        compress_ratio=ratio,
    )


def page_schemas_from_groups(groups: Sequence[Any]) -> tuple[DSV4CachePage, ...]:
    """Capture every per-cache spec before scheduler representative collapse.

    No cache is invented, and no layer number or group count is assumed. The
    currently supported model normally supplies six groups; group registration
    order and each group's name order are retained exactly.
    """
    spec_types = _runtime_spec_types()
    uniform_type = spec_types[0]
    result = []
    for group_id, group in enumerate(groups):
        outer = group.kv_cache_spec
        if not isinstance(outer, uniform_type):
            raise ValueError("Full per-cache UniformType specs are required; do not reconstruct from a representative")
        result.extend(
            _page_from_spec(name, group_id, outer.kv_cache_specs[name], spec_types)
            for name in group.layer_names
        )
    return tuple(result)


def validate_scheduler_groups(schemas: Sequence[DSV4CachePage | Mapping[str, Any]], groups: Sequence[Any]) -> None:
    """Keep the original per-cache geometry when scheduler specs collapse.

    Native grouping already validates names and common lifetimes. Check that
    the serialized schemas still match those groups, without rebuilding each
    schema from the representative's potentially different payload width.
    """
    pages = tuple(page if isinstance(page, DSV4CachePage) else DSV4CachePage.from_dict(page) for page in schemas)
    expected_groups = {name: group_id for group_id, group in enumerate(groups) for name in group.layer_names}
    if {page.name: page.group_id for page in pages} != expected_groups:
        raise ValueError("Scheduler cache names/group IDs do not match the complete schemas")
    for page in pages:
        spec = groups[page.group_id].kv_cache_spec
        if hasattr(spec, "kv_cache_specs"):
            spec = spec.kv_cache_specs[page.name]
        if (page.block_size, page.sliding_window, page.compress_ratio) != (
            spec.block_size, getattr(spec, "sliding_window", None), getattr(spec, "compress_ratio", 1)
        ):
            raise ValueError(f"Scheduler lifecycle changed for {page.name}")


def make_dsv4_cache_views(arena: torch.Tensor, page: DSV4CachePage) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Make zero-copy address views, without allocating or zeroing storage.

    Indexer uses a 64B address quantum: each owned native page becomes
    a key view and a scale view sharing one address ID.
    Its leading dimension counts virtual *addresses*, NOT allocatable pages.
    Adjacent virtual entries overlap; only IDs encoding authorized L2-local
    page starts may reach the backend. Scatter consumers MUST use int64 indices
    because their sorting/address intermediate can overflow int32.
    Never clone or make the whole virtual view contiguous; materialize only
    bounded slices of allocator-authorized physical pages when a copy is needed.
    """
    import torch

    nbytes = arena.numel()
    count = _virtual_view_count(page, nbytes)

    def view(dtype_name: str, width: int, payload_offset: int = 0) -> torch.Tensor:
        itemsize = _dtype_bytes(dtype_name)
        # Cropping a possible untyped tail is a view, not a copy. Respect an
        # arena slice's storage_offset when forming the scale sub-view.
        typed = arena[: nbytes // itemsize * itemsize].view(getattr(torch, dtype_name))
        return typed.as_strided(
            (count, page.block_size, 1, width),
            (page.address_quantum_bytes // itemsize, width, width, 1),
            typed.storage_offset() + payload_offset // itemsize,
        )

    primary = view(page.dtype_name, page.head_size)
    if not page.scale_dim:
        return primary
    scale_offset = page.block_size * page.head_size * _dtype_bytes(page.dtype_name)
    return primary, view(page.scale_dtype_name, page.scale_dim, scale_offset)
