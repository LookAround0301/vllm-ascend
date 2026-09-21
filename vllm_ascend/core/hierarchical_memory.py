# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Shared byte geometry for expert, KV and optional Conv/SSM ownership.

The descriptor covers one arena. Ordinary KV/GDN uses singleton cache groups;
DS V4 retains its native composite groups and maps each logical block to a
separate physical page for every cache in that group. The hierarchical
allocator grants nonoverlapping typed pages, without padding a cache state
to the size of another state type.
"""

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass, replace
from math import gcd, lcm, prod

from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import get_dtype_size

from vllm_ascend.expert_offload.expert_memory import EXPERT_ALIGNMENT, ExpertMemoryLayout

from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec, KVCacheTensor, MambaSpec
from vllm_ascend.core.hierarchical_block_pool import HierarchicalBlockPool
from vllm_ascend.expert_offload.config import minimum_expert_slots

HIERARCHICAL_MODEL_TYPES = ("qwen3_moe", "qwen3_5_moe_text", "deepseek_v4", "minimax_m2")


def hierarchical_omoe_enabled(vllm_config) -> bool:
    additional_config = getattr(vllm_config, "additional_config", None) or {}
    omoe_config = additional_config.get("omoe_config", {})
    model_config = getattr(vllm_config, "model_config", None)
    text_config = getattr(model_config, "hf_text_config", None)
    return (
        isinstance(omoe_config, dict)
        and omoe_config.get("enabled") is True
        and getattr(text_config, "model_type", None) in HIERARCHICAL_MODEL_TYPES
    )


def validate_hierarchical_cache_config(vllm_config, mamba_specs=()):
    model_type = vllm_config.model_config.hf_text_config.model_type
    omoe_config = (getattr(vllm_config, "additional_config", None) or {}).get("omoe_config", {})
    if not (
        isinstance(omoe_config, dict) and omoe_config.get("enabled") is True
        and model_type in HIERARCHICAL_MODEL_TYPES
    ):
        raise ValueError("Hierarchical cache geometry requires an explicitly enabled supported O-MoE model")
    cache_config = vllm_config.cache_config
    if cache_config.enable_prefix_caching:
        raise ValueError("Hierarchical O-MoE requires prefix caching to be disabled")
    if getattr(cache_config, "num_gpu_blocks_override", None) is not None:
        raise ValueError("Hierarchical O-MoE does not support num_gpu_blocks_override")
    if getattr(vllm_config.model_config, "original_max_model_len", None) == -1:
        raise ValueError("Hierarchical O-MoE requires an explicit max_model_len")
    if model_type == "deepseek_v4":
        if getattr(vllm_config.scheduler_config, "disable_hybrid_kv_cache_manager", False):
            raise ValueError("Hierarchical DS V4 must retain the native hybrid cache lifetimes")
    if vllm_config.model_config.hf_text_config.model_type == "qwen3_5_moe_text":
        from vllm.model_executor.layers.mamba.mamba_utils import get_conv_state_layout
        from vllm_ascend.core.gdn_state_pages import validate_gdn_state_page_config
        validate_gdn_state_page_config(vllm_config, mamba_specs, conv_state_layout=get_conv_state_layout())


def get_cache_group_geometry(vllm_config, groups):
    layer_names, group_kinds, full_attention_specs, mamba_specs = [], [], [], []
    for group in groups:
        if len(group.layer_names) != 1 or getattr(group, "is_eagle_group", False):
            raise ValueError("Hierarchical O-MoE requires singleton non-MTP cache groups")
        name = group.layer_names[0]
        if not isinstance(name, str) or not name or name in layer_names:
            raise ValueError("Hierarchical cache group names must be nonempty and unique")
        layer_names.append(name)
        spec = group.kv_cache_spec
        if type(spec) is FullAttentionSpec:
            if (
                spec.page_size_padded is not None
                or spec.head_size_v != spec.head_size
                or spec.sliding_window is not None
                or spec.attention_chunk_size is not None
                or spec.non_causal
                or spec.indexes_kv_by_block_stride
                or getattr(spec.kv_quant_mode, "name", None) != "NONE"
                or str(spec.dtype) not in ("torch.bfloat16", "bfloat16")
            ):
                raise ValueError(
                    "Hierarchical KV requires unpadded BF16 causal full attention"
                )
            full_attention_specs.append(spec)
            group_kinds.append("kv")
        elif type(spec) is MambaSpec:
            if spec.page_size_padded is not None:
                raise ValueError("Hierarchical GDN state specs must not contain shared-layout padding")
            if spec.mamba_cache_mode == "none" and spec.block_size != vllm_config.model_config.max_model_len:
                raise ValueError(
                    "Hierarchical GDN cache mode 'none' requires MambaSpec.block_size == max_model_len "
                    "for one state per layer per request"
                )
            mamba_specs.append(spec)
            group_kinds.append("mamba")
        else:
            raise ValueError(f"Unsupported hierarchical cache spec for {name}: {type(spec).__name__}")
    model_type = vllm_config.model_config.hf_text_config.model_type
    if not full_attention_specs:
        raise ValueError("Hierarchical geometry requires full-attention groups")
    if model_type == "qwen3_5_moe_text" and not mamba_specs:
        raise ValueError("Qwen3.5 hierarchical geometry needs both full-attention and GDN groups")
    if model_type in ("qwen3_moe", "minimax_m2") and mamba_specs:
        raise ValueError("Qwen3 MoE and MiniMax M2 hierarchical geometry supports only full-attention groups")
    if any(spec != full_attention_specs[0] for spec in full_attention_specs) or any(
        spec != mamba_specs[0] for spec in mamba_specs
    ):
        raise ValueError("Each hierarchical state type must have identical geometry across layers")
    expected_layers = getattr(vllm_config.model_config.hf_text_config, "num_hidden_layers", None)
    if expected_layers is not None and len(layer_names) != expected_layers:
        raise ValueError("Hierarchical cache groups must cover every decoder layer exactly once")
    validate_hierarchical_cache_config(vllm_config, mamba_specs)
    if not mamba_specs:
        return tuple(layer_names), tuple(group_kinds), full_attention_specs[0].page_size_bytes, None, 1, None
    conv_shape, ssm_shape = mamba_specs[0].shapes
    conv_dtype, ssm_dtype = mamba_specs[0].dtypes
    return (
        tuple(layer_names),
        tuple(group_kinds),
        full_attention_specs[0].page_size_bytes,
        prod(conv_shape) * get_dtype_size(conv_dtype),
        conv_shape[0],
        prod(ssm_shape) * get_dtype_size(ssm_dtype),
    )


@dataclass(frozen=True)
class HierarchicalMemoryLayout:
    num_blocks: int
    nbytes: int
    expert_page_bytes: int
    kv_page_bytes: int
    ssm_page_bytes: int | None
    conv_page_bytes: int | None
    conv_rows: int
    huge_page_bytes: int
    local_experts_per_layer: int
    expert_extent_nbytes: int
    expert_span_pages: int
    bootstrap_expert_pages: int
    layer_names: tuple[str, ...]
    group_kinds: tuple[str, ...]
    # Complete per-cache geometry survives the scheduler's representative-spec
    # reduction through the existing all-rank pool-info handshake, not globals.
    cache_schemas: tuple = ()

    @classmethod
    def build(
        cls,
        *,
        nbytes,
        expert_page_bytes,
        kv_page_bytes,
        ssm_page_bytes,
        conv_page_bytes,
        conv_rows,
        local_experts_per_layer,
        expert_extent_nbytes,
        layer_names=(),
        group_kinds=(),
    ):
        """Validate byte geometry without importing model or device modules."""
        # Model dimensions are validated by the configuration entry point.
        # Keep the page constraints needed by the byte geometry here.
        if any(
            size <= 0
            for size in (expert_page_bytes, kv_page_bytes, ssm_page_bytes, conv_page_bytes, conv_rows)
            if size is not None
        ):
            raise ValueError("Page sizes and Conv row count must be positive")
        if (ssm_page_bytes is None) != (conv_page_bytes is None):
            raise ValueError("Conv and SSM geometry must either both be present or both be absent")
        if ssm_page_bytes is None and conv_rows != 1:
            raise ValueError("Absent Conv state geometry requires conv_rows=1")
        huge_page_bytes = lcm(expert_page_bytes, ssm_page_bytes) if ssm_page_bytes is not None else expert_page_bytes
        if nbytes % huge_page_bytes or nbytes < 2 * huge_page_bytes:
            raise ValueError("Arena bytes must contain whole Huge pages plus usable space after null Huge0")
        if any(size % kv_page_bytes for size in (expert_page_bytes, ssm_page_bytes) if size is not None):
            raise ValueError("KV pair bytes must exactly divide expert and SSM page bytes")
        if conv_page_bytes is not None and conv_page_bytes % conv_rows:
            raise ValueError("Conv page bytes must contain complete native state rows")
        if conv_page_bytes is not None and conv_page_bytes > max(expert_page_bytes, ssm_page_bytes):
            raise ValueError("A native Conv state must fit in an L2 page")
        if expert_page_bytes % EXPERT_ALIGNMENT:
            raise ValueError(f"Expert projection bytes must preserve the {EXPERT_ALIGNMENT}-byte expert alignment")
        span_pages = cdiv(expert_extent_nbytes, expert_page_bytes)
        bootstrap_pages = span_pages * local_experts_per_layer
        if huge_page_bytes + bootstrap_pages * expert_page_bytes >= nbytes:
            raise ValueError("Arena must fit null Huge0, every local routed expert, and request state space")
        return cls(
            nbytes // kv_page_bytes,
            nbytes,
            expert_page_bytes,
            kv_page_bytes,
            ssm_page_bytes,
            conv_page_bytes,
            conv_rows,
            huge_page_bytes,
            local_experts_per_layer,
            expert_extent_nbytes,
            span_pages,
            bootstrap_pages,
            tuple(layer_names),
            tuple(group_kinds),
        )

    @classmethod
    def from_config(cls, vllm_config, kv_config, *, cache_schemas=None):
        if vllm_config.model_config.hf_text_config.model_type == "deepseek_v4":
            return cls.from_dsv4_config(vllm_config, kv_config, cache_schemas=cache_schemas)
        if cache_schemas:
            raise ValueError("DS cache schemas cannot be installed on a non-DS model")
        layer_names, group_kinds, kv_bytes, conv_bytes, conv_rows, ssm_bytes = get_cache_group_geometry(
            vllm_config,
            kv_config.kv_cache_groups,
        )
        if len(kv_config.kv_cache_tensors) != 1:
            raise ValueError("Hierarchical O-MoE requires exactly one whole-arena descriptor")
        arena_descriptor = kv_config.kv_cache_tensors[0]
        if arena_descriptor.offset or arena_descriptor.block_stride:
            raise ValueError("Hierarchical arena descriptors must have zero offset and block_stride")
        if len(arena_descriptor.shared_by) != len(layer_names) or set(arena_descriptor.shared_by) != set(layer_names):
            raise ValueError("The whole-arena descriptor must list each cache group name exactly once")
        text_config = vllm_config.model_config.hf_text_config
        hidden_size = text_config.hidden_size
        intermediate_size = getattr(text_config, "moe_intermediate_size", None) or text_config.intermediate_size
        local_expert_count = vllm_config.model_config.get_num_experts()
        parallel_config = vllm_config.parallel_config
        # TP keeps every expert and shards its intermediate dimension.
        intermediate_size //= parallel_config.tensor_parallel_size
        expert_layout = ExpertMemoryLayout(hidden_size, intermediate_size, alignment=EXPERT_ALIGNMENT)
        result = cls.build(
            nbytes=arena_descriptor.size,
            expert_page_bytes=hidden_size * intermediate_size,
            kv_page_bytes=kv_bytes,
            ssm_page_bytes=ssm_bytes,
            conv_page_bytes=conv_bytes,
            conv_rows=conv_rows,
            local_experts_per_layer=local_expert_count,
            expert_extent_nbytes=expert_layout.extent_nbytes,
            layer_names=layer_names,
            group_kinds=group_kinds,
        )
        if kv_config.num_blocks != result.num_blocks:
            raise ValueError("Scheduler block capacity must equal whole-arena bytes divided by full KV pair bytes")
        return result

    @classmethod
    def from_dsv4_config(cls, vllm_config, kv_config, *, cache_schemas=None):
        """Bind original logical groups to one arena of independently typed pages.

        A scheduler config only retains each group's representative spec. Its
        complete schema must therefore come from the workers' checked handshake;
        never interpret that representative's head size as the entire group.
        ``num_blocks`` bounds logical blocks, not the Indexer's virtual shape.
        """
        from vllm_ascend.core.dsv4_cache_pages import (
            DSV4CachePage,
            page_schemas_from_groups,
            validate_scheduler_groups,
        )

        validate_hierarchical_cache_config(vllm_config)
        if cache_schemas is None:
            pages = page_schemas_from_groups(kv_config.kv_cache_groups)
        else:
            pages = tuple(
                page if isinstance(page, DSV4CachePage) else DSV4CachePage.from_dict(page) for page in cache_schemas
            )
            validate_scheduler_groups(pages, kv_config.kv_cache_groups)
        layer_names = tuple(page.name for page in pages)
        if len(kv_config.kv_cache_tensors) != 1:
            raise ValueError("Hierarchical DS V4 requires exactly one whole-arena descriptor")
        arena_descriptor = kv_config.kv_cache_tensors[0]
        if (
            arena_descriptor.offset
            or arena_descriptor.block_stride
            or len(arena_descriptor.shared_by) != len(layer_names)
            or set(arena_descriptor.shared_by) != set(layer_names)
        ):
            raise ValueError("DS whole-arena descriptor must cover every real cache exactly once")
        text_config = vllm_config.model_config.hf_text_config
        hidden_size = text_config.hidden_size
        intermediate_size = text_config.moe_intermediate_size
        local_expert_count = text_config.n_routed_experts
        parallel_config = vllm_config.parallel_config
        # TP keeps every expert and shards its intermediate dimension.
        intermediate_size //= parallel_config.tensor_parallel_size
        expert = ExpertMemoryLayout(hidden_size, intermediate_size, alignment=EXPERT_ALIGNMENT)
        main_sizes = {page.allocation_nbytes for page in pages if page.kind == "main_kv"}
        if len(main_sizes) != 1:
            raise ValueError("DS V4 requires one supported main/SWA KV page geometry")
        # Reuse expert/Huge validation; DS logical groups are composite, not
        # singleton KV or Mamba groups. No fictitious SSM page enlarges Huge.
        base = cls.build(
            nbytes=arena_descriptor.size,
            expert_page_bytes=hidden_size * intermediate_size,
            kv_page_bytes=main_sizes.pop(),
            ssm_page_bytes=None,
            conv_page_bytes=None,
            conv_rows=1,
            local_experts_per_layer=local_expert_count,
            expert_extent_nbytes=expert.extent_nbytes,
        )
        group_bytes = Counter()
        for page in pages:
            group_bytes[page.group_id] += page.allocation_nbytes
            if page.allocation_nbytes > base.expert_page_bytes or base.expert_page_bytes % page.address_quantum_bytes:
                raise ValueError("DS typed pages must fit and preserve backend alignment within expert L2")
        # Even if every byte belonged to the smallest compound group, there
        # cannot be more live blocks than this bound. Null block 0 is extra;
        # actual admission still uses the allocator and minimum expert set.
        capacity = 1 + arena_descriptor.size // min(group_bytes.values())
        result = replace(
            base,
            num_blocks=capacity,
            layer_names=layer_names,
            group_kinds=("dsv4",) * len(kv_config.kv_cache_groups),
            cache_schemas=pages,
        )
        if kv_config.num_blocks != capacity:
            raise ValueError("DS logical block capacity differs from the whole-arena schema bound")
        return result

    @property
    def index_map_nbytes(self):
        """Persistent device mapping bytes; virtual view numel is never storage."""
        if self.ssm_page_bytes is not None:
            return 2 * self.num_blocks * 8  # Conv and SSM maps are INT64.
        return len(self.cache_schemas) * self.num_blocks * 4

    @property
    def conv_address_bytes(self):
        return gcd(
            self.conv_page_bytes // self.conv_rows,
            self.expert_page_bytes,
            self.ssm_page_bytes or self.expert_page_bytes,
        )

    def allocator_kwargs(self):
        result = {
            "total_bytes": self.nbytes,
            "expert_page_bytes": self.expert_page_bytes,
            "kv_page_bytes": self.kv_page_bytes,
            "ssm_page_bytes": self.ssm_page_bytes,
            "conv_page_bytes": self.conv_page_bytes,
            "conv_rows": self.conv_rows,
            "alignment_bytes": 4,
        }
        if self.cache_schemas:
            from vllm_ascend.core.hierarchical_block_pool import LocalPageGeometry

            result["local_page_geometries"] = {
                page.kind: LocalPageGeometry(page.allocation_nbytes, page.address_quantum_bytes)
                for page in self.cache_schemas
            }
        return result

    def allocator_wire_kwargs(self):
        """Primitive-only pool description for worker RPC and equality checks."""
        result = self.allocator_kwargs()
        if "local_page_geometries" in result:
            result["local_page_geometries"] = {
                kind: [geometry.allocation_nbytes, geometry.alignment_bytes]
                for kind, geometry in result["local_page_geometries"].items()
            }
        return result


def build_hierarchical_kv_cache_configs(vllm_config, kv_cache_specs, available_memory):
    """Plan one common arena size per rank with model-specific logical groups.

    Ordinary KV/GDN uses singleton groups; DS V4 keeps its native composite
    groups while budgeting independent typed payloads and their index maps.
    """

    validate_hierarchical_cache_config(vllm_config)
    if not kv_cache_specs or len(kv_cache_specs) != len(available_memory):
        raise ValueError("Worker cache specs and memory budgets must have the same nonzero rank count")
    if len(kv_cache_specs) != vllm_config.parallel_config.tensor_parallel_size:
        raise ValueError("Hierarchical O-MoE cache planning requires one spec and memory budget per TP rank")
    if vllm_config.model_config.hf_text_config.model_type == "deepseek_v4":
        return build_dsv4_kv_cache_configs(vllm_config, kv_cache_specs, available_memory)
    unpadded_rank_specs = []
    for rank_specs in kv_cache_specs:
        specs = {}
        for name, spec in rank_specs.items():
            if type(spec) in (FullAttentionSpec, MambaSpec):
                specs[name] = replace(spec, page_size_padded=None)
            else:
                raise ValueError(f"Unsupported hierarchical cache spec: {type(spec).__name__}")
        unpadded_rank_specs.append(specs)
    if any(specs != unpadded_rank_specs[0] for specs in unpadded_rank_specs[1:]):
        raise ValueError("Hierarchical cache names and typed geometry must match across all TP ranks")
    ordered_layer_names = sorted(
        unpadded_rank_specs[0], key=lambda name: (type(unpadded_rank_specs[0][name]) is MambaSpec, name)
    )
    groups = [KVCacheGroupSpec([name], unpadded_rank_specs[0][name]) for name in ordered_layer_names]
    _, _, kv_bytes, _, _, ssm_bytes = get_cache_group_geometry(vllm_config, groups)
    text_config = vllm_config.model_config.hf_text_config
    intermediate_size = getattr(text_config, "moe_intermediate_size", None) or text_config.intermediate_size
    expert_bytes = text_config.hidden_size * intermediate_size
    expert_bytes //= vllm_config.parallel_config.tensor_parallel_size
    huge_bytes = lcm(expert_bytes, ssm_bytes) if ssm_bytes is not None else expert_bytes
    # vLLM supplies a cache budget; O-MoE uses it for the whole shared arena,
    # including expert pages, and keeps identical byte geometry on all ranks.
    budget = min(available_memory)
    # Conv/SSM maps live outside the arena. Each Huge adds an integral
    # number of logical KV blocks and two INT64 entries per logical block.
    map_bytes_per_huge = 2 * (huge_bytes // kv_bytes) * 8 if ssm_bytes is not None else 0
    arena_bytes = budget // (huge_bytes + map_bytes_per_huge) * huge_bytes
    configurations = []
    for specs in unpadded_rank_specs:
        config = KVCacheConfig(
            num_blocks=arena_bytes // kv_bytes,
            kv_cache_tensors=[KVCacheTensor(size=arena_bytes, shared_by=list(ordered_layer_names))],
            kv_cache_groups=[KVCacheGroupSpec([name], specs[name]) for name in ordered_layer_names],
        )
        layout = HierarchicalMemoryLayout.from_config(vllm_config, config)
        pool = HierarchicalBlockPool(**layout.allocator_kwargs())
        pool.reserve_at("huge", (0,))
        max_model_len = vllm_config.model_config.max_model_len
        request_counts = {
            "expert": minimum_expert_slots(vllm_config, layout.local_experts_per_layer) * layout.expert_span_pages,
            "kv": sum(
                cdiv(max_model_len, spec.block_size) for spec in specs.values() if type(spec) is FullAttentionSpec
            ),
            "conv": layout.group_kinds.count("mamba"),
            "ssm": layout.group_kinds.count("mamba"),
        }
        if not pool.can_reserve(request_counts):
            raise ValueError("Arena cannot fit max_model_len request state and every local expert after null Huge0")
        configurations.append(config)
    return configurations


def build_dsv4_kv_cache_configs(vllm_config, kv_cache_specs, available_memory):
    """Keep native grouping, but budget actual payloads plus mapping tables.

    Native grouping may pad shared tensors. Work on private spec copies, retain
    the resulting logical group membership, then remove that physical padding.
    A binary search reserves persistent index maps *outside* the arena from the
    same available-memory budget. Expert storage remains *inside* the arena.
    """
    from vllm.v1.core.kv_cache_utils import get_kv_cache_groups
    from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec, KVCacheTensor, UniformTypeKVCacheSpecs

    from vllm_ascend.core.dsv4_cache_pages import page_schemas_from_groups
    from vllm_ascend.core.hierarchical_block_pool import HierarchicalBlockPool
    from vllm_ascend.expert_offload.config import minimum_expert_slots

    rank_groups = []
    for specs in kv_cache_specs:
        groups = []
        for group in get_kv_cache_groups(vllm_config, deepcopy(specs)):
            group_cache_specs = getattr(group.kv_cache_spec, "kv_cache_specs", None)
            if group_cache_specs is None:
                unpadded_spec = replace(group.kv_cache_spec, page_size_padded=None)
            else:
                unpadded_spec = UniformTypeKVCacheSpecs.from_specs(
                    {name: replace(spec, page_size_padded=None) for name, spec in group_cache_specs.items()}
                )
                if unpadded_spec is None:
                    raise ValueError("Removing DS physical padding must not change a native group's lifetime")
            groups.append(KVCacheGroupSpec(list(group.layer_names), unpadded_spec, group.is_eagle_group))
        rank_groups.append(groups)
    schemas = [page_schemas_from_groups(groups) for groups in rank_groups]
    if any(pages != schemas[0] for pages in schemas[1:]):
        raise ValueError("DS hierarchical cache schemas must match on every TP rank")
    pages = schemas[0]
    layer_names = [page.name for page in pages]
    group_bytes = Counter()
    for page in pages:
        group_bytes[page.group_id] += page.allocation_nbytes
    min_group_block_bytes = min(group_bytes.values())
    text_config = vllm_config.model_config.hf_text_config
    huge_page_bytes = text_config.hidden_size * text_config.moe_intermediate_size
    huge_page_bytes //= vllm_config.parallel_config.tensor_parallel_size
    # vLLM supplies a cache budget; O-MoE uses it for the whole shared arena,
    # including expert pages, and keeps identical byte geometry on all ranks.
    budget = min(available_memory)
    min_huge_blocks, max_huge_blocks = 0, budget // huge_page_bytes
    while min_huge_blocks < max_huge_blocks:
        candidate_huge_blocks = (min_huge_blocks + max_huge_blocks + 1) // 2
        arena_bytes = candidate_huge_blocks * huge_page_bytes
        map_bytes = (1 + arena_bytes // min_group_block_bytes) * len(pages) * 4
        if arena_bytes + map_bytes <= budget:
            min_huge_blocks = candidate_huge_blocks
        else:
            max_huge_blocks = candidate_huge_blocks - 1
    arena_bytes = min_huge_blocks * huge_page_bytes
    configurations = []
    for groups in rank_groups:
        config = KVCacheConfig(
            num_blocks=1 + arena_bytes // min_group_block_bytes,
            kv_cache_tensors=[KVCacheTensor(size=arena_bytes, shared_by=list(layer_names))],
            kv_cache_groups=groups,
        )
        layout = HierarchicalMemoryLayout.from_config(vllm_config, config)
        pool = HierarchicalBlockPool(**layout.allocator_kwargs())
        pool.reserve_at("huge", (0,))
        required = Counter(
            expert=minimum_expert_slots(vllm_config, layout.local_experts_per_layer) * layout.expert_span_pages
        )
        for group_id, group in enumerate(groups):
            spec = group.kv_cache_spec
            group_cache_specs = getattr(spec, "kv_cache_specs", None)
            group_specs = tuple(group_cache_specs.values()) if group_cache_specs is not None else (spec,)
            # Original spec methods include compression, sliding-window and
            # chunk admission limits. Only the resulting count is reused.
            count = max(
                cdiv(cache_spec.max_memory_usage_bytes(vllm_config), cache_spec.page_size_bytes)
                for cache_spec in group_specs
            )
            for page in pages:
                if page.group_id == group_id:
                    required[page.kind] += count
        if not pool.can_reserve(required):
            raise ValueError("DS arena cannot fit native request admission bounds and a full local expert layer")
        configurations.append(config)
    return configurations
