#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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

from dataclasses import dataclass
from enum import Enum
from typing import ClassVar, List, Optional, Tuple, Type

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.nn as nn
import torch_npu
from vllm.attention.backends.abstract import (AttentionBackend, AttentionImpl,
                                              AttentionLayer, AttentionType)
from vllm.config import VllmConfig
from vllm.distributed import (get_dcp_group,
                              get_decode_context_model_parallel_rank,
                              get_decode_context_model_parallel_world_size)
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.utils import cdiv, direct_register_custom_op, round_down
from vllm.v1.attention.backends.utils import AttentionCGSupport
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import AttentionSpec

from vllm_ascend.attention.utils import (AscendCommonAttentionMetadata,
                                         maybe_save_kv_layer_to_connector,
                                         split_decodes_and_prefills,
                                         wait_for_kv_layer_from_connector)
from vllm_ascend.compilation.acl_graph import (get_graph_params,
                                               update_graph_params_workspaces)
from vllm_ascend.ops.attention import vanilla_chunked_prefill
from vllm_ascend.utils import (ACL_FORMAT_FRACTAL_NZ, aligned_16, is_310p,
                               nd_to_nz_2d, nd_to_nz_spec,
                               prefill_context_parallel_enable, version_check)

from ..utils import weak_ref_tensors

if prefill_context_parallel_enable():
    from vllm.distributed import (
        get_pcp_group, get_prefill_context_model_parallel_rank,
        get_prefill_context_model_parallel_world_size)


class AscendAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = True

    @staticmethod
    def get_name() -> str:
        return "ASCEND"

    @staticmethod
    def get_impl_cls() -> Type["AscendAttentionBackendImpl"]:
        return AscendAttentionBackendImpl

    @staticmethod
    def get_metadata_cls() -> Type["AscendMetadata"]:
        return AscendMetadata

    @staticmethod
    def get_builder_cls() -> type["AscendAttentionMetadataBuilder"]:
        return AscendAttentionMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> Tuple[int, ...]:
        if is_310p():
            return (2, num_blocks, num_kv_heads * head_size // 16, block_size,
                    16)
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_bsh_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> Tuple[int, ...]:
        return (2, num_blocks, block_size, num_kv_heads * head_size)

    @staticmethod
    def swap_blocks(
        src_kv_cache: List[torch.Tensor],
        dst_kv_cache: List[torch.Tensor],
        src_to_dst: torch.Tensor,
    ) -> None:
        src_key_cache, src_value_cache = src_kv_cache[0], src_kv_cache[1]
        dst_key_cache, dst_value_cache = dst_kv_cache[0], dst_kv_cache[1]
        src_indices = src_to_dst[:, 0]
        dst_indices = src_to_dst[:, 1]

        dst_key_cache[dst_indices] = src_key_cache[src_indices].to(
            dst_key_cache.device)
        dst_value_cache[dst_indices] = src_value_cache[src_indices].to(
            dst_key_cache.device)

    @staticmethod
    def copy_blocks(
        kv_caches: List[torch.Tensor],
        src_to_dists: torch.Tensor,
    ) -> None:
        src_indices = src_to_dists[:, 0]
        dst_indices = src_to_dists[:, 1]

        for kv_cache in kv_caches:
            key_caches = kv_cache[0]
            value_caches = kv_cache[1]
            key_caches[dst_indices] = key_caches[src_indices]
            value_caches[dst_indices] = value_caches[src_indices]

    @staticmethod
    def get_supported_block_size() -> list[int]:
        return [64]


class AscendAttentionState(Enum):
    PrefillNoCache = 0
    PrefillCacheHit = 1
    DecodeOnly = 2
    ChunkedPrefill = 3
    SpecDecoding = 4


@dataclass
class AscendPCPMetadata:
    q_head_idx: torch.Tensor = None
    q_tail_idx: torch.Tensor = None
    kv_with_q_head_nomask_idx: torch.Tensor = None
    kv_with_q_head_mask_idx: torch.Tensor = None
    kv_with_q_tail_nomask_idx: torch.Tensor = None
    kv_with_q_tail_mask_idx: torch.Tensor = None
    attn_mask_seqlens: torch.Tensor = None
    head_attn_nomask_seqlens: torch.Tensor = None
    tail_attn_nomask_seqlens: torch.Tensor = None
    q_full_idx: torch.Tensor = None
    pcp_prefill_mask: torch.Tensor = None


@dataclass
class AscendMetadataForPrefill:
    @dataclass
    class ChunkedContextMetadata:
        cu_seq_lens: torch.Tensor
        starts: torch.Tensor
        seq_tot: list[int]
        max_seq_lens: list[int]
        workspace: torch.Tensor
        chunk_seq_lens: torch.Tensor
    """ Prefill Specific Metadata for Ascend"""
    pcp_metadata: Optional[AscendPCPMetadata] = None
    pcp_allgather_restore_idx: Optional[List[int]] = None
    chunked_context: Optional[ChunkedContextMetadata] = None
    num_computed_tokens_of_pcp_dcp: Optional[list[Optional[list[Optional[list[int]]]]]] = None
    num_computed_tokens_of_cp_sp_single: Optional[list[Optional[list[Optional[list[int]]]]]] = None
    num_computed_tokens_of_cp_sp_current: Optional[list[Optional[list[Optional[list[int]]]]]] = None
    num_computed_tokens_of_cp_sp_accum: Optional[list[Optional[list[Optional[list[int]]]]]] = None


@dataclass
class AscendMetadataForDecode:
    """ Decode Specific Metadata for Ascend"""
    num_computed_tokens_of_pcp_dcp: Optional[list[Optional[list[Optional[
        list[int]]]]]] = None
    num_computed_tokens_of_cp_sp_single: Optional[list[Optional[list[Optional[list[int]]]]]] = None
    num_computed_tokens_of_cp_sp_current: Optional[list[Optional[list[Optional[list[int]]]]]] = None
    num_computed_tokens_of_cp_sp_accum: Optional[list[Optional[list[Optional[list[int]]]]]] = None


@dataclass
class AscendMetadata:
    # **************************** Basic Properties ************************** #
    attn_mask: Optional[torch.Tensor] = None
    # Current state of this attention run.
    attn_state: AscendAttentionState = AscendAttentionState.ChunkedPrefill

    # Number of tokens excluding padding.
    num_actual_tokens_pcp_padded: int = 0
    num_actual_tokens: int = 0
    num_decode_tokens: int = 0
    num_prefills: int = 0
    num_decodes: int = 0

    # The sequence length per sequence. Sequence length means the computed
    # tokens + new tokens (is None if it is a decoding).
    # (batch_size,)
    # TODO(Angazenn): The following parameters are quite redundant and
    # contains similar information (such as seq_lens seq_lens_list). We
    # should simplified these parameters once attention schema in vLLM-Ascend
    # is unified.
    seq_lens: torch.Tensor = None
    seq_lens_list: List[int] = None  # type: ignore
    actual_seq_lengths_q: List[int] = None  # type: ignore

    query_start_loc: torch.Tensor = None
    query_lens: torch.Tensor = None
    # Maximum query length in the batch (None for decoding).
    max_query_len: Optional[int] = None

    # ********************** KV Cache Related Properties ********************* #
    # Block addresses per sequence (Seq id -> list of physical block).
    # (batch_size, max_blocks_per_seq)
    block_tables: torch.Tensor = None

    # The indices of the token slots that input tokens will be stored into.
    # E.g., if `slot_mapping` is [35, 2, 17] and the block size is 16, the
    # three tokens are stored in the 3rd slot in block 2, 2nd slot in block 0,
    # and 1st slot in block 1, respectively.
    # (num_tokens,)
    slot_mapping: torch.Tensor = None

    # *************************** Other Properties *************************** #
    enable_dbo_across_dp: bool = False

    prefill: Optional[AscendMetadataForPrefill] = None

    decode_meta: Optional[AscendMetadataForDecode] = None


class AscendAttentionMetadataBuilder:
    # Does this backend/builder support ACL Graphs for attention (default: no).
    aclgraph_support: ClassVar[AttentionCGSupport] = \
        AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE
    # Does this backend/builder reorder the batch?
    # If not, set this to None. Otherwise set it to the query
    # length that will be pulled into the front of the batch.
    reorder_batch_threshold: ClassVar[int] = 1

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.device = device
        self.max_num_blocks_per_req = cdiv(
            self.model_config.max_model_len,
            AscendAttentionBackend.get_supported_block_size()[0])
        
        scheduler_config = vllm_config.scheduler_config
        self.block_size = vllm_config.cache_config.block_size
        self.max_blocks = (vllm_config.model_config.max_model_len +
                           self.block_size - 1) // self.block_size
        self.chunked_prefill_enabled = scheduler_config.chunked_prefill_enabled
        if self.chunked_prefill_enabled:
            self.chunked_prefill_workspace_size = min(
                # Max sure there is enough for 8 full length request or at least
                # 4 pages of cache per request
                max(8 * self.model_config.max_model_len,
                    4 * scheduler_config.max_num_seqs * self.block_size),
                # For long-context models try not to over-allocate limiting
                # kv-cache space, limiting it to 64k tokens,
                # which would result in the workspace being:
                #   2*(576)*(64*1024) = 144mb
                # (assuming 576 MLA head dim, and fp16)
                # which would result in up-projected context being
                #   2*(192*128)*(64*1024) = 3gb
                # (assuming 192 QK head dim, 128 heads, and fp16)
                128 * 1024)
            assert self.chunked_prefill_workspace_size >= \
                scheduler_config.max_num_seqs * self.block_size
            self.chunked_prefill_workspace = torch.empty(
                (self.chunked_prefill_workspace_size,
                 self.model_config.get_head_size()),
                dtype=self.model_config.dtype,
                device=device,
            )

    def reorder_batch(self, input_batch,
                      scheduler_output: "SchedulerOutput") -> bool:
        return False

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: AscendCommonAttentionMetadata,
        model: Optional[nn.Module] = None,
    ):
        num_reqs = common_attn_metadata.num_reqs
        num_actual_tokens = common_attn_metadata.num_actual_tokens
        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu[:
                                                                       num_reqs
                                                                       + 1]

        decode_threshold = 1
        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = \
            split_decodes_and_prefills(common_attn_metadata, decode_threshold=decode_threshold)
        assert num_decodes + num_prefills == num_reqs
        assert num_decode_tokens + num_prefill_tokens == num_actual_tokens

        block_table = common_attn_metadata.block_table_tensor
        query_lens = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
        seq_lens = common_attn_metadata.seq_lens_cpu[:num_reqs]

        long_seq_metadata = common_attn_metadata.prefill_context_parallel_metadata
        num_actual_tokens_pcp_padded = long_seq_metadata.num_actual_tokens_pcp_padded if long_seq_metadata else None
        if num_actual_tokens_pcp_padded is None:
            num_actual_tokens_pcp_padded = num_actual_tokens

        slot_mapping = common_attn_metadata.slot_mapping[:
                                                         num_actual_tokens_pcp_padded]
        # slot_mapping = common_attn_metadata.slot_mapping[:num_actual_tokens]
        attn_mask = common_attn_metadata.attn_mask
        attn_state = common_attn_metadata.attn_state
        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu[:
                                                                       num_reqs
                                                                       + 1]

        # query_start_loc = query_start_loc_cpu.to(self.device,
        #                                          non_blocking=True)
        # TODO twq
        # query_seq_lens_cpu = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
        # query_lens = query_seq_lens_cpu[:num_reqs]
        # seq_lens = common_attn_metadata.seq_lens_cpu[:num_reqs]
        num_computed_tokens_cpu = (seq_lens - query_lens)


        if attn_state == AscendAttentionState.DecodeOnly and \
                common_attn_metadata.num_input_tokens > num_actual_tokens:
            padded_num_tokens = common_attn_metadata.num_input_tokens - num_actual_tokens
            seq_lens = torch.cat([
                seq_lens,
                torch.ones(padded_num_tokens,
                           dtype=seq_lens.dtype,
                           device=seq_lens.device)
            ])
            block_table_padding = torch.zeros(
                (padded_num_tokens, ) + block_table.shape[1:],
                dtype=block_table.dtype,
                device=block_table.device)
            block_table = torch.cat([block_table, block_table_padding], dim=0)
            query_start_loc_cpu = torch.cat([
                query_start_loc_cpu,
                torch.arange(query_start_loc_cpu[-1] + 1,
                             query_start_loc_cpu[-1] + padded_num_tokens,
                             dtype=query_start_loc_cpu.dtype,
                             device=query_start_loc_cpu.device)
            ])

        query_start_loc = query_start_loc_cpu.to(self.device,
                                                 non_blocking=True)

        if is_310p():
            if attn_state == AscendAttentionState.PrefillNoCache:
                mask_nz = nd_to_nz_2d(attn_mask)
                attn_mask = torch_npu.npu_format_cast(mask_nz.contiguous(),
                                                      ACL_FORMAT_FRACTAL_NZ)
            elif attn_state == AscendAttentionState.ChunkedPrefill:
                mask_nz = nd_to_nz_spec(attn_mask)
                attn_mask = torch_npu.npu_format_cast(mask_nz.contiguous(),
                                                      ACL_FORMAT_FRACTAL_NZ)

        prefill_metadata = None
        chunked_context_metadata = None
        if num_prefills > 0:
            reqs_start = num_decodes  # prefill_start
            # tokens_start = num_decode_tokens
            # max_query_len = query_lens[reqs_start:].max().item()
            # max_seq_lens = seq_lens[reqs_start:].max().item()
            # prefill_query_start_loc = query_start_loc[
            #     reqs_start:] - query_start_loc[reqs_start]

            context_lens_cpu = num_computed_tokens_cpu[reqs_start:num_reqs]
            max_context_len_cpu = context_lens_cpu.max().item()
            num_prefills_with_context_cpu = (context_lens_cpu > 0).sum().item()

            # TODO 开CP 的时候构造chunk prefill
            if self.chunked_prefill_enabled and max_context_len_cpu > 0:
                max_context_chunk = (self.chunked_prefill_workspace_size //
                                     num_prefills_with_context_cpu)
                max_context_chunk = round_down(max_context_chunk,
                                               self.block_size)

                assert max_context_chunk > 0
                num_chunks = cdiv(max_context_len_cpu, max_context_chunk)
                chunk_starts = torch.arange(num_chunks, dtype=torch.int32) \
                    .unsqueeze(1).expand(-1, num_prefills) * max_context_chunk
                chunk_ends = torch.min(context_lens_cpu.unsqueeze(0),
                                       chunk_starts + max_context_chunk)
                chunk_seq_lens = (chunk_ends - chunk_starts).clamp(min=0)
                cu_seq_lens_cpu = torch.zeros(num_chunks,
                                              num_prefills + 1,
                                              dtype=torch.int32,
                                              pin_memory=True)
                torch.cumsum(chunk_seq_lens,
                             dim=1,
                             out=cu_seq_lens_cpu[:, 1:],
                             dtype=torch.int32)
                chunked_context_metadata = \
                    AscendMetadataForPrefill.ChunkedContextMetadata(
                    cu_seq_lens=cu_seq_lens_cpu.to(self.device, non_blocking=True),
                    starts=chunk_starts.to(self.device, non_blocking=True),
                    seq_tot=chunk_seq_lens.sum(dim=1).tolist(),
                    max_seq_lens=chunk_seq_lens.max(dim=1).values.tolist(),
                    chunk_seq_lens=chunk_seq_lens,
                    workspace=self.chunked_prefill_workspace,
                )


            pcp_metadata = None
            common_long_seq_metadata = common_attn_metadata.prefill_context_parallel_metadata
            if common_long_seq_metadata is not None:
                pcp_metadata = AscendPCPMetadata(
                    q_head_idx=common_long_seq_metadata.q_head_idx_tensor,
                    q_tail_idx=common_long_seq_metadata.q_tail_idx_tensor,
                    kv_with_q_head_nomask_idx=common_long_seq_metadata.
                    kv_with_q_head_nomask_idx_tensor,
                    kv_with_q_head_mask_idx=common_long_seq_metadata.
                    kv_with_q_head_mask_idx_tensor,
                    kv_with_q_tail_nomask_idx=common_long_seq_metadata.
                    kv_with_q_tail_nomask_idx_tensor,
                    kv_with_q_tail_mask_idx=common_long_seq_metadata.
                    kv_with_q_tail_mask_idx_tensor,
                    attn_mask_seqlens=common_long_seq_metadata.
                    attn_mask_seqlens,
                    head_attn_nomask_seqlens=common_long_seq_metadata.
                    head_attn_nomask_seqlens,
                    tail_attn_nomask_seqlens=common_long_seq_metadata.
                    tail_attn_nomask_seqlens,
                    q_full_idx=common_long_seq_metadata.q_full_idx,
                    pcp_prefill_mask=common_long_seq_metadata.pcp_prefill_mask)
            prefill_metadata = AscendMetadataForPrefill(
                chunked_context=chunked_context_metadata,
                pcp_metadata=pcp_metadata,
                pcp_allgather_restore_idx=common_long_seq_metadata.
                pcp_allgather_restore_idx
                if common_long_seq_metadata is not None else None,
                num_computed_tokens_of_pcp_dcp=common_long_seq_metadata.num_computed_tokens_of_pcp_dcp if common_long_seq_metadata is not None else None,
                num_computed_tokens_of_cp_sp_single=common_long_seq_metadata.num_computed_tokens_of_cp_sp_single if common_long_seq_metadata is not None else None,
                num_computed_tokens_of_cp_sp_current=common_long_seq_metadata.num_computed_tokens_of_cp_sp_current if common_long_seq_metadata is not None else None,
                num_computed_tokens_of_cp_sp_accum=np.array(common_long_seq_metadata.num_computed_tokens_of_cp_sp_accum) if common_long_seq_metadata is not None else None
                
                )
        # TODO
        decode_metadata = None
        if num_decodes > 0:
            common_long_seq_metadata = common_attn_metadata.prefill_context_parallel_metadata
            if common_long_seq_metadata is not None:
                num_computed_tokens_of_pcp_dcp = common_long_seq_metadata.num_computed_tokens_of_pcp_dcp
                num_computed_tokens_of_pcp_dcp = np.array(
                    num_computed_tokens_of_pcp_dcp)
                num_computed_tokens_of_cp_sp_single = common_long_seq_metadata.num_computed_tokens_of_cp_sp_single
                num_computed_tokens_of_cp_sp_current = common_long_seq_metadata.num_computed_tokens_of_cp_sp_current
                num_computed_tokens_of_cp_sp_accum = common_long_seq_metadata.num_computed_tokens_of_cp_sp_accum
                num_computed_tokens_of_cp_sp_accum = np.array(num_computed_tokens_of_cp_sp_accum)
                decode_metadata = AscendMetadataForDecode(
                    num_computed_tokens_of_pcp_dcp=num_computed_tokens_of_pcp_dcp,
                     num_computed_tokens_of_cp_sp_single=num_computed_tokens_of_cp_sp_single,
                    num_computed_tokens_of_cp_sp_current=num_computed_tokens_of_cp_sp_current,
                    num_computed_tokens_of_cp_sp_accum=num_computed_tokens_of_cp_sp_accum
                    
                    )
                

        attn_metadata = AscendMetadata(
            num_actual_tokens=num_actual_tokens,
            num_decode_tokens=num_decode_tokens,
            num_actual_tokens_pcp_padded=num_actual_tokens_pcp_padded,
            block_tables=block_table,
            query_start_loc=query_start_loc,
            query_lens=query_lens,
            seq_lens=seq_lens,
            seq_lens_list=seq_lens.tolist(),
            max_query_len=common_attn_metadata.max_query_len,
            actual_seq_lengths_q=query_start_loc_cpu[1:].tolist(),
            slot_mapping=slot_mapping,
            attn_mask=attn_mask,
            attn_state=attn_state,
            enable_dbo_across_dp=common_attn_metadata.enable_dbo_across_dp,
            num_prefills=num_prefills,
            num_decodes=num_decodes,
            prefill=prefill_metadata,
            decode_meta=decode_metadata)
        return attn_metadata

    def build_for_graph_capture(
        self,
        common_attn_metadata: AscendCommonAttentionMetadata,
        attn_state: AscendAttentionState = AscendAttentionState.DecodeOnly,
        model: Optional[nn.Module] = None,
    ):
        if attn_state == AscendAttentionState.DecodeOnly:
            attn_metadata = self.build(
                common_prefix_len=0,
                common_attn_metadata=common_attn_metadata,
            )
        else:
            raise NotImplementedError(
                "Currently we only support building dummy metadata for DecodeOnly state"
            )

        attn_metadata.attn_state = attn_state
        return attn_metadata


class AscendAttentionBackendImpl(AttentionImpl):

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: Optional[List[float]],
        sliding_window: Optional[int],
        kv_cache_dtype: str,
        logits_soft_cap: Optional[float],
        attn_type: str,
        kv_sharing_target_layer_name: Optional[str],
        **kwargs,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        self.hidden_size = self.num_heads * self.head_size
        self.kv_cache_dtype = kv_cache_dtype
        self.sliding_window = sliding_window
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes,
                                        dtype=torch.float32,
                                        device="npu")
        self.alibi_slopes = alibi_slopes
        self.attn_type = attn_type

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        self.key_cache = None
        self.value_cache = None
        self.torch_npu_check = version_check()
        self.pcp_size = get_prefill_context_model_parallel_world_size(
        ) if prefill_context_parallel_enable() else 1
        self.pcp_rank = get_prefill_context_model_parallel_rank(
        ) if self.pcp_size > 1 else 0
        self.pcp_group = get_pcp_group(
        ).device_group if self.pcp_size > 1 else None

        self.dcp_size = get_decode_context_model_parallel_world_size()
        self.dcp_rank = get_decode_context_model_parallel_rank(
        ) if self.dcp_size > 1 else 0
        self.dcp_group = get_dcp_group(
        ).device_group if self.dcp_size > 1 else None

    def _forward_prefill_no_cache(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: Optional[torch.Tensor] = None,
        num_tokens=0,
    ) -> torch.Tensor:
        assert attn_metadata is not None
        assert attn_metadata.attn_mask is not None

        mask = attn_metadata.attn_mask

        if is_310p():
            # align q k v output tensors
            query = aligned_16(query)
            key = aligned_16(key)
            value = aligned_16(value)
            output = aligned_16(output)
            # do reformat in case of broadcasted tensors
            mask = mask.repeat(attn_metadata.seq_lens.size(0), 1, 1, 1)
            mask = torch_npu.npu_format_cast(mask.contiguous(),
                                             ACL_FORMAT_FRACTAL_NZ)

        torch_npu._npu_flash_attention(
            query=query,
            key=key,
            value=value,
            mask=mask,
            seq_len=attn_metadata.seq_lens[attn_metadata.num_decode_tokens:],
            scale_value=self.scale,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            out=output)
        assert output is not None
        return output[:num_tokens, :, :]

    def _forward_prefill_cache_hit(
        self,
        query: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert attn_metadata is not None
        assert attn_metadata.attn_mask is not None

        compress_mask = attn_metadata.attn_mask
        batch_size = attn_metadata.query_lens.shape[0]
        block_table = attn_metadata.block_tables[:batch_size, :]

        torch_npu._npu_flash_attention_qlens(
            query=query,
            key_cache=self.key_cache,
            value_cache=self.value_cache,
            block_table=block_table,
            mask=compress_mask,
            seq_len=attn_metadata.query_lens,
            context_lens=attn_metadata.seq_lens,
            num_kv_heads=self.num_kv_heads,
            num_heads=self.num_heads,
            scale_value=self.scale,
            out=output)
        return output

    def _forward_decode_only(
        self,
        query: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if is_310p():
            # seq_lens_tensor needs to be transferred to the device for 310P.
            attn_metadata.seq_lens = \
                attn_metadata.seq_lens.to(device=query.device)
        if self.sliding_window is not None and attn_metadata.seq_lens.shape[
                0] == query.size(0):
            batch_size = attn_metadata.seq_lens.shape[0]
            block_size = 128
            query = query.view(batch_size, 1, self.num_heads * self.head_size)
            key = self.key_cache
            value = self.value_cache
            if self.key_cache is not None and self.value_cache is not None:
                block_size = self.key_cache.shape[1]
                key = self.key_cache.flatten(2, 3).contiguous()
                value = self.value_cache.flatten(2, 3).contiguous()

            output, _ = torch_npu.npu_fused_infer_attention_score(
                query,
                key,
                value,
                num_heads=self.num_heads,
                num_key_value_heads=self.num_kv_heads,
                input_layout="BSH",
                block_size=block_size,
                pre_tokens=self.sliding_window,
                scale=self.scale,
                block_table=attn_metadata.block_tables,
                actual_seq_lengths=[1] * len(attn_metadata.seq_lens),
                actual_seq_lengths_kv=attn_metadata.seq_lens)

            output = output.view(batch_size, self.num_heads, self.head_size)
        else:
            graph_params = get_graph_params()
            forward_context: ForwardContext = get_forward_context()
            num_tokens = query.shape[0]
            if forward_context.capturing:
                if self.torch_npu_check:
                    # Get workspace from cache or calculate it if not present.
                    workspace = graph_params.workspaces.get(num_tokens)
                    if workspace is None:
                        workspace = torch_npu._npu_paged_attention_get_workspace(
                            query=query,
                            key_cache=self.key_cache,
                            value_cache=self.value_cache,
                            num_kv_heads=self.num_kv_heads,
                            num_heads=self.num_heads,
                            scale_value=self.scale,
                            block_table=attn_metadata.block_tables,
                            context_lens=attn_metadata.seq_lens,
                            out=output)
                        update_graph_params_workspaces(num_tokens, workspace)

                # Handle graph capturing mode
                stream = torch_npu.npu.current_stream()

                event = torch.npu.ExternalEvent()
                event.wait(stream)
                event.reset(stream)
                graph_params.events[num_tokens].append(event)
                graph_params.attn_params[num_tokens].append((
                    weak_ref_tensors(query),
                    weak_ref_tensors(self.key_cache),
                    weak_ref_tensors(self.value_cache),
                    self.num_kv_heads,
                    self.num_heads,
                    self.scale,
                    weak_ref_tensors(attn_metadata.block_tables),
                    attn_metadata.seq_lens,
                    weak_ref_tensors(output),
                ))

                torch.npu.graph_task_group_begin(stream)

                if self.torch_npu_check:
                    torch_npu._npu_paged_attention(
                        query=query,
                        key_cache=self.key_cache,
                        value_cache=self.value_cache,
                        num_kv_heads=self.num_kv_heads,
                        num_heads=self.num_heads,
                        scale_value=self.scale,
                        block_table=attn_metadata.block_tables,
                        context_lens=attn_metadata.seq_lens,
                        out=output,
                        workspace=workspace)
                else:
                    torch_npu._npu_paged_attention(
                        query=query,
                        key_cache=self.key_cache,
                        value_cache=self.value_cache,
                        num_kv_heads=self.num_kv_heads,
                        num_heads=self.num_heads,
                        scale_value=self.scale,
                        block_table=attn_metadata.block_tables,
                        context_lens=attn_metadata.seq_lens,
                        out=output)
                handle = torch.npu.graph_task_group_end(stream)
                graph_params.handles[num_tokens].append(handle)
            else:
                torch_npu._npu_paged_attention(
                    query=query,
                    key_cache=self.key_cache,
                    value_cache=self.value_cache,
                    num_kv_heads=self.num_kv_heads,
                    num_heads=self.num_heads,
                    scale_value=self.scale,
                    block_table=attn_metadata.block_tables,
                    context_lens=attn_metadata.seq_lens,
                    out=output)
        return output

    def _forward_v1_style(
        self,
        query: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Use chunked prefill for head size 192 scenario, like deepseek
        # paged_attention_splitfuse maybe crash at such scenario.
        # TODO: vanilla path will be removed after the kernel support
        # head_size 192 scenario.
        if self.head_size == 192:
            cu_seqlen_q = [0] + attn_metadata.query_lens.tolist()
            cu_seqlen_k = [0] + attn_metadata.seq_lens.tolist()
            cu_seqlen_q = torch.tensor(cu_seqlen_q, device=query.device)
            cu_seqlen_k = torch.tensor(cu_seqlen_k, device=query.device)
            cu_seqlen_q = torch.cumsum(cu_seqlen_q, dim=0)
            cu_seqlen_k = torch.cumsum(cu_seqlen_k, dim=0)
            max_seqlen_q = torch.max(attn_metadata.query_lens)
            max_seqlen_k = torch.max(attn_metadata.seq_lens)
            vanilla_chunked_prefill(output, query, self.key_cache,
                                    self.value_cache,
                                    attn_metadata.block_tables, cu_seqlen_q,
                                    cu_seqlen_k, max_seqlen_q, max_seqlen_k,
                                    self.scale, None, True)
            return output

        # Use paged attention.
        assert attn_metadata is not None
        assert attn_metadata.attn_mask is not None

        if is_310p():
            # Do reformat in case of broadcasted tensors.
            attn_metadata.attn_mask = \
                torch_npu.npu_format_cast(attn_metadata.attn_mask.contiguous(),
                                          ACL_FORMAT_FRACTAL_NZ)
            attn_metadata.seq_lens = \
                attn_metadata.seq_lens.to(device=query.device)

        if torch.version.cann.startswith("8.3"):
            # TODO:The npu_fused_infer_attention_score op is planned to
            # be utilized in a wider range in upcoming versions.
            num_block, block_size, _, _ = self.key_cache.shape  # type: ignore
            key = self.key_cache.view(  # type: ignore
                num_block, block_size, -1)
            value = self.value_cache.view(  # type: ignore
                num_block, block_size, -1)

            output, _ = torch_npu.npu_fused_infer_attention_score(
                query=query,
                key=key,
                value=value,
                atten_mask=attn_metadata.attn_mask,
                block_table=attn_metadata.block_tables,
                input_layout="TND",
                block_size=block_size,
                actual_seq_lengths=attn_metadata.actual_seq_lengths_q,
                actual_seq_lengths_kv=attn_metadata.seq_lens_list,
                num_key_value_heads=self.num_kv_heads,
                num_heads=self.num_heads,
                scale=self.scale,
                sparse_mode=3,
            )
        else:
            torch_npu._npu_paged_attention_splitfuse(
                query=query,
                key_cache=self.key_cache,
                value_cache=self.value_cache,
                mask=attn_metadata.attn_mask,
                block_table=attn_metadata.block_tables,
                seq_len=attn_metadata.query_lens,
                context_lens=attn_metadata.seq_lens,
                num_kv_heads=self.num_kv_heads,
                num_heads=self.num_heads,
                scale_value=self.scale,
                out=output)
        return output

    def _attention_with_nomask_and_mask(self, q: torch.Tensor,
                                        q_seqlens: List[int],
                                        k_nomask: torch.Tensor,
                                        v_nomask: torch.Tensor,
                                        kv_seqlens_nomask: List[int],
                                        k_mask: torch.Tensor,
                                        v_mask: torch.Tensor,
                                        kv_seqlens_mask: List[int],
                                        mask: torch.Tensor) -> torch.Tensor:

        # nomask Attention
        if k_nomask is not None:
            attn_out_nomask, attn_lse_nomask = torch.ops.npu.npu_fused_infer_attention_score(
                q,
                k_nomask,
                v_nomask,
                num_heads=self.num_heads,
                num_key_value_heads=self.num_kv_heads,
                input_layout="TND",
                atten_mask=None,
                scale=self.scale,
                sparse_mode=0,
                antiquant_mode=0,
                antiquant_scale=None,
                softmax_lse_flag=True,
                actual_seq_lengths_kv=kv_seqlens_nomask,
                actual_seq_lengths=q_seqlens)

        # mask Attention
        attn_out_mask, attn_lse_mask = torch.ops.npu.npu_fused_infer_attention_score(
            q,
            k_mask,
            v_mask,
            num_heads=self.num_heads,
            num_key_value_heads=self.num_kv_heads,
            input_layout="TND",
            atten_mask=mask,
            scale=self.scale,
            sparse_mode=3,
            antiquant_mode=0,
            antiquant_scale=None,
            softmax_lse_flag=True,
            actual_seq_lengths_kv=kv_seqlens_mask,
            actual_seq_lengths=q_seqlens)

        # update
        output = attn_out_mask
        lse = attn_lse_mask
        if k_nomask is not None:
            output, lse = self._update_out_and_lse(
                torch.stack([attn_out_nomask, attn_out_mask], dim=0),
                torch.stack([attn_lse_nomask, attn_lse_mask], dim=0))

        return output, lse

    def _forward_prefill_cp(self, query: torch.Tensor, key: torch.Tensor,
                            value: torch.Tensor,
                            attn_metadata: AscendMetadata) -> torch.Tensor:
        assert attn_metadata is not None
        assert attn_metadata.prefill is not None
        assert attn_metadata.prefill.pcp_metadata is not None
        # Use precomputed indices from the metadata (already converted to tensors and on device)
        q_head_idx = attn_metadata.prefill.pcp_metadata.q_head_idx
        q_tail_idx = attn_metadata.prefill.pcp_metadata.q_tail_idx
        kv_with_q_head_nomask_idx = attn_metadata.prefill.pcp_metadata.kv_with_q_head_nomask_idx
        kv_with_q_head_mask_idx = attn_metadata.prefill.pcp_metadata.kv_with_q_head_mask_idx
        kv_with_q_tail_nomask_idx = attn_metadata.prefill.pcp_metadata.kv_with_q_tail_nomask_idx
        kv_with_q_tail_mask_idx = attn_metadata.prefill.pcp_metadata.kv_with_q_tail_mask_idx
        attn_mask_seqlens = attn_metadata.prefill.pcp_metadata.attn_mask_seqlens
        head_attn_nomask_seqlens = attn_metadata.prefill.pcp_metadata.head_attn_nomask_seqlens
        tail_attn_nomask_seqlens = attn_metadata.prefill.pcp_metadata.tail_attn_nomask_seqlens
        mask = attn_metadata.prefill.pcp_metadata.pcp_prefill_mask

        # 1. Attention calculation in the first half of Q in load balancing
        output_head, head_lse = self._attention_with_nomask_and_mask(
            q=torch.index_select(query, 0, q_head_idx),
            q_seqlens=attn_mask_seqlens[0].tolist(),
            k_nomask=torch.index_select(key, 0, kv_with_q_head_nomask_idx)
            if self.pcp_rank > 0 else None,
            v_nomask=torch.index_select(value, 0, kv_with_q_head_nomask_idx)
            if self.pcp_rank > 0 else None,
            kv_seqlens_nomask=head_attn_nomask_seqlens[1].tolist(),
            k_mask=torch.index_select(key, 0, kv_with_q_head_mask_idx),
            v_mask=torch.index_select(value, 0, kv_with_q_head_mask_idx),
            kv_seqlens_mask=attn_mask_seqlens[0].tolist(),
            mask=mask)

        # 2. the Attention calculation in the latter half of Q in load balancing
        # pcp_rank0: Q3*KV0~KV2 + Q3*KV3
        # pcp_rank1: Q2*KV0~KV1 + Q2*KV2
        output_tail, tail_lse = self._attention_with_nomask_and_mask(
            q=torch.index_select(query, 0, q_tail_idx),
            q_seqlens=attn_mask_seqlens[0].tolist(),
            k_nomask=torch.index_select(key, 0, kv_with_q_tail_nomask_idx),
            v_nomask=torch.index_select(value, 0, kv_with_q_tail_nomask_idx),
            kv_seqlens_nomask=tail_attn_nomask_seqlens[1].tolist(),
            k_mask=torch.index_select(key, 0, kv_with_q_tail_mask_idx),
            v_mask=torch.index_select(value, 0, kv_with_q_tail_mask_idx),
            kv_seqlens_mask=attn_mask_seqlens[0].tolist(),
            mask=mask)

        # 3. Combine the output of the first half and second half.
        q_full_idx = attn_metadata.prefill.pcp_metadata.q_full_idx
        output = torch.index_select(
            torch.cat([output_head, output_tail], dim=0), 0, q_full_idx)
        
        attn_lse = torch.index_select(
            torch.cat([head_lse, tail_lse], dim=0), 0, q_full_idx)

        return output, attn_lse

    def _update_out_and_lse(self, out_list: torch.Tensor,
                            lse_list: torch.Tensor) -> torch.Tensor:
        """LSE_final = log(sum(exp(LSE_i))), O_final = sum(exp(LSE_i - LSE_final) * O_i)
        Args:
            out_list: shape = [N, batch_size, num_heads, head_size]
            lse_list: shape = [N, batch_size, num_heads, 1]
        Returns:
            out_final: shape = [batch_size, num_heads, head_size]
            lse_final: shape = [batch_size, num_heads, 1]
        """
        lse_final = torch.logsumexp(lse_list, dim=0, keepdim=False)
        out_final = torch.sum(torch.exp(lse_list - lse_final) * out_list,
                              dim=0)
        return out_final, lse_final

    def _forward_decode_pcp_dcp(self, query: torch.Tensor,
                                attn_metadata: AscendMetadata) -> torch.Tensor:
        assert self.key_cache is not None
        assert self.value_cache is not None

        if self.dcp_size > 1:
            query = get_dcp_group().all_gather(query, 1)
            num_heads = self.num_heads * self.dcp_size
        else:
            num_heads = self.num_heads

        # 1. Compute out&lse by "npu_fused_infer_attention_score"
        attn_out, attn_lse = torch.ops.npu.npu_fused_infer_attention_score(
            query.view(query.shape[0], 1, query.shape[1], query.shape[2]),
            # [b,num_heads,head_size] -> [b,1,num_heads,head_size]
            self.key_cache.view(self.key_cache.shape[0],
                                self.key_cache.shape[1], -1),
            self.value_cache.view(self.key_cache.shape[0],
                                  self.key_cache.shape[1], -1),
            num_heads=num_heads,
            num_key_value_heads=self.num_kv_heads,
            input_layout="BSND",
            atten_mask=None,
            scale=self.scale,
            antiquant_mode=0,
            antiquant_scale=None,
            softmax_lse_flag=True,
            block_table=attn_metadata.block_tables,
            block_size=self.key_cache.shape[1],
            actual_seq_lengths_kv=attn_metadata.decode_meta.
            num_computed_tokens_of_pcp_dcp[:, self.pcp_rank, self.dcp_rank],
        )

        attn_out = attn_out.view(attn_out.shape[0], attn_out.shape[2],
                                 attn_out.shape[3])
        attn_lse = attn_lse.view(attn_lse.shape[0], attn_lse.shape[1], 1)
        if self.dcp_size > 1:
            # Concat out&lse: [bs,num_heads,v_head_dim] + [bs,num_heads,1] -> [bs,num_heads,v_head_dim+1]
            attn_out_lse = torch.cat([attn_out, attn_lse], dim=-1)
            # permute: [bs, num_heads, v_head_dim+1] -> [num_heads, v_head_dim+1, bs]
            attn_out_lse = attn_out_lse.permute([1, 2, 0]).contiguous()
            attn_out_lse_all2all = torch.empty_like(attn_out_lse)
            dist.all_to_all_single(attn_out_lse_all2all,
                                   attn_out_lse,
                                   group=self.dcp_group)
            # permute: [num_heads, v_head_dim+1, bs] -> [bs, num_heads, v_head_dim+1]
            attn_out_lse_all2all = attn_out_lse_all2all.permute([2, 0, 1])
            attn_out_lse_split_on_seq = list(
                torch.chunk(attn_out_lse_all2all, self.dcp_size, dim=1))

            attn_out_lse_split_dcp = torch.stack(
                attn_out_lse_split_on_seq,
                dim=0)  # [dcp, batch_size, num_heads, head_size+1]
            # Update out&lse
            attn_out_split_dcp, attn_lse_split_dcp = torch.split(
                attn_out_lse_split_dcp, [self.head_size, 1], dim=-1)
            attn_out, attn_lse = self._update_out_and_lse(
                attn_out_split_dcp, attn_lse_split_dcp)
        if self.pcp_size > 1:
            # 2. Concat out&lse: [bs,num_heads,head_size] + [bs,num_heads,1] -> [bs,num_heads,head_size+1]
            attn_out_lse = torch.cat([attn_out, attn_lse], dim=-1)
            # 3. AllGather out&lse within CP group
            attn_out_lse_list = [
                torch.empty_like(attn_out_lse) for _ in range(self.pcp_size)
            ]
            dist.all_gather(attn_out_lse_list,
                            attn_out_lse,
                            group=self.pcp_group)
            # 4. Update out&lse
            attn_out_lse_allgather = torch.stack(
                attn_out_lse_list,
                dim=0)  # [pcp, batch_size, num_heads, head_size+1]
            attn_out_allgather, attn_lse_allgather = torch.split(
                attn_out_lse_allgather, [self.head_size, 1], dim=-1)
            attn_out, _ = self._update_out_and_lse(attn_out_allgather,
                                                   attn_lse_allgather)
        return attn_out

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: Tuple[torch.Tensor],
        attn_metadata: AscendMetadata,
        output: Optional[torch.Tensor] = None,
        trace_flag: bool = True,
    ) -> torch.Tensor:
        """Forward pass with Ascend attention.
        Args:
            query: shape = [batch_size, seq_len, num_heads * head_size]
            key: shape = [batch_size, seq_len, num_kv_heads * head_size]
            value: shape = [batch_size, seq_len, num_kv_heads * head_size]
            kv_cache: shape = [key_cache, value_cache]
                      key_cache = [num_blocks, block_size,
                                   num_kv_heads, head_size]
                      value_cache = [num_blocks, block_size,
                                     num_kv_heads, head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [batch_size * seq_len, num_heads, head_size]
        """
        num_tokens = query.shape[0]
        use_kv_cache_int8 = len(
            kv_cache) > 0 and kv_cache[0].dtype == torch.int8
        if output is None:
            output = torch.empty(num_tokens,
                                 self.num_heads,
                                 self.head_size,
                                 dtype=query.dtype,
                                 device=query.device)
        ori_output = output
        if trace_flag:
            torch.ops.vllm.unified_ascend_attention_with_output(
                query=query,
                key=key,
                value=value,
                output=output,
                layer_name=layer.layer_name)

        elif hasattr(layer, 'quant_method') and use_kv_cache_int8:
            output = layer.quant_method.apply(layer, query, key, value,
                                              kv_cache, attn_metadata,
                                              self.attn_type, self.scale,
                                              output)

        else:
            if attn_metadata is None:
                return output.view(num_tokens, self.hidden_size)
            num_decode_tokens = attn_metadata.num_decode_tokens
            has_decode = attn_metadata.num_decodes > 0
            has_prefill = attn_metadata.num_prefills > 0

            assert layer._k_scale_float == 1.0 and layer._v_scale_float == 1.0
            attn_type = self.attn_type
            if attn_type != AttentionType.DECODER and attn_type != AttentionType.ENCODER_ONLY:
                raise NotImplementedError("Encoder/decoder cross-attention "
                                          "are not implemented for "
                                          "PallasAttentionBackendImpl")
            # View q k v to BSH.
            query = query.view(-1, self.num_heads, self.head_size)
            key = key.view(-1, self.num_kv_heads, self.head_size)
            value = value.view(-1, self.num_kv_heads, self.head_size)
            # TODO: Remove this contiguous in the future.
            value = value.contiguous()

            if len(kv_cache) > 1:
                if self.key_cache is None:
                    self.key_cache, self.value_cache = kv_cache[0], kv_cache[1]

                if has_decode:
                    slot_mapping = attn_metadata.slot_mapping[:num_decode_tokens * self.pcp_size: self.pcp_size] \
                        if self.pcp_size * self.dcp_size > 1 else attn_metadata.slot_mapping[:num_decode_tokens]
                    torch_npu._npu_reshape_and_cache(
                        key=key[:num_decode_tokens],
                        value=value[:num_decode_tokens],
                        key_cache=self.key_cache,
                        value_cache=self.value_cache,
                        slot_indices=slot_mapping)

                if has_prefill:
                    if self.pcp_size > 1:
                        kv = torch.cat([key, value], dim=-1)
                        all_kv = get_pcp_group().all_gather(kv, dim=0)
                        pcp_allgather_restore_idx = attn_metadata.prefill.pcp_allgather_restore_idx if attn_metadata.prefill else None
                        all_kv = torch.index_select(all_kv, 0,
                                                    pcp_allgather_restore_idx)
                        key, value = all_kv.split(
                            [self.head_size, self.head_size], dim=-1)

                    torch_npu._npu_reshape_and_cache(
                        key=key[self.pcp_size *
                                num_decode_tokens:attn_metadata.
                                num_actual_tokens_pcp_padded],
                        value=value[self.pcp_size *
                                    num_decode_tokens:attn_metadata.
                                    num_actual_tokens_pcp_padded],
                        key_cache=self.key_cache,
                        value_cache=self.value_cache,
                        slot_indices=attn_metadata.
                        slot_mapping[self.pcp_size *
                                     num_decode_tokens:attn_metadata.
                                     num_actual_tokens_pcp_padded])

            if self.pcp_size * self.dcp_size > 1:
                output = self._forward_pcp_dcp(query, key, value, kv_cache, attn_metadata, output)
            elif attn_type == AttentionType.ENCODER_ONLY:
                cum_seq_len = attn_metadata.query_start_loc[1:].tolist()
                attn_out = torch_npu.npu_fusion_attention(
                    query,
                    key,
                    value,
                    head_num=self.num_heads,
                    input_layout="TND",
                    scale=self.scale,
                    sparse_mode=4,
                    atten_mask=attn_metadata.attn_mask,
                    next_tockens=attn_metadata.max_query_len,
                    actual_seq_qlen=cum_seq_len,
                    actual_seq_kvlen=cum_seq_len,
                )
                output = attn_out[0]
            # V0-Style scheduler situation.
            elif attn_metadata.attn_state == AscendAttentionState.PrefillNoCache:
                output = self._forward_prefill_no_cache(
                    query, key, value, attn_metadata, output, num_tokens)
            elif attn_metadata.attn_state == \
                    AscendAttentionState.PrefillCacheHit:
                output = self._forward_prefill_cache_hit(
                    query, attn_metadata, output)
            elif attn_metadata.attn_state == AscendAttentionState.DecodeOnly:
                output = self._forward_decode_only(query, attn_metadata,
                                                   output)
            # Normal V1 situation.
            else:
                if torch.version.cann.startswith("8.3"):
                    # npu_fused_infer_attention_score does not support cases
                    # where query.shape[0] != attn_metadata.query_start_loc[-1].
                    # Thus we need unpad it here.
                    num_tokens = attn_metadata.query_start_loc[-1]
                    query = query[:num_tokens]
                output = self._forward_v1_style(query, attn_metadata, output)

        # to make in-place change to the output tensor
        if hasattr(layer, 'quant_method') and use_kv_cache_int8:
            output = output.view(num_tokens, self.num_heads, self.head_size)
        ori_output[:num_tokens, :, :] = output[:num_tokens, :, :]
        return output.view(num_tokens, self.hidden_size)

    def _forward_pcp_dcp(self, query: torch.Tensor, key: torch.Tensor,
                         value: torch.Tensor, kv_cache: Tuple[torch.Tensor], attn_metadata: AscendMetadata,
                         output: Optional[torch.Tensor]) -> torch.Tensor:
        assert attn_metadata is not None
        has_decode = attn_metadata.num_decodes > 0
        has_prefill = attn_metadata.num_prefills > 0
        num_decode_tokens = attn_metadata.num_decode_tokens
        if output is None:
            raise ValueError("Output buffer is required")
        if has_decode:
            decode_query = query[:num_decode_tokens]
            output_decode = self._forward_decode_pcp_dcp(
                decode_query, attn_metadata)
            output[:num_decode_tokens] = output_decode
        if has_prefill:
            prefill_query = query[num_decode_tokens:]
            key = key[self.pcp_size * num_decode_tokens:]
            value = value[self.pcp_size * num_decode_tokens:]
            if self.pcp_size > 1:
                output_prefill, attn_lse = self._forward_prefill_cp(prefill_query, key, value, attn_metadata)
            else:
                # assigned_mask_dim = 2048
                # mask = torch.triu(torch.ones(assigned_mask_dim, assigned_mask_dim),diagonal=1).to(torch.int8).to(prefill_query.device)
                seq_lens = attn_metadata.query_lens[attn_metadata.num_decode_tokens:]
                output_prefill, attn_lse = torch.ops.npu.npu_fused_infer_attention_score(
                    prefill_query,
                    key,
                    value,
                    num_heads=self.num_heads,
                    num_key_value_heads=self.num_kv_heads,
                    input_layout="TND",
                    atten_mask=attn_metadata.prefill.cp_prefill_mask,
                    scale=self.scale,
                    sparse_mode=3,
                    antiquant_mode=0,
                    antiquant_scale=None,
                    softmax_lse_flag=True,
                    actual_seq_lengths=seq_lens,   # 待确认
                    actual_seq_lengths_kv=attn_metadata.query_start_loc[1:].tolist())  # 待确认

            if attn_metadata.prefill is not None and attn_metadata.prefill.chunked_context is not None:
                    prefill_query_all = get_pcp_group().all_gather(prefill_query.contiguous(), 0) if self.pcp_size > 1 else prefill_query
                    prefill_query_all = torch.index_select(prefill_query_all, 0, attn_metadata.prefill.pcp_allgather_restore_idx)  if self.pcp_size > 1 else prefill_query_all# TODO 重点排查
                    output_prefill, attn_lse = self._compute_prefill_context(prefill_query_all, kv_cache, attn_metadata, output_prefill, attn_lse)
            output[num_decode_tokens:] = output_prefill
        return output

    def _update_out_and_lse_new(
        self,
        out: torch.Tensor,
        lse: torch.Tensor,
        block_out: torch.Tensor,
        block_lse: torch.Tensor,
        mask: torch.Tensor = None,
    ):
        if out is None:
            out = block_out.to(torch.float32)
            lse = block_lse
        else:
            if mask is None:
                mask = torch.ones([block_out.size(0)],
                                  dtype=torch.uint8,
                                  device=block_out.device)
            out_mask = mask[:, None, None].expand_as(block_out)
            lse_mask = mask[:, None, None].expand_as(block_lse)
            block_out = block_out.to(torch.float32)
            out_without_update = out.clone()
            lse_without_update = lse.clone()

            out = out - F.sigmoid(block_lse - lse) * (out - block_out)
            lse = lse - F.logsigmoid(lse - block_lse)
            # mask
            out = torch.where(out_mask, out, out_without_update)
            lse = torch.where(lse_mask, lse, lse_without_update)
        return out, lse

    def _compute_prefill_context(
            self,
            query: torch.Tensor,
            kv_cache: Tuple[torch.Tensor],
            attn_metadata: AscendMetadata,
            prefix_output: torch.Tensor,
            prefix_lse: torch.Tensor,
    ):
        # 先考虑CP 场景的chunk
        assert len(kv_cache) > 1
        assert attn_metadata is not None
        assert attn_metadata.prefill is not None

        prefill_metadata = attn_metadata.prefill

        if prefill_metadata.chunked_context is None:
            return prefix_output, prefix_lse

        iters = len(prefill_metadata.chunked_context.seq_tot)

        cache_key = kv_cache[0]
        cache_value = kv_cache[1]
        num_heads = cache_key.size(2)
        head_size = kv_cache[0].size(-1)

        # token -> request mapping for building per-token masks when CP>1
        num_tokens_all = query.size(0)
        # 如果开启CP，则是切过CP的seq 长度
        current_seq_lens= torch.tensor(attn_metadata.query_lens, dtype=torch.int32, device=query.device).contiguous()
        current_seq_lens.mul_(self.pcp_size)  # q_full
        prefix_lse_bt = prefix_lse

        # Keep the causal mask; do not override to all-ones. kv lengths  【req_id】[chunk_id][cp-rank][dcp_rank]
        num_computed_tokens_of_cp_sp_accum = attn_metadata.prefill.num_computed_tokens_of_cp_sp_accum
        context_starts_rank = None

        for i in range(iters):
           
            if self.pcp_size * self.dcp_size > 1:
                ## DCP mode: each rank processes its own (cp,dcp) historical context slice per request dimension
                seq_lens_per_chunk = prefill_metadata.chunked_context.chunk_seq_lens[i]
                num_requests = len(seq_lens_per_chunk)
                # Before dealing with a new chunk, set to zero, and accumulate the start positions as chunk prefill step increases
                context_starts_rank = torch.zeros(num_requests, dtype=torch.int32,
                                                  device=query.device) if context_starts_rank is None else context_starts_rank
                
                # Calculate tokens each rank should process per request
                seq_lens_per_chunk_rank = torch.zeros_like(seq_lens_per_chunk, dtype=torch.int32)
                total_toks = 0

                for req_idx in range(num_requests):
                    n_computed_acc = num_computed_tokens_of_cp_sp_accum[req_idx][i]
                    total_toks += n_computed_acc[self.pcp_rank][self.dcp_rank]
                    seq_lens_per_chunk_rank[req_idx] = n_computed_acc[self.pcp_rank][self.dcp_rank]

            
                if total_toks > 0:
                    key = torch.empty(total_toks,
                                              num_heads,
                                              head_size,
                                              dtype=query.dtype,
                                              device=query.device)
                    value = torch.empty(total_toks,
                                       num_heads,
                                       head_size,
                                       dtype=query.dtype,
                                       device=query.device)
                    
                    torch_npu.atb.npu_paged_cache_load(
                        cache_key,
                        cache_value,
                        attn_metadata.block_tables,
                        seq_lens_per_chunk_rank.to(query.device),
                        seq_starts=context_starts_rank,  # slot offsets of current chunk in current iteration
                        key=key,
                        value=value,
                    )
                    seq_lens_per_chunk_rank = seq_lens_per_chunk_rank.to(query.device)
                    
                else:
                    # If current rank has no tokens to process, create empty tensors
                    key = torch.empty(0, self.num_heads, self.head_size,
                                              dtype=query.dtype, device=query.device)
                    value = torch.empty(0, self.num_heads, self.head_size,
                                       dtype=query.dtype, device=query.device)
                    seq_lens_per_chunk_rank = torch.zeros((len(seq_lens_per_chunk),), dtype=torch.int32, device=query.device)

                for req_idx in range(num_requests):
                    # Before dealing with a new chunk, set to zero, and accumulate the start positions as chunk prefill step increases
                    context_starts_rank[req_idx] += num_computed_tokens_of_cp_sp_accum[req_idx][i][self.pcp_rank][self.dcp_rank]
            else:
                # Original logic: CP-only mode
                toks = prefill_metadata.chunked_context.seq_tot[i]
                seq_lens_per_chunk= prefill_metadata.chunked_context.chunk_seq_lens[i]
                seq_lens_per_chunk_rank = seq_lens_per_chunk.to(query.device, dtype=torch.int32).contiguous()

                key = torch.empty(toks,
                                          self.num_heads, self.head_size,
                                          dtype=query.dtype,
                                          device=query.device)
                value = torch.empty(toks,
                                   self.num_heads, self.head_size,
                                   dtype=query.dtype,
                                   device=query.device)

                torch_npu.atb.npu_paged_cache_load(
                    cache_key,
                    cache_value,
                    attn_metadata.block_tables,
                    seq_lens_per_chunk_rank,
                    seq_starts=prefill_metadata.chunked_context.starts[i],
                    key=key,
                    value=value,
                )

            if self.dcp_size > 1:
                kv_local = torch.cat([key, value], dim=-1)
                all_kv = get_pcp_group().all_gather(kv_local, dim=0)
                key, value = all_kv.split([self.head_size, self.head_size], dim=-1)

                seq_lens_per_chunk_rank.mul_(self.dcp_size)  # chunk len: seq/(cp*dcp) -> seq/cp

            if self.pcp_size > 1:
                # CP+DCP mode: first compute this rank's contribution to the chunk
                # Case that no kv_cache has been stored on this rank, no need to do following computation.
                block_out_local = torch.zeros(
                    [num_tokens_all, self.num_heads, self.head_size],
                    dtype=query.dtype,
                    device=query.device)
                block_lse_local = torch.full((self.num_heads, num_tokens_all),
                                             float('-inf'),
                                             dtype=torch.float32,
                
                if seq_lens_per_chunk_rank.item() > 0:
                    # 调用fia算子计算attention
                    block_out_local, block_lse_local = torch.ops.npu.npu_fused_infer_attention_score(
                        query,
                        key,
                        value,
                        num_heads=self.num_heads,
                        num_key_value_heads=self.num_kv_heads,
                        input_layout="TND",     # 
                        atten_mask=None,
                        scale=self.scale,
                        sparse_mode=0,
                        antiquant_mode=0,
                        antiquant_scale=None,
                        softmax_lse_flag=True,  
                        actual_seq_lengths_kv=seq_lens_per_chunk_rank.tolist(),   # 历史KV长度
                        actual_seq_lengths=current_seq_lens)        # 当前q的长度


                # CP dimension fusion (SP already handled above)
            
                block_lse_local_bt = block_lse_local
                out_lse_local = torch.cat([block_out_local, block_lse_local_bt], dim=-1)

                # CP dimension all_gather and fusion
                out_lse_list = [torch.empty_like(out_lse_local) for _ in range(self.pcp_size)]
                dist.all_gather(out_lse_list, out_lse_local, group=self.pcp_group)
                chunk_out_g = None
                chunk_lse_g = None
                for r in range(self.pcp_size):
                    out_lse_r = out_lse_list[r]
                    out_r, lse_r = torch.split(out_lse_r, [self.head_size, 1], dim=-1)
                    token_mask = torch.ones([out_r.size(0)], dtype=torch.uint8, device=out_r.device)
                    chunk_out_g, chunk_lse_g = self._update_out_and_lse_new(
                        chunk_out_g, chunk_lse_g, out_r, lse_r, token_mask)

                chunk_out_g = chunk_out_g[self.pcp_rank * (num_tokens_all // self.pcp_size):(self.pcp_rank + 1) * (
                            num_tokens_all // self.pcp_size)]  # pick q result of cp rank
                chunk_lse_g = chunk_lse_g[self.pcp_rank * (num_tokens_all // self.pcp_size):(self.pcp_rank + 1) * (
                            num_tokens_all // self.pcp_size)]
                if chunk_out_g is not None:
                    if prefix_lse_bt is None:
                        prefix_output = chunk_out_g.to(torch.float32)
                        prefix_lse_bt = chunk_lse_g
                    else:
                        prefix_output, prefix_lse_bt = self._update_out_and_lse_new(
                            prefix_output, prefix_lse_bt, chunk_out_g, chunk_lse_g)
            else:
                # compute this chunk block then update prefix tensors to keep shapes consistent
                block_out_local2 = torch.empty(
                    num_tokens_all, self.num_heads, self.head_size,
                    dtype=query.dtype, device=query.device)
                block_lse_local2 = torch.empty(
                    self.num_heads, num_tokens_all,
                    dtype=torch.float32, device=query.device)

                # TODO 调用fia算子计算attention 
                block_out_local2, block_lse_local2 = torch.ops.npu.npu_fused_infer_attention_score(
                    query,
                    key,
                    value,
                    num_heads=self.num_heads,
                    num_key_value_heads=self.num_kv_heads,
                    input_layout="TND",     # 
                    atten_mask=None,
                    scale=self.scale,
                    sparse_mode=0,
                    antiquant_mode=0,
                    antiquant_scale=None,
                    softmax_lse_flag=True,  
                    actual_seq_lengths_kv=seq_lens_per_chunk_rank.tolist(),   # 历史KV长度
                    actual_seq_lengths=current_seq_lens)        # 当前q的长度

                block_lse_local_bt2 = block_lse_local2

                if prefix_lse_bt is None:
                    prefix_output = block_out_local2.to(torch.float32)
                    prefix_lse_bt = block_lse_local_bt2
                else:
                    prefix_output, prefix_lse_bt = self._update_out_and_lse_new(
                        prefix_output, prefix_lse_bt, block_out_local2, block_lse_local_bt2)
        # convert lse back to [heads, bs]
        if prefix_lse_bt is not None:
            prefix_lse = prefix_lse_bt
        return prefix_output, prefix_lse


def unified_ascend_attention_with_output(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    wait_for_kv_layer_from_connector(layer_name)
    forward_context: ForwardContext = get_forward_context()
    attn_metadata = forward_context.attn_metadata
    if isinstance(attn_metadata, dict):
        attn_metadata = attn_metadata[layer_name]
    self = forward_context.no_compile_layers[layer_name]
    kv_cache = self.kv_cache[forward_context.virtual_engine]
    self.impl.forward(self,
                      query,
                      key,
                      value,
                      kv_cache,
                      attn_metadata,
                      output,
                      trace_flag=False)
    maybe_save_kv_layer_to_connector(layer_name, kv_cache)
    return


def unified_attention_with_output_fake(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    return


direct_register_custom_op(
    op_name="unified_ascend_attention_with_output",
    op_func=unified_ascend_attention_with_output,
    mutates_args=["output"],
    fake_impl=unified_attention_with_output_fake,
    dispatch_key="PrivateUse1",
)
