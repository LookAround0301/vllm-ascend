from dataclasses import dataclass
from typing import Optional

import torch
from vllm_ascend.worker.kvcomp_utils import KVCompConfig
from vllm_ascend.worker.kvcomp_utils import HashEncoder


@dataclass
class AscendKvcompSparseMetadata:
    # kvcomp
    hash_topk_config: KVCompConfig = None #hashtopk config initilized by moderrunner, which is fixed for all requests and all layers, contaon top_k
    hash_encoder: HashEncoder = None # Hash encoder for hash topk attention initilized by moderrunner, which is fixed for all requests and all layers
    hashk_cache: torch.Tensor = None # Hashk cache for hash topk attention, which refers to the hashk_cache in forward_contex
    # num_decode_tokens_device = torch.tensor([self.num_decode_tokens], dtype=torch.int32) \
    #                                 .to(device=device, non_blocking=True)  
    # num_prefill_tokens_device = torch.tensor([self.num_actual_tokens-self.num_decode_tokens], dtype=torch.int32) \
    #                                 .to(device=device, non_blocking=True) 
    num_decode_tokens_device: torch.Tensor = None  
    num_prefill_tokens_device: torch.Tensor = None  
    num_actual_tokens_device: torch.Tensor = None  
