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

import torch
import torch_npu
from typing import Optional, Tuple

#import logging
#logger = logging.getLogger(__name__)


class HashEncoder:
    """
    Hash encoder that projects input vectors into binary hash codes, which is further packed into unit8 numbers.
    
    According to the HATA paper:
    hash(q) = sign(qW)
    where W is a Gaussian random matrix and sign(x) = 1 if x >= 0, 0 otherwise.
    
    This version packs bits into uint8 (unsigned 8-bit integers).
    """
    
    def __init__(self, input_dim: int, hash_bits: int, dtype: torch.dtype, device: torch.device, is_orthogonal_matrix: bool = False, reduce_dim: int = None):
        """
        Initilize the hash weight matrix and its shape.
        
        Args:
            input_dim (int): head_size of the model
            hash_bits (int): the number of hash bits for each input q or k
            dtype: the hash weight's dtype, only supports FLOAT, FLOAT16, and DOUBLE
            device: the device to store the hash weight
        
        Example:
            hash_encoder = HashEncoder(128, 128, 'npu:0')
            W matrix will be a random matrix of shape (128, 128), dtype=float32, and device='npu:0'
            
            hash_encoder = HashEncoder(128, 128, 'npu:0', True)
            W matrix will be an orthogonal matrix of shape (128, 128), dtype=float32, and device='npu:0'
            
        """
        self.input_dim = input_dim
        self.hash_bits = hash_bits
        self.reduce_dim = reduce_dim
        if self.reduce_dim is not None:
            self.hash_numbers = reduce_dim // 8
        else:
            self.hash_numbers = hash_bits // 8  #the number of uint8 numbers after packing hash bits 

        if self.hash_bits != self.input_dim:
            #logger.warning(f"hash_bits {self.hash_bits} != input_dim {self.input_dim}, this may cause performance degradation")
            print(f"hash_bits {self.hash_bits} != input_dim {self.input_dim}, this may cause performance degradation")
        
        if dtype not in [torch.float16, torch.float32, torch.float64]:
            print(f"the input dtype={dtype} is not supported, only support FLOAT, FLOAT16, and DOUBLE for storing hash_weight")
            print("automatically set dtype=torch.float16 now!")
            dtype = torch.float16
            
        self.dtype = dtype 
           
        self.device = device
        
        if is_orthogonal_matrix:
            # Step 1: 随机高斯矩阵
            random_weights = torch.normal(
                                mean=0,
                                std=2,
                                size=(self.input_dim, self.hash_bits),
                                dtype=self.dtype,
                                device=self.device)

            # Step 2: QR 分解
            Q, R = torch.linalg.qr(random_weights)
            
            # Step 3: 调整符号，保证 Haar 分布
            d = torch.sign(torch.diag(R))

            self.hash_weight = Q * d
        else: 
            # Initialize Gaussian random weight matrix with mean 0 and standard deviation 2
            #  W.shape = (input_dim, hash_bits)
            # call aclnnNormalFloatFloat, the output only supports dtype-list [DT_FLOAT16,DT_FLOAT,DT_DOUBLE]
            # https://www.hiascend.com/document/detail/zh/canncommercial/81RC1/apiref/aolapi/context/aclnnNormalFloatTensor.md
            self.hash_weight = torch.normal(mean=0, std=2, size=(input_dim, hash_bits), dtype=dtype, device=device)

        if self.reduce_dim is not None:
            self.hash_weight = self.hash_weight[:,:self.reduce_dim]
        self.powers = torch.tensor([1 << i for i in range(8)], dtype=torch.int32, device=device)
    
    def set_hash_weight(self, w):
        """manually set hash weight matrix"""
        assert w.dtype == self.dtype, f"the input hash weight matrix w.dtype={w.dtype}, expect {self.dtype}"
        
        assert w.device == self.device, f"the input hash weight matrix w.device={w.device}, expect {self.device}"
        
        assert w.shape[0] == self.input_dim, f"w.shape[0]={w.shape[0]} is not equal to input_dim={self.input_dim}"
        
        assert w.shape[1] == self.hash_bits, f"w.shape[1]={w.shape[1]} is not equal to input_dim={self.hash_bits}"
        
        #deep copy here 
        self.hash_weight.copy_(w)
        
    def sign_pack_torch(self, in_data: torch.Tensor, size: int):
        """
        将 in_data（GPU Tensor）的符号位打包成 uint8 Tensor，并 reshape 成 (size, -1)。
        与 numpy 版本语义一致，但全程在 GPU 上完成。
        """
        device = in_data.device
        n = in_data.numel()

        # 1. 取符号 → 0/1
        bits = (torch.sign(in_data).int() + 1) // 2   # 0/1 的 int32

        # 2. 补齐到 8 的倍数
        pad = (-n) & 7
        if pad:
            bits = torch.nn.functional.pad(bits.view(-1), (0, pad), value=0)

        # 3. 按 little-endian 每 8 位打包
        bits = bits.view(-1, 8)                       # [N, 8]
        # 构造 8 位权值 [1, 2, 4, ..., 128]
        packed = (bits * self.powers).sum(dim=1).to(torch.uint8)  # [N]

        # 4. reshape 成 (size, -1)
        return packed.view(size, -1)

    def compute_hash(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input vectors into binary hash codes.
        
        Args:
            x: Input tensor of shape [..., input_dim]
            
        Returns:
            hash_codes: Binary hash codes of shape [..., hash_bits//8] = [..., hash_numbers]
                       Packed into 8-bit unsigned integers (uint8), all 8 bits used
    
        """
        input_dim = x.shape[-1]
        assert input_dim == self.input_dim, f"Expected input_dim {self.input_dim}, got {input_dim}"
        
        # if x.shape = (s1,s2,s3,input_dim), then orig_shape=(s1,s2,s3)
        orig_shape = x.shape[:-1]
        
        #x_falt.shape = (s1*s2*s3, input_dim)
        # x_flat = x.view(-1, input_dim)
        x_flat = x.reshape(-1, input_dim)
        
        # matrix multiplication of x_flat and W, xW.shape=(s1*s2*s3,hash_bits)
        if x.dtype == self.dtype:
            xW = torch.matmul(x_flat, self.hash_weight)  # [..., hash_bits]
        else:
            #print(f"the input tensor x.dtype={x.dtype} is not equal to W.dtype={self.dtype}, automallty convert to {self.dtype}")
            xW = torch.matmul(x_flat.to(self.dtype), self.hash_weight)
        
        xW_flat = xW.view(-1)
        
        # torch_npu.npu_sign_bits_pack only supports float32 and float16 in A2&A3
        # see https://www.hiascend.com/document/detail/zh/Pytorch/700/apiref/apilist/ptaoplist_000494.html
        # packed_code.shape = (s1*s2*s3*hash_numbers) where hash_numbers = hash_bits//8, dtype=uint8
        packed_codes_flat = torch_npu.npu_sign_bits_pack(xW_flat, size=1)
        # packed_codes_flat = self.sign_pack_torch(xW_flat, size=1)


        # out_shape=(s1,s2,s3,hash_numbers)
        out_shape = orig_shape + (self.hash_numbers,)
        
        # packed_codes.shape=(s1,s2,s3,hash_numbers)
        packed_codes = packed_codes_flat.view(*out_shape)

        return packed_codes
    
   
    
    def _unpack_hash(self, packed_codes: torch.Tensor) -> torch.Tensor:
        """
        Unpack 8-bit unsigned integers into binary codes (0/1)
        Args:
            packed_codes: Packed codes of shape [..., hash_bits//8] (uint8)
        Returns:
            binary_codes: Binary codes of shape [..., hash_bits] (1 or -1 values, float16)
        """
        if packed_codes.dtype != torch.uint8:
            raise ValueError("packed_codes must be of dtype torch.uint8")
        
        orig_shape = packed_codes.shape[:-1]
        
        assert packed_codes.shape[-1] == self.hash_numbers, f"packed_codes.shape[-1]=\
                                     {packed_codes.shape[-1]}, expected {self.hash_numbers}"
        
        packed_codes_flat = packed_codes.view(-1)
        
        #see https://www.hiascend.com/document/detail/zh/Pytorch/700/apiref/apilist/ptaoplist_000495.html
        # binary_codes_flat is 1D tensor with dtype=torch.float16
        binary_codes_flat = torch_npu.npu_sign_bits_unpack(packed_codes_flat,size=1, dtype=torch.float16)
        
        # Reshape back to original leading shape
        out_shape = orig_shape + (self.hash_numbers * 8,)
        
        binary_codes = binary_codes_flat.view(*out_shape)
        
        return binary_codes
    


if __name__ == "__main__":
    #test HashEncoder
    print("Running simple HashEncoder test...")
    input_dim = 8
    hash_bits = 8
    batch_shape = (2,)
    dtype = torch.bfloat16
    device = "npu" if torch.npu.is_available() else "cpu"
    torch.manual_seed(42)
    
    # Create random input
    x = torch.randn(*batch_shape, input_dim, dtype=dtype, device=device)
    print("x shape:", x.shape)
    #print("x:", x)
    encoder = HashEncoder(input_dim=input_dim, hash_bits=hash_bits, dtype=dtype, device=device)

    x_hash = encoder.compute_hash(x)    
    print(f"x_hash.shape={x_hash.shape}, x_hash.dtype={x_hash.dtype}, x_hash.device={x_hash.device}")
    print("x_hash:", x_hash)
    
    binary_codes = encoder._unpack_hash(x_hash)
    print(f"binary_codes.shape={binary_codes.shape}, binary_codes.dtype={binary_codes.dtype}, binary_codes.device={binary_codes.device}")
    print("binary_codes:", binary_codes)
   
    print(f"x_hash[0]={bin(x_hash[0].item())}")
    print(f"binary_codes[0]>0 = {binary_codes[0]>0}")
    
    print(f"x_hash[1]={bin(x_hash[1].item())}")
    print(f"binary_codes[1]>0 = {binary_codes[1]>0}")
    
    #here is the output
    """
    Running simple HashEncoder test...
    x shape: torch.Size([2, 8])
    the input dtype=torch.bfloat16 is not supported, only support FLOAT, FLOAT16, and DOUBLE for storing hash_weight
    automatically set dtype=torch.float16 now!
    the inptu tensor x.dtype=torch.bfloat16 is not equal to W.dtype=torch.float16, automallty convert to torch.float16
    x_hash.shape=torch.Size([2, 1]), x_hash.dtype=torch.uint8, x_hash.device=npu:0
    x_hash: tensor([[109],
            [210]], device='npu:0', dtype=torch.uint8)
    binary_codes.shape=torch.Size([2, 8]), binary_codes.dtype=torch.float16, binary_codes.device=npu:0
    [W906 22:07:32.331039460 compiler_depend.ts:164] Warning: Warning: Device do not support double dtype now, dtype cast repalce with float. (function operator())
    binary_codes: tensor([[ 1., -1.,  1.,  1., -1.,  1.,  1., -1.],
            [-1.,  1., -1., -1.,  1., -1.,  1.,  1.]], device='npu:0',
        dtype=torch.float16)
    x_hash[0]=0b1101101
    binary_codes[0]>0 = tensor([ True, False,  True,  True, False,  True,  True, False],
        device='npu:0')
    x_hash[1]=0b11010010
    binary_codes[1]>0 = tensor([False,  True, False, False,  True, False,  True,  True],
        device='npu:0')
    """
    
    #Clearly, it works now. Since x_hash[0]=0b1101101 = [0 (MSB) 1 1 0 1 1 0 1 (LSB)] 
    #                                                 = [False (MSB), True, True, False, True, True, False, True (LSB)] 
    # similar for x_hash[1]
    