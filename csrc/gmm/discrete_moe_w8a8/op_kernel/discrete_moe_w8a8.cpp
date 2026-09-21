// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#include "discrete_moe_w8a8_kernel.h"

extern "C" __global__ __aicore__ void discrete_gmm_nd_kernel(
    GM_ADDR x, GM_ADDR descriptor, GM_ADDR groups, GM_ADDR output, GM_ADDR up,
    GM_ADDR tiling, uint32_t rows, uint32_t groupCount, bool down, uint64_t expectedShape)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    DiscreteGmm<CubeFormat::ND> op;
    op.Run(x, descriptor, groups, output, up, tiling, rows, groupCount, down, expectedShape);
}

namespace vllm_ascend {
uint32_t discrete_moe_w8a8_gmm_nd_launch(
    void* stream, void* x, void* descriptor, void* groups, void* output, void* up,
    void* tiling, uint32_t rows, uint32_t groupCount, uint32_t cores,
    bool down, uint64_t expectedShape)
{
    return discrete_gmm_nd_kernel<<<cores, nullptr, stream>>>(
        x, descriptor, groups, output, up, tiling, rows, groupCount, down, expectedShape);
}
}  // namespace vllm_ascend
