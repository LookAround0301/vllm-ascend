// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
// Exactly one named kernel per translation unit; see the NZ entry comment.
#include "discrete_moe_w8a8_kernel.h"

extern "C" __global__ __aicore__ void discrete_activation_kernel(
    GM_ADDR gate, GM_ADDR up, GM_ADDR xScale, GM_ADDR descriptor, GM_ADDR groups,
    GM_ADDR output, GM_ADDR outputScale, GM_ADDR tiling, uint32_t rows,
    uint32_t groupCount, float limit, uint64_t expectedShape)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    DiscreteActivation op;
    op.Run(gate, up, xScale, descriptor, groups, output, outputScale,
           tiling, rows, groupCount, limit, expectedShape);
}

namespace vllm_ascend {
uint32_t discrete_moe_w8a8_activation_launch(
    void* stream, void* gate, void* up, void* xScale, void* descriptor, void* groups,
    void* output, void* outputScale, void* tiling, uint32_t rows,
    uint32_t groupCount, uint32_t cores, float limit, uint64_t expectedShape)
{
    return discrete_activation_kernel<<<cores, nullptr, stream>>>(
        gate, up, xScale, descriptor, groups, output, outputScale, tiling,
        rows, groupCount, limit, expectedShape);
}

}  // namespace vllm_ascend
