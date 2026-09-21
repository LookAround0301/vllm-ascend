// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#include "discrete_moe_w8a8_kernel.h"

extern "C" __global__ __aicore__ void discrete_output_kernel(
    GM_ADDR input, GM_ADDR scale, GM_ADDR descriptor, GM_ADDR groups,
    GM_ADDR output, GM_ADDR tiling, uint32_t rows, uint32_t groupCount, uint64_t expectedShape)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    DiscreteOutput op;
    op.Run(input, scale, descriptor, groups, output, tiling, rows, groupCount, expectedShape);
}

namespace vllm_ascend {
uint32_t discrete_moe_w8a8_output_launch(
    void* stream, void* input, void* scale, void* descriptor, void* groups,
    void* output, void* tiling, uint32_t rows, uint32_t groupCount, uint32_t cores,
    uint64_t expectedShape)
{
    return discrete_output_kernel<<<cores, nullptr, stream>>>(
        input, scale, descriptor, groups, output, tiling, rows, groupCount, expectedShape);
}
}  // namespace vllm_ascend
