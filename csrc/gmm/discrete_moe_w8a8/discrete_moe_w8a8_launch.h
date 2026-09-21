// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#ifndef VLLM_ASCEND_DISCRETE_MOE_W8A8_LAUNCH_H
#define VLLM_ASCEND_DISCRETE_MOE_W8A8_LAUNCH_H

#include <cstdint>

namespace vllm_ascend {
uint32_t discrete_moe_w8a8_gmm_nd_launch(
    void* stream, void* x, void* descriptor, void* groups, void* output, void* up,
    void* tiling, uint32_t rows, uint32_t groupCount, uint32_t cores,
    bool down, uint64_t expectedShape);
uint32_t discrete_moe_w8a8_gmm_nz_launch(
    void* stream, void* x, void* descriptor, void* groups, void* output, void* up,
    void* tiling, uint32_t rows, uint32_t groupCount, uint32_t cores,
    bool down, uint64_t expectedShape);
uint32_t discrete_moe_w8a8_activation_launch(
    void* stream, void* gate, void* up, void* xScale, void* descriptor, void* groups,
    void* output, void* outputScale, void* tiling, uint32_t rows,
    uint32_t groupCount, uint32_t cores, float limit, uint64_t expectedShape);
uint32_t discrete_moe_w8a8_output_launch(
    void* stream, void* input, void* scale, void* descriptor, void* groups,
    void* output, void* tiling, uint32_t rows, uint32_t groupCount, uint32_t cores,
    uint64_t expectedShape);

inline uint64_t discrete_moe_w8a8_launch(
    void* stream, void* x, void* xScale, void* descriptor, void* groupList,
    void* tiling, void* gate, void* up, void* activation, void* activationScale,
    void* down, void* output, uint32_t rows, uint32_t groups,
    uint32_t cubeCores, uint32_t vectorCores, bool isNz, float swigluLimit,
    uint64_t expectedShape)
{
    auto gmm = isNz ? discrete_moe_w8a8_gmm_nz_launch : discrete_moe_w8a8_gmm_nd_launch;
    uint32_t status = gmm(stream, x, descriptor, groupList, gate, up,
        tiling, rows, groups, cubeCores, false, expectedShape);
    if (status != 0) {
        return (uint64_t{1} << 32) | status;
    }
    status = discrete_moe_w8a8_activation_launch(stream, gate, up, xScale, descriptor,
        groupList, activation, activationScale, tiling, rows, groups,
        vectorCores, swigluLimit, expectedShape);
    if (status != 0) {
        return (uint64_t{2} << 32) | status;
    }
    status = gmm(stream, activation, descriptor, groupList,
        down, down, tiling, rows, groups, cubeCores, true, expectedShape);
    if (status != 0) {
        return (uint64_t{3} << 32) | status;
    }
    status = discrete_moe_w8a8_output_launch(stream, down, activationScale, descriptor,
        groupList, output, tiling, rows, groups, vectorCores, expectedShape);
    return status == 0 ? 0 : (uint64_t{4} << 32) | status;
}
}  // namespace vllm_ascend
#endif
