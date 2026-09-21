// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#ifndef VLLM_ASCEND_DISCRETE_MOE_W8A8_TILING_H
#define VLLM_ASCEND_DISCRETE_MOE_W8A8_TILING_H

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"

namespace vllm_ascend::discrete_moe {

constexpr uint32_t DESCRIPTOR_COLUMNS = 8;
constexpr uint32_t GATE_ADDRESS = 0;
constexpr uint32_t UP_ADDRESS = 1;
constexpr uint32_t DOWN_ADDRESS = 2;
constexpr uint32_t GATE_SCALE_ADDRESS = 3;
constexpr uint32_t UP_SCALE_ADDRESS = 4;
constexpr uint32_t DOWN_SCALE_ADDRESS = 5;
constexpr uint32_t GATE_ROW_STRIDE = 6;
constexpr uint32_t UP_ROW_STRIDE = 7;
// Bounds for the current full-row vector buffers, not fixed model dimensions.
// Actual H/I and core counts are supplied through DiscreteMoeTiling.
constexpr uint32_t MAX_INTERMEDIATE_SIZE = 4096;
constexpr uint32_t MAX_HIDDEN_SIZE = 8192;
constexpr uint32_t MATRIX_ALIGNMENT = 32;
constexpr uint32_t TILING_STORAGE_BYTES = 1024;

// Prepared once for a shape/layout, independently of token counts and expert
// addresses. This is NOT an allocation table and owns no expert memory.
struct DiscreteMoeTiling {
    uint32_t hiddenSize;
    uint32_t intermediateSize;
    uint32_t tileM;
    uint32_t tileN;
    uint32_t cubeCores;
    uint32_t vectorCores;
    uint32_t isNz;
    uint32_t reserved;
    AscendC::tiling::TCubeTiling gate;
    AscendC::tiling::TCubeTiling down;
};
static_assert(sizeof(DiscreteMoeTiling) <= TILING_STORAGE_BYTES);

}  // namespace vllm_ascend::discrete_moe
#endif
