// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#include "kernel_operator.h"
#include "../discrete_moe_w8a8_tiling.h"
using namespace AscendC;
using namespace vllm_ascend::discrete_moe;
// Persistent descriptors select the shared cache or a fixed slot in the
// current compact buffer. Only active, non-resident, not-yet-prefetched
// experts need copies; dispatch counts stay on the device.
extern "C" __global__ __aicore__ void omoe_prepare_experts(
    GM_ADDR sourceBases, GM_ADDR descriptor, GM_ADDR resident, GM_ADDR ready,
    GM_ADDR groups, GM_ADDR observations, uint32_t matrixBytes, bool cumulative,
    bool observe, uint32_t experts, bool record)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    const uint32_t expert = GetBlockIdx();
    GlobalTensor<int64_t> groupList, counts, sources, addresses;
    GlobalTensor<int32_t> cached, loaded;
    groupList.SetGlobalBuffer((__gm__ int64_t*)groups);
    counts.SetGlobalBuffer((__gm__ int64_t*)observations);
    cached.SetGlobalBuffer((__gm__ int32_t*)resident);
    loaded.SetGlobalBuffer((__gm__ int32_t*)ready);
    // Adjacent scalar stores can share cache lines. Submit this single-AIV
    // commit after the parallel copies instead of letting independent blocks
    // overwrite each other's observation/ready metadata.
    if (record) {
        int64_t previous = 0;
        for (uint32_t expert = 0; expert < experts; ++expert) {
            const int64_t end = groupList.GetValue(expert);
            const int64_t count = cumulative ? end - previous : end;
            previous = end;
            if (observe) {
                counts.SetValue(expert, counts.GetValue(expert) + count);
            }
            if (count != 0 && cached.GetValue(expert) == 0) {
                loaded.SetValue(expert, 1);
            }
        }
        return;
    }
    const int64_t count = groupList.GetValue(expert) -
        (cumulative && expert != 0 ? groupList.GetValue(expert - 1) : 0);
    if (count == 0 || cached.GetValue(expert) != 0 || loaded.GetValue(expert) != 0) {
        return;
    }
    sources.SetGlobalBuffer((__gm__ int64_t*)sourceBases);
    addresses.SetGlobalBuffer((__gm__ int64_t*)descriptor);
    TPipe pipe;
    TBuf<TPosition::VECCALC> scratch;
    constexpr uint32_t COPY_TILE_BYTES = 32768;
    constexpr uint32_t PROJECTION_COUNT = 3;
    pipe.InitBuffer(scratch, COPY_TILE_BYTES);
    LocalTensor<uint8_t> tile = scratch.Get<uint8_t>();
    for (uint32_t projection = 0; projection < PROJECTION_COUNT; ++projection) {
        GlobalTensor<uint8_t> source, destination;
        source.SetGlobalBuffer((__gm__ uint8_t*)(sources.GetValue(projection) +
            static_cast<uint64_t>(expert) * matrixBytes));
        destination.SetGlobalBuffer((__gm__ uint8_t*)addresses.GetValue(
            expert * DESCRIPTOR_COLUMNS + projection));
        for (uint32_t offset = 0; offset < matrixBytes; offset += COPY_TILE_BYTES) {
            const uint32_t bytes = matrixBytes - offset < COPY_TILE_BYTES ? matrixBytes - offset : COPY_TILE_BYTES;
            DataCopy(tile, source[offset], bytes);
            PipeBarrier<PIPE_ALL>();
            DataCopy(destination[offset], tile, bytes);
            PipeBarrier<PIPE_ALL>();
        }
    }
}
extern "C" uint32_t omoe_prepare_experts_launch(
    void* stream, void* sourceBases, void* descriptor, void* resident, void* ready,
    void* groups, void* observations, uint32_t matrixBytes, uint32_t experts,
    bool cumulative, bool observe)
{
    const auto status = omoe_prepare_experts<<<experts, nullptr, stream>>>((GM_ADDR)sourceBases,
        (GM_ADDR)descriptor, (GM_ADDR)resident, (GM_ADDR)ready, (GM_ADDR)groups,
        (GM_ADDR)observations, matrixBytes, cumulative, observe, experts, false);
    if (status != 0) {
        return status;
    }
    return omoe_prepare_experts<<<1, nullptr, stream>>>((GM_ADDR)sourceBases,
        (GM_ADDR)descriptor, (GM_ADDR)resident, (GM_ADDR)ready, (GM_ADDR)groups,
        (GM_ADDR)observations, matrixBytes, cumulative, observe, experts, true);
}
