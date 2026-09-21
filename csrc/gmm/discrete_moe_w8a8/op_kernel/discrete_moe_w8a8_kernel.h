// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
// Descriptor-addressed INT8 expert computation. Each matrix has independent
// storage; all expert groups share a shape and physical ND/NZ layout.

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "../discrete_moe_w8a8_tiling.h"

using namespace AscendC;
using namespace matmul;
using namespace vllm_ascend::discrete_moe;

namespace {
constexpr float QUANT_MAX = 127.0f;
constexpr uint32_t VECTOR_SCRATCH_BYTES = 256;

constexpr MatmulConfig DISCRETE_MM_CONFIG = GetMDLConfig();

__aicore__ inline uint32_t CeilDiv(uint32_t value, uint32_t divisor)
{
    return (value + divisor - 1) / divisor;
}

__aicore__ inline void ReadTiling(GM_ADDR address, DiscreteMoeTiling& data)
{
    auto source = reinterpret_cast<__gm__ uint32_t*>(address);
    auto target = reinterpret_cast<uint32_t*>(&data);
    for (uint32_t i = 0; i < sizeof(DiscreteMoeTiling) / sizeof(uint32_t); ++i) {
        target[i] = source[i];
    }
}

__aicore__ inline bool MatchesShape(const DiscreteMoeTiling& data, uint64_t expected)
{
    const uint64_t actual = static_cast<uint64_t>(data.hiddenSize) |
        (static_cast<uint64_t>(data.intermediateSize) << 16) |
        (static_cast<uint64_t>(data.isNz) << 32);
    return actual == expected;
}

// Device values are deliberately not copied to the CPU. Invalid ranges are
// skipped rather than clamped into a different (incorrect) expert routing.
// The public contract requires monotonic cumulative ends with final end M.
__aicore__ inline bool ReadRange(GlobalTensor<int64_t>& groups, uint32_t group,
                                 uint32_t rows, uint32_t& start, uint32_t& count)
{
    int64_t begin = group == 0 ? 0 : groups.GetValue(group - 1);
    int64_t end = groups.GetValue(group);
    if (begin < 0 || end < begin || end > static_cast<int64_t>(rows)) {
        return false;
    }
    start = static_cast<uint32_t>(begin);
    count = static_cast<uint32_t>(end - begin);
    return count != 0;
}

__aicore__ inline bool ValidGroups(GlobalTensor<int64_t>& groups,
                                    uint32_t groupCount, uint32_t rows)
{
    int64_t previous = 0;
    for (uint32_t group = 0; group < groupCount; ++group) {
        const int64_t end = groups.GetValue(group);
        if (end < previous || end > static_cast<int64_t>(rows)) {
            return false;
        }
        previous = end;
    }
    return previous == static_cast<int64_t>(rows);
}

template <typename T>
__aicore__ inline void BindAddress(GlobalTensor<T>& tensor,
                                   GlobalTensor<int64_t>& descriptor,
                                   uint32_t group, uint32_t column)
{
    tensor.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(
        descriptor.GetValue(static_cast<uint64_t>(group) * DESCRIPTOR_COLUMNS + column)));
}

template <CubeFormat FORMAT>
class DiscreteGmm {
    using A = MatmulType<TPosition::GM, CubeFormat::ND, int8_t, false>;
    using B = MatmulType<TPosition::GM, FORMAT, int8_t, false>;
    using C = MatmulType<TPosition::GM, CubeFormat::ND, int32_t>;
    using Mm = MatmulImpl<A, B, C, C, DISCRETE_MM_CONFIG>;

public:
    __aicore__ inline void Run(GM_ADDR input, GM_ADDR descriptor, GM_ADDR groupList,
                                GM_ADDR output, GM_ADDR upOutput, GM_ADDR tiling,
                                uint32_t rows, uint32_t groupCount, bool down,
                                uint64_t expectedShape)
    {
        DiscreteMoeTiling data;
        ReadTiling(tiling, data);
        if (!MatchesShape(data, expectedShape)) {
            return;
        }
        const uint32_t k = down ? data.intermediateSize : data.hiddenSize;
        const uint32_t n = down ? data.hiddenSize : data.intermediateSize;
        const uint32_t core = GetBlockIdx();
        const uint32_t cores = GetBlockNum();
        GlobalTensor<int8_t> x;
        GlobalTensor<int64_t> desc, groups;
        GlobalTensor<int32_t> y, up;
        x.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(input));
        desc.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(descriptor));
        groups.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(groupList));
        if (!ValidGroups(groups, groupCount, rows)) {
            return;
        }
        y.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(output));
        up.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(upOutput));
        TPipe pipe;
        Mm mm;
        mm.SetSubBlockIdx(0);
        mm.Init(down ? &data.down : &data.gate, &pipe);
        uint64_t taskBase = 0;
        for (uint32_t group = 0; group < groupCount; ++group) {
            uint32_t start = 0, count = 0;
            if (!ReadRange(groups, group, rows, start, count)) {
                continue;
            }
            const uint32_t mTiles = CeilDiv(count, data.tileM);
            const uint32_t nTiles = CeilDiv(n, data.tileN);
            const uint32_t projections = down ? 1 : 2;
            const uint64_t groupTasks = static_cast<uint64_t>(mTiles) * nTiles * projections;
            uint64_t firstTask = (core + cores - taskBase % cores) % cores;
            for (uint64_t task = firstTask; task < groupTasks; task += cores) {
                const uint32_t projection = task % projections;
                const uint64_t matrixTask = task / projections;
                const uint32_t m0 = (matrixTask / nTiles) * data.tileM;
                const uint32_t n0 = (matrixTask % nTiles) * data.tileN;
                const uint32_t curM = count - m0 < data.tileM ? count - m0 : data.tileM;
                const uint32_t curN = n - n0 < data.tileN ? n - n0 : data.tileN;
                GlobalTensor<int8_t> weight;
                BindAddress(weight, desc, group, down ? DOWN_ADDRESS : projection);
                const uint32_t stride = down ? n : static_cast<uint32_t>(desc.GetValue(
                    static_cast<uint64_t>(group) * DESCRIPTOR_COLUMNS + GATE_ROW_STRIDE + projection));
                // In NZ a column tile is a contiguous K x Ntile fragment. ND
                // fused W13 retains its physical 2I row stride without copies.
                const uint64_t weightOffset = FORMAT == CubeFormat::NZ
                    ? static_cast<uint64_t>(n0) * CeilDiv(k, 16) * 16 : n0;
                const uint32_t orgN = FORMAT == CubeFormat::NZ ? n : stride;
                auto result = projection == 1 && !down ? up : y;
                mm.SetOrgShape(count, orgN, k, k, n);
                mm.SetSingleShape(curM, curN, k);
                mm.SetTensorA(x[static_cast<uint64_t>(start + m0) * k], false);
                mm.SetTensorB(weight[weightOffset], false);
                mm.IterateAll(result[static_cast<uint64_t>(start + m0) * n + n0],
                              0, false, true);
            }
            taskBase += groupTasks;
        }
        mm.End();
    }
};

class DiscreteActivation {
public:
    __aicore__ inline void Run(GM_ADDR gate, GM_ADDR up, GM_ADDR xScale,
                                GM_ADDR descriptor, GM_ADDR groupList,
                                GM_ADDR output, GM_ADDR outputScale,
                                GM_ADDR tiling, uint32_t rows, uint32_t groupCount,
                                float limit, uint64_t expectedShape)
    {
        DiscreteMoeTiling data;
        ReadTiling(tiling, data);
        if (!MatchesShape(data, expectedShape)) {
            return;
        }
        const uint32_t width = data.intermediateSize;
        GlobalTensor<int32_t> gateGm, upGm;
        GlobalTensor<int64_t> desc, groups;
        GlobalTensor<float> xs, os;
        GlobalTensor<int8_t> out;
        gateGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(gate));
        upGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(up));
        desc.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(descriptor));
        groups.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(groupList));
        if (!ValidGroups(groups, groupCount, rows)) {
            return;
        }
        xs.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(xScale));
        os.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(outputScale));
        out.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(output));
        TPipe pipe;
        TQue<QuePosition::VECIN, 1> inputQueue;
        TQue<QuePosition::VECOUT, 1> outputQueue;
        TBuf<TPosition::VECCALC> scratch;
        TBuf<TPosition::VECCALC> channelCache;
        pipe.InitBuffer(inputQueue, 1, width * sizeof(int32_t) * 2);
        pipe.InitBuffer(outputQueue, 1, width);
        pipe.InitBuffer(scratch, width * sizeof(float) * 4 + VECTOR_SCRATCH_BYTES);
        pipe.InitBuffer(channelCache, width * sizeof(float) * 2);
        for (uint32_t group = 0; group < groupCount; ++group) {
            uint32_t start = 0, count = 0;
            if (!ReadRange(groups, group, rows, start, count)) {
                continue;
            }
            uint32_t offset = (GetBlockIdx() + GetBlockNum() - start % GetBlockNum()) % GetBlockNum();
            if (offset >= count) {
                continue;
            }
            GlobalTensor<float> gateScale, upScale;
            BindAddress(gateScale, desc, group, GATE_SCALE_ADDRESS);
            BindAddress(upScale, desc, group, UP_SCALE_ADDRESS);
            // Channel scales are invariant for every row of this expert.
            // Keep them separate from row scratch, which quantization reuses.
            auto gateChannels = channelCache.Get<float>();
            auto upChannels = gateChannels[width];
            DataCopy(gateChannels, gateScale, width);
            DataCopy(upChannels, upScale, width);
            PipeBarrier<PIPE_ALL>();
            for (; offset < count; offset += GetBlockNum()) {
                const uint32_t row = start + offset;
                auto input = inputQueue.AllocTensor<int32_t>();
                DataCopy(input, gateGm[static_cast<uint64_t>(row) * width], width);
                DataCopy(input[width], upGm[static_cast<uint64_t>(row) * width], width);
                inputQueue.EnQue(input);
                input = inputQueue.DeQue<int32_t>();
                auto g = scratch.Get<float>();
                auto u = g[width];
                auto act = g[width * 2];
                auto temp = g[width * 3];
                Cast(g, input, RoundMode::CAST_NONE, width);
                Cast(u, input[width], RoundMode::CAST_NONE, width);
                PipeBarrier<PIPE_V>();
                const float tokenScale = xs.GetValue(row);
                Muls(g, g, tokenScale, width);
                Muls(u, u, tokenScale, width);
                PipeBarrier<PIPE_V>();
                Mul(g, g, gateChannels, width);
                Mul(u, u, upChannels, width);
                PipeBarrier<PIPE_V>();
                if (limit > 0.0f) {
                    Mins(g, g, limit, width);
                    Mins(u, u, limit, width);
                    PipeBarrier<PIPE_V>();
                    Maxs(u, u, -limit, width);
                    PipeBarrier<PIPE_V>();
                }
                // Keep the original fused W8A8 SwiGLU evaluation order.
                // Algebraically equivalent Sigmoid/Mul sequences can differ
                // in FP32 and move later FP16/INT8/BF16 rounding boundaries.
                SwiGLU<float, false>(act, u, g, 1.0f, width);
                PipeBarrier<PIPE_V>();
                Abs(temp, act, width);
                PipeBarrier<PIPE_V>();
                ReduceMax(temp, temp, temp, width, false);
                PipeBarrier<PIPE_ALL>();
                const float maximum = temp.GetValue(0);
                const float scale = maximum / QUANT_MAX;
                const float inverse = maximum > 0.0f ? 1.0f / scale : 0.0f;
                Muls(act, act, inverse, width);
                PipeBarrier<PIPE_V>();
                auto quant = outputQueue.AllocTensor<int8_t>();
                // Match the original 0812 W8A8 quantizer: FP32 -> FP16 ->
                // INT8, with round-to-nearest at both conversions. Direct
                // FP32 -> INT32 rounding differs near half-integer bins
                // (e.g. 125.49 -> FP16 125.5 -> INT8 126, not 125).
                // Up is dead; keep this scratch disjoint from unread act.
                auto halfValue = u.ReinterpretCast<half>();
                Cast(halfValue, act, RoundMode::CAST_RINT, width);
                PipeBarrier<PIPE_V>();
                Cast(quant, halfValue, RoundMode::CAST_RINT, width);
                outputQueue.EnQue(quant);
                quant = outputQueue.DeQue<int8_t>();
                DataCopy(out[static_cast<uint64_t>(row) * width], quant, width);
                temp.SetValue(0, scale);
                PipeBarrier<PIPE_ALL>();
                DataCopyExtParams scaleCopy{1, sizeof(float), 0, 0, 0};
                DataCopyPad(os[row], temp, scaleCopy);
                // temp is a scratch buffer, not an output queue tensor. Its
                // scale DMA must complete before the next row reuses scratch.
                PipeBarrier<PIPE_ALL>();
                inputQueue.FreeTensor(input);
                outputQueue.FreeTensor(quant);
            }
        }
    }
};

class DiscreteOutput {
public:
    __aicore__ inline void Run(GM_ADDR input, GM_ADDR scale, GM_ADDR descriptor,
                                GM_ADDR groupList, GM_ADDR output, GM_ADDR tiling,
                                uint32_t rows, uint32_t groupCount, uint64_t expectedShape)
    {
        DiscreteMoeTiling data;
        ReadTiling(tiling, data);
        if (!MatchesShape(data, expectedShape)) {
            return;
        }
        const uint32_t width = data.hiddenSize;
        GlobalTensor<int32_t> in;
        GlobalTensor<float> tokenScales;
        GlobalTensor<int64_t> desc, groups;
        GlobalTensor<bfloat16_t> out;
        in.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(input));
        tokenScales.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(scale));
        desc.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(descriptor));
        groups.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(groupList));
        if (!ValidGroups(groups, groupCount, rows)) {
            return;
        }
        out.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(output));
        TPipe pipe;
        TQue<QuePosition::VECIN, 1> inputQueue;
        TQue<QuePosition::VECOUT, 1> outputQueue;
        TBuf<TPosition::VECCALC> scratch;
        pipe.InitBuffer(inputQueue, 1, width * sizeof(int32_t));
        pipe.InitBuffer(outputQueue, 1, width * sizeof(bfloat16_t));
        pipe.InitBuffer(scratch, width * sizeof(float) * 2);
        for (uint32_t group = 0; group < groupCount; ++group) {
            uint32_t start = 0, count = 0;
            if (!ReadRange(groups, group, rows, start, count)) {
                continue;
            }
            uint32_t offset = (GetBlockIdx() + GetBlockNum() - start % GetBlockNum()) % GetBlockNum();
            if (offset >= count) {
                continue;
            }
            GlobalTensor<float> channelScale;
            BindAddress(channelScale, desc, group, DOWN_SCALE_ADDRESS);
            // The second half of scratch is read-only throughout a group.
            auto scales = scratch.Get<float>()[width];
            // This cache is independent of inputQueue. Complete the previous
            // group's vector reads before MTE2 overwrites its channel scales.
            PipeBarrier<PIPE_ALL>();
            DataCopy(scales, channelScale, width);
            PipeBarrier<PIPE_ALL>();
            for (; offset < count; offset += GetBlockNum()) {
                const uint32_t row = start + offset;
                auto inputLocal = inputQueue.AllocTensor<int32_t>();
                DataCopy(inputLocal, in[static_cast<uint64_t>(row) * width], width);
                inputQueue.EnQue(inputLocal);
                inputLocal = inputQueue.DeQue<int32_t>();
                auto value = scratch.Get<float>();
                Cast(value, inputLocal, RoundMode::CAST_NONE, width);
                PipeBarrier<PIPE_V>();
                // Match original GMM2: apply the BF16-exact channel scale
                // first, then the FP32 token scale. Reassociation changes
                // rare BF16 ties even when the INT32 accumulator is exact.
                Mul(value, value, scales, width);
                PipeBarrier<PIPE_V>();
                Muls(value, value, tokenScales.GetValue(row), width);
                PipeBarrier<PIPE_V>();
                auto outputLocal = outputQueue.AllocTensor<bfloat16_t>();
                Cast(outputLocal, value, RoundMode::CAST_RINT, width);
                outputQueue.EnQue(outputLocal);
                outputLocal = outputQueue.DeQue<bfloat16_t>();
                DataCopy(out[static_cast<uint64_t>(row) * width], outputLocal, width);
                inputQueue.FreeTensor(inputLocal);
                outputQueue.FreeTensor(outputLocal);
            }
        }
    }
};
}  // namespace
