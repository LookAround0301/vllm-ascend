// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#ifndef VLLM_ASCEND_DISCRETE_MOE_W8A8_TORCH_ADPT_H
#define VLLM_ASCEND_DISCRETE_MOE_W8A8_TORCH_ADPT_H

#include <torch/extension.h>
#include <acl/acl.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <tuple>
#include <utility>
#include "torch_npu/csrc/core/npu/NPUGuard.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "torch_npu/csrc/core/npu/NPUFormat.h"
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"
#include "discrete_moe_w8a8_tiling.h"
#include "discrete_moe_w8a8_launch.h"

namespace vllm_ascend {
namespace discrete_moe {

constexpr int64_t MAX_TILE_N = 256;

inline void CheckShape(int64_t hiddenSize, int64_t intermediateSize)
{
    TORCH_CHECK(hiddenSize > 0 && hiddenSize <= MAX_HIDDEN_SIZE &&
                    hiddenSize % MATRIX_ALIGNMENT == 0,
                "discrete W8A8 requires H divisible by 32 and 0 < H <= ", MAX_HIDDEN_SIZE);
    TORCH_CHECK(intermediateSize > 0 && intermediateSize <= MAX_INTERMEDIATE_SIZE &&
                    intermediateSize % MATRIX_ALIGNMENT == 0,
                "discrete W8A8 requires I divisible by 32 and 0 < I <= ", MAX_INTERMEDIATE_SIZE);
}

inline void CheckDeviceTensor(const at::Tensor& tensor, const at::Tensor& reference,
                               at::ScalarType dtype, const char* name)
{
    TORCH_CHECK(tensor.device() == reference.device(), name, " must be on the input NPU");
    TORCH_CHECK(tensor.scalar_type() == dtype, name, " has incorrect dtype");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
    const auto format = at_npu::native::get_npu_format(tensor);
    TORCH_CHECK(format == ACL_FORMAT_ND || format == ACL_FORMAT_NCHW,
                name, " must use unpadded ND storage");
}

inline void BuildMatmulTiling(AscendC::tiling::TCubeTiling& result,
                              const platform_ascendc::PlatformAscendC& platform,
                              int64_t k, int64_t n, bool isNz,
                              int64_t tileM, int64_t tileN)
{
    matmul_tiling::MatmulApiTiling tiler(platform);
    tiler.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                  matmul_tiling::DataType::DT_INT8);
    tiler.SetBType(matmul_tiling::TPosition::GM,
                  isNz ? matmul_tiling::CubeFormat::NZ : matmul_tiling::CubeFormat::ND,
                  matmul_tiling::DataType::DT_INT8);
    tiler.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                  matmul_tiling::DataType::DT_INT32);
    tiler.SetBias(false);
    tiler.SetShape(tileM, tileN, k);
    tiler.SetOrgShape(tileM, n, k);
    tiler.SetBufferSpace(-1, -1, -1);
    TORCH_CHECK(tiler.GetTiling(result) == 0, "discrete W8A8 MatmulApiTiling failed");
}

}  // namespace discrete_moe

inline at::Tensor prepare_discrete_moe_w8a8_tiling(
    const at::Tensor& deviceTensor, int64_t hiddenSize, int64_t intermediateSize,
    bool isNz, int64_t tileM, int64_t tileN)
{
    using namespace discrete_moe;
    TORCH_CHECK(deviceTensor.device().type() == c10::DeviceType::PrivateUse1,
                "discrete W8A8 tiling requires an NPU tensor");
    CheckShape(hiddenSize, intermediateSize);
    TORCH_CHECK(tileM > 0 && tileM <= 128 && tileM % 16 == 0,
                "tile_m must be a multiple of 16 in [16, 128]");
    TORCH_CHECK(tileN > 0 && tileN <= MAX_TILE_N && tileN % 32 == 0,
                "tile_n must be a multiple of 32 in [32, ", MAX_TILE_N, "]");
    c10_npu::NPUGuard deviceGuard(deviceTensor.device());
    auto* platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    TORCH_CHECK(platform != nullptr, "cannot discover Ascend tiling platform");
    DiscreteMoeTiling data{};
    data.hiddenSize = hiddenSize;
    data.intermediateSize = intermediateSize;
    data.tileM = tileM;
    data.tileN = tileN;
    data.cubeCores = platform->GetCoreNumAic();
    data.vectorCores = platform->GetCoreNumAiv();
    data.isNz = isNz;
    TORCH_CHECK(data.cubeCores > 0 && data.vectorCores > 0,
                "discrete W8A8 requires separate cube and vector cores");
    BuildMatmulTiling(data.gate, *platform, hiddenSize, intermediateSize, isNz, tileM, tileN);
    BuildMatmulTiling(data.down, *platform, intermediateSize, hiddenSize, isNz, tileM, tileN);
    auto host = at::zeros({TILING_STORAGE_BYTES},
                         at::TensorOptions().dtype(at::kByte).device(at::kCPU));
    std::memcpy(host.data_ptr(), &data, sizeof(data));
    // Preparation only: one transfer, not repeated in the compute hot path.
    return host.to(deviceTensor.device(), at::kByte, false, true);
}

using DiscreteMoeResult = std::tuple<at::Tensor, at::Tensor, at::Tensor,
                                     at::Tensor, at::Tensor, at::Tensor>;

// Internal implementation: only the trace entry exposes all six tensors.
// Output-only execution may overwrite dead gate/up storage with the down
// projection, so its other tuple members must never escape to the caller.
inline DiscreteMoeResult discrete_moe_w8a8_impl(
    const at::Tensor& x, const at::Tensor& xScale, const at::Tensor& descriptor,
    const at::Tensor& groupList, const at::Tensor& tiling,
    int64_t hiddenSize, int64_t intermediateSize, bool isNz, double swigluLimit,
    bool keepIntermediates)
{
    using namespace discrete_moe;
    TORCH_CHECK(x.device().type() == c10::DeviceType::PrivateUse1,
                "discrete W8A8 requires NPU input");
    c10_npu::NPUGuard deviceGuard(x.device());
    CheckShape(hiddenSize, intermediateSize);
    CheckDeviceTensor(x, x, at::kChar, "x");
    CheckDeviceTensor(xScale, x, at::kFloat, "x_scale");
    CheckDeviceTensor(descriptor, x, at::kLong, "descriptor");
    CheckDeviceTensor(groupList, x, at::kLong, "group_list");
    CheckDeviceTensor(tiling, x, at::kByte, "tiling");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(tiling.data_ptr()) % alignof(uint32_t) == 0,
                "tiling requires a 4-byte aligned base address");
    TORCH_CHECK(x.dim() == 2 && x.size(1) == hiddenSize, "x must have shape [M, H]");
    const int64_t rows = x.size(0);
    TORCH_CHECK(rows <= std::numeric_limits<uint32_t>::max(), "too many grouped token rows");
    TORCH_CHECK(xScale.dim() == 1 && xScale.numel() == rows,
                "x_scale must have shape [M]");
    TORCH_CHECK(groupList.dim() == 1, "group_list must have shape [E]");
    const int64_t groups = groupList.numel();
    TORCH_CHECK(groups <= std::numeric_limits<uint32_t>::max(), "too many expert groups");
    TORCH_CHECK(descriptor.dim() == 2 && descriptor.size(0) == groups &&
                    descriptor.size(1) == DESCRIPTOR_COLUMNS,
                "descriptor must have shape [E, 8]");
    TORCH_CHECK(tiling.dim() == 1 && tiling.numel() == TILING_STORAGE_BYTES,
                "tiling must be prepared by prepare_discrete_moe_w8a8_tiling");
    TORCH_CHECK(std::isfinite(swigluLimit) && swigluLimit >= 0.0 &&
                    swigluLimit <= std::numeric_limits<float>::max(),
                "swiglu_limit must be finite and nonnegative");
    TORCH_CHECK(groups != 0 || rows == 0, "nonempty x requires at least one expert group");
    at::Tensor integerScratch, gate, up, down;
    if (keepIntermediates) {
        gate = at::empty({rows, intermediateSize}, x.options().dtype(at::kInt));
        up = at::empty_like(gate);
    } else {
        // Allocate per invocation, on the guarded execution stream. All GMM
        // results are ND even when the expert weights use physical NZ.
        // Flat segments preserve each matrix's contiguous row stride; slicing
        // columns from [M, max(2I,H)] would produce incompatible strided views.
        const int64_t projectionElements = rows * intermediateSize;
        integerScratch = at::empty(
            {rows * std::max(2 * intermediateSize, hiddenSize)}, x.options().dtype(at::kInt));
        gate = integerScratch.narrow(0, 0, projectionElements).view({rows, intermediateSize});
        up = integerScratch.narrow(0, projectionElements, projectionElements).view({rows, intermediateSize});
    }
    auto activation = at::empty({rows, intermediateSize}, x.options());
    auto activationScale = at::empty({rows}, x.options().dtype(at::kFloat));
    if (keepIntermediates) {
        down = at::empty({rows, hiddenSize}, x.options().dtype(at::kInt));
    } else {
        // The same-stream activation kernel finishes reading gate/up before
        // GMM2 starts. The INT8 activation and its scale remain independent.
        down = integerScratch.narrow(0, 0, rows * hiddenSize).view({rows, hiddenSize});
    }
    // Invalid device group ranges are skipped, not interpreted as another
    // route. Their untouched output stays zero; validation is a caller duty.
    auto output = at::zeros({rows, hiddenSize}, x.options().dtype(at::kBFloat16));
    if (rows == 0) {
        return {output, gate, up, activation, activationScale, down};
    }
    auto* platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    TORCH_CHECK(platform != nullptr, "cannot discover Ascend execution platform");
    const uint32_t cubeCores = platform->GetCoreNumAic();
    const uint32_t vectorCores = platform->GetCoreNumAiv();
    const uint64_t expectedShape = static_cast<uint64_t>(hiddenSize) |
        (static_cast<uint64_t>(intermediateSize) << 16) |
        (static_cast<uint64_t>(isNz) << 32);
    // RunOpApi submits this handler in order with preceding work. Reading the
    // handle must not drain that same host task queue before every invocation.
    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    // Capture tensor handles, not only raw pointers: torch_npu may enqueue the
    // host custom handler and execute it after this C++ function returns.
    auto launchHandler = [x, xScale, descriptor, groupList, tiling, integerScratch, gate, up,
                              activation, activationScale, down, output, stream,
                              rows, groups, cubeCores, vectorCores, isNz,
                              swigluLimit, expectedShape]() -> int {
        const uint64_t status = discrete_moe_w8a8_launch(
            stream, x.data_ptr(), xScale.data_ptr(), descriptor.data_ptr(),
            groupList.data_ptr(), tiling.data_ptr(), gate.data_ptr(), up.data_ptr(),
            activation.data_ptr(), activationScale.data_ptr(), down.data_ptr(),
            output.data_ptr(), rows, groups, cubeCores, vectorCores, isNz,
            static_cast<float>(swigluLimit), expectedShape);
        TORCH_CHECK(status == 0, "discrete W8A8 launch failed at stage ",
                    status >> 32, " with CANN status ", static_cast<uint32_t>(status));
        return 0;
    };
    // The OpApi queue transfers the handler to its release queue after launch.
    // The compile-and-execute CustomHandler path can retain captured tensors
    // in execution-ring slots until those slots are reused. Keep all owners,
    // but submit through the framework path intended for direct runtime ops.
    at_npu::native::OpCommand::RunOpApi(
        keepIntermediates ? "discrete_moe_w8a8" : "discrete_moe_w8a8_output", std::move(launchHandler));
    return {output, gate, up, activation, activationScale, down};
}

// Preserve independent six-stage outputs for numerical verification.
inline DiscreteMoeResult discrete_moe_w8a8(
    const at::Tensor& x, const at::Tensor& xScale, const at::Tensor& descriptor,
    const at::Tensor& groupList, const at::Tensor& tiling,
    int64_t hiddenSize, int64_t intermediateSize, bool isNz, double swigluLimit)
{
    return discrete_moe_w8a8_impl(x, xScale, descriptor, groupList, tiling,
                                hiddenSize, intermediateSize, isNz, swigluLimit, true);
}

// Production entry: only the independently allocated BF16 output escapes.
inline at::Tensor discrete_moe_w8a8_output(
    const at::Tensor& x, const at::Tensor& xScale, const at::Tensor& descriptor,
    const at::Tensor& groupList, const at::Tensor& tiling,
    int64_t hiddenSize, int64_t intermediateSize, bool isNz, double swigluLimit)
{
    return std::get<0>(discrete_moe_w8a8_impl(x, xScale, descriptor, groupList, tiling,
                                           hiddenSize, intermediateSize, isNz, swigluLimit, false));
}

extern "C" uint32_t omoe_prepare_experts_launch(
    void* stream, void* sourceBases, void* descriptor, void* resident, void* ready,
    void* groups, void* observations, uint32_t matrixBytes, uint32_t experts,
    bool cumulative, bool observe);

inline void prepare_omoe_experts(
    const at::Tensor& sourceBases, const at::Tensor& descriptor,
    const at::Tensor& resident, const at::Tensor& ready,
    const at::Tensor& groups, const at::Tensor& observations,
    int64_t matrixBytes, bool cumulative, bool observe)
{
    using namespace discrete_moe;
    c10_npu::NPUGuard guard(groups.device());
    CheckDeviceTensor(sourceBases, groups, at::kLong, "source_bases");
    CheckDeviceTensor(descriptor, groups, at::kLong, "descriptor");
    CheckDeviceTensor(resident, groups, at::kInt, "resident");
    CheckDeviceTensor(ready, groups, at::kInt, "ready");
    CheckDeviceTensor(groups, groups, at::kLong, "groups");
    CheckDeviceTensor(observations, groups, at::kLong, "observations");
    const auto experts = groups.numel();
    TORCH_CHECK(sourceBases.numel() == 3 && descriptor.dim() == 2 &&
        descriptor.size(0) == experts && descriptor.size(1) == DESCRIPTOR_COLUMNS &&
        resident.numel() == experts && ready.numel() == experts && observations.numel() == experts,
        "O-MoE source and destination tables must cover every expert");
    TORCH_CHECK(matrixBytes > 0 && matrixBytes % 32 == 0 &&
        matrixBytes <= std::numeric_limits<uint32_t>::max(), "O-MoE matrix bytes must be 32-byte aligned");
    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    auto launchHandler = [sourceBases, descriptor, resident, ready, groups, observations,
                          stream, matrixBytes, experts, cumulative, observe]() -> int {
        const auto status = omoe_prepare_experts_launch(stream, sourceBases.data_ptr(), descriptor.data_ptr(),
            resident.data_ptr(), ready.data_ptr(), groups.data_ptr(), observations.data_ptr(),
            matrixBytes, experts, cumulative, observe);
        TORCH_CHECK(status == 0, "O-MoE device weight preparation failed: ", status);
        return 0;
    };
    at_npu::native::OpCommand::RunOpApi("prepare_omoe_experts", std::move(launchHandler));
}

inline std::vector<int64_t> register_omoe_host_sources(at::TensorList sources)
{
    std::vector<int64_t> addresses;
    try {
        for (const auto& source : sources) {
            TORCH_CHECK(source.device().is_cpu() && source.is_contiguous(),
                "O-MoE host sources must be contiguous CPU tensors");
            void* mapped = nullptr;
            const auto status = aclrtHostRegister(source.data_ptr(), source.nbytes(), ACL_HOST_REGISTER_MAPPED, &mapped);
            TORCH_CHECK(status == ACL_SUCCESS, "O-MoE host mapping failed: ", status);
            addresses.push_back(reinterpret_cast<int64_t>(mapped));
        }
    } catch (...) {
        for (size_t i = 0; i < addresses.size(); ++i) {
            aclrtHostUnregister(sources[i].data_ptr());
        }
        throw;
    }
    return addresses;
}

inline void unregister_omoe_host_sources(at::TensorList sources)
{
    // The caller retains these tensors and drains all consuming streams first.
    for (const auto& source : sources) {
        const auto status = aclrtHostUnregister(source.data_ptr());
        TORCH_CHECK(status == ACL_SUCCESS, "O-MoE host unmapping failed: ", status);
    }
}

}  // namespace vllm_ascend
#endif
