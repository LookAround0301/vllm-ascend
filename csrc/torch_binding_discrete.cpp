// SPDX-License-Identifier: Apache-2.0
#include "gmm/discrete_moe_w8a8/discrete_moe_w8a8_torch_adpt.h"
#include <torch/library.h>

namespace vllm_ascend {

at::Tensor prepare_discrete_moe_w8a8_tiling_meta(
    const at::Tensor& deviceTensor, int64_t hiddenSize, int64_t intermediateSize,
    bool isNz, int64_t tileM, int64_t tileN)
{
    return at::empty({discrete_moe::TILING_STORAGE_BYTES}, deviceTensor.options().dtype(at::kByte));
}

DiscreteMoeResult discrete_moe_w8a8_meta(
    const at::Tensor& x, const at::Tensor& xScale, const at::Tensor& descriptor,
    const at::Tensor& groupList, const at::Tensor& tiling,
    int64_t hiddenSize, int64_t intermediateSize, bool isNz, double swigluLimit)
{
    const auto rows = x.sym_size(0);
    auto output = at::empty_symint({rows, hiddenSize}, x.options().dtype(at::kBFloat16));
    auto gate = at::empty_symint({rows, intermediateSize}, x.options().dtype(at::kInt));
    auto up = at::empty_like(gate);
    auto activation = at::empty_symint({rows, intermediateSize}, x.options().dtype(at::kChar));
    auto scale = at::empty_symint({rows}, x.options().dtype(at::kFloat));
    auto down = at::empty_symint({rows, hiddenSize}, x.options().dtype(at::kInt));
    return {output, gate, up, activation, scale, down};
}

at::Tensor discrete_moe_w8a8_output_meta(
    const at::Tensor& x, const at::Tensor& xScale, const at::Tensor& descriptor,
    const at::Tensor& groupList, const at::Tensor& tiling,
    int64_t hiddenSize, int64_t intermediateSize, bool isNz, double swigluLimit)
{
    return at::empty_symint({x.sym_size(0), hiddenSize}, x.options().dtype(at::kBFloat16));
}

}  // namespace vllm_ascend

TORCH_LIBRARY_FRAGMENT(_C_ascend, m)
{
    m.def("prepare_omoe_experts(Tensor source_bases, Tensor descriptor, Tensor resident, "
          "Tensor(a!) ready, Tensor groups, Tensor(b!) observations, int matrix_bytes, "
          "bool cumulative, bool observe=True) -> ()");
    m.def("register_omoe_host_sources(Tensor[] sources) -> int[]");
    m.def("unregister_omoe_host_sources(Tensor[] sources) -> ()");
    m.def("prepare_discrete_moe_w8a8_tiling(Tensor device_tensor, int hidden_size, "
          "int intermediate_size, bool is_nz, int tile_m=128, int tile_n=256) -> Tensor");
    m.def("discrete_moe_w8a8(Tensor x, Tensor x_scale, Tensor descriptor, Tensor group_list, "
          "Tensor tiling, int hidden_size, int intermediate_size, bool is_nz, "
          "float swiglu_limit=10.) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
    m.def("discrete_moe_w8a8_output(Tensor x, Tensor x_scale, Tensor descriptor, Tensor group_list, "
          "Tensor tiling, int hidden_size, int intermediate_size, bool is_nz, "
          "float swiglu_limit=10.) -> Tensor");
}

TORCH_LIBRARY_IMPL(_C_ascend, CPU, m)
{
    m.impl("register_omoe_host_sources", &vllm_ascend::register_omoe_host_sources);
    m.impl("unregister_omoe_host_sources", &vllm_ascend::unregister_omoe_host_sources);
}

TORCH_LIBRARY_IMPL(_C_ascend, PrivateUse1, m)
{
    m.impl("prepare_omoe_experts", &vllm_ascend::prepare_omoe_experts);
    m.impl("prepare_discrete_moe_w8a8_tiling", &vllm_ascend::prepare_discrete_moe_w8a8_tiling);
    m.impl("discrete_moe_w8a8", &vllm_ascend::discrete_moe_w8a8);
    m.impl("discrete_moe_w8a8_output", &vllm_ascend::discrete_moe_w8a8_output);
}

TORCH_LIBRARY_IMPL(_C_ascend, Meta, m)
{
    m.impl("prepare_discrete_moe_w8a8_tiling", &vllm_ascend::prepare_discrete_moe_w8a8_tiling_meta);
    m.impl("discrete_moe_w8a8", &vllm_ascend::discrete_moe_w8a8_meta);
    m.impl("discrete_moe_w8a8_output", &vllm_ascend::discrete_moe_w8a8_output_meta);
}
