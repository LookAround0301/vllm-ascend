#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
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

from collections.abc import Callable
from dataclasses import dataclass

import torch
from vllm.distributed import get_tensor_model_parallel_rank
from vllm.model_executor.layers.fused_moe import FusedMoEMethodBase, FusedMoeWeightScaleSupported
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
from vllm.model_executor.layers.linear import LinearMethodBase, RowParallelLinear
from vllm.model_executor.layers.quantization.kv_cache import BaseKVCacheMethod
from vllm.model_executor.parameter import PerTensorScaleParameter
from vllm.model_executor.utils import set_weight_attrs

from vllm_ascend.distributed.parallel_state import get_mlp_tp_group, get_otp_group
from vllm_ascend.quantization.quant_type import QuantType
from vllm_ascend.utils import mlp_tp_enable, oproj_tp_enable

from .methods import AscendAttentionScheme, AscendLinearScheme, AscendMoEScheme, is_mx_quant_type


@dataclass(frozen=True)
class OMoECPUWeightSchema:
    """Normal checkpoint shapes; no device expert allocation or slot capacity."""

    local_experts: int
    hidden_size: int
    intermediate_size: int
    checkpoint_dtype: torch.dtype


def validate_omoe_cpu_weights(layer: torch.nn.Module) -> OMoECPUWeightSchema:
    """Validate checkpoint or post-transpose CPU sources before any device move."""
    schema = getattr(layer, "_omoe_cpu_weight_schema", None)
    if not isinstance(schema, OMoECPUWeightSchema):
        raise RuntimeError("O-MoE expert parameters require a CPU checkpoint schema")
    e, h, i = schema.local_experts, schema.hidden_size, schema.intermediate_size
    ready = getattr(layer, "_omoe_cpu_weights_ready", False)
    shapes = {
        "w13_weight": (e, h, 2 * i) if ready else (e, 2 * i, h),
        "w2_weight": (e, i, h) if ready else (e, h, i),
        "w13_weight_scale": (e, 2 * i) if ready else (e, 2 * i, 1),
        "w13_weight_offset": (e, 2 * i) if ready else (e, 2 * i, 1),
        "w2_weight_scale": (e, h) if ready else (e, h, 1),
        "w2_weight_offset": (e, h) if ready else (e, h, 1),
    }
    for name, shape in shapes.items():
        value = getattr(layer, name, None)
        dtype = torch.int8 if name in ("w13_weight", "w2_weight") else schema.checkpoint_dtype
        if ready and name.endswith("_scale"):
            dtype = torch.float32
        if not isinstance(value, torch.nn.Parameter):
            raise RuntimeError(f"O-MoE requires checkpoint parameter {name}")
        if value.device.type != "cpu" or value.dtype != dtype or tuple(value.shape) != shape:
            raise RuntimeError(f"O-MoE {name} must remain CPU {dtype} with shape {shape}")
        if not value.is_contiguous():
            raise RuntimeError(f"O-MoE {name} must be contiguous")
    return schema


def _omoe_cpu_expert_config(layer: torch.nn.Module):
    # Resolved per-engine configuration is available before RoutedExperts first
    # creates weights. Runner flags are set too late to prevent that allocation.
    from vllm_ascend.ascend_config import get_ascend_config

    config = get_ascend_config()
    if getattr(getattr(config, "omoe_config", None), "enabled", False) is not True:
        return None
    if "mtp" in getattr(layer, "layer_name", "").split("."):
        return None
    if config.expert_offload_config.h2d_backend != "torch":
        raise NotImplementedError("O-MoE CPU checkpoint sources currently require h2d_backend='torch'")
    return config


class AscendLinearMethod(LinearMethodBase):
    """Linear method for Ascend quantization.

    This wrapper class delegates to the actual quantization scheme implementation.
    The scheme is determined by the Config class and passed directly to this wrapper.

    Args:
        scheme: The quantization scheme instance (e.g., AscendW8A8DynamicLinearMethod).
    """

    def __init__(self, scheme: AscendLinearScheme) -> None:
        self.quant_method = scheme

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")

        weight_dict = self.quant_method.get_weight(input_size_per_partition, output_size_per_partition, params_dtype)

        # Extract packing information (if present)
        packed_dim = weight_dict.pop("_packed_dim", None)
        packed_factor = weight_dict.pop("_packed_factor", None)

        for weight_name, weight_param in weight_dict.items():
            param = torch.nn.Parameter(weight_param, requires_grad=False)
            set_weight_attrs(param, {"input_dim": 1, "output_dim": 0})

            # Set packing attributes if the weight is packed
            if packed_dim is not None and packed_factor is not None:
                set_weight_attrs(param, {"packed_dim": packed_dim, "packed_factor": packed_factor})

            layer.register_parameter(weight_name, param)
            set_weight_attrs(param, extra_weight_attrs)

        # NOTE: In flatquant quantization implementation,
        # the shape of pertensor_param requires introducing layer_type
        layer_type = "row" if isinstance(layer, RowParallelLinear) else "others"

        pertensor_dict = self.quant_method.get_pertensor_param(params_dtype, layer_type=layer_type)
        for pertensor_name, pertensor_param in pertensor_dict.items():
            param = PerTensorScaleParameter(data=pertensor_param, weight_loader=weight_loader)
            # disable warning
            param.ignore_warning = True
            layer.register_parameter(pertensor_name, param)
            param.weight_loader = extra_weight_attrs.get("weight_loader")

        perchannel_dict = self.quant_method.get_perchannel_param(output_size_per_partition, params_dtype)
        for perchannel_name, perchannel_param in perchannel_dict.items():
            param = torch.nn.Parameter(perchannel_param, requires_grad=False)
            set_weight_attrs(param, {"output_dim": 0})
            layer.register_parameter(perchannel_name, param)
            set_weight_attrs(param, extra_weight_attrs)

        # NOTE: In w4a8 quantization implementation,
        # for down_proj and o_proj scale_bias shape is [output_size, 16],
        # others are [output_size, 1]
        layer_type = "row" if isinstance(layer, RowParallelLinear) else "others"

        pergroup_dict = self.quant_method.get_pergroup_param(
            input_size_per_partition, output_size_per_partition, params_dtype, layer_type=layer_type
        )
        scale_packed_dim = pergroup_dict.pop("_packed_dim", None)
        scale_packed_factor = pergroup_dict.pop("_packed_factor", None)
        for pergroup_name, pergroup_param in pergroup_dict.items():
            param = torch.nn.Parameter(pergroup_param, requires_grad=False)
            set_weight_attrs(param, {"output_dim": 0})
            layer.register_parameter(pergroup_name, param)
            set_weight_attrs(param, extra_weight_attrs)
            if scale_packed_dim is not None and scale_packed_factor is not None:
                set_weight_attrs(param, {"packed_dim": scale_packed_dim, "packed_factor": scale_packed_factor})
            if (
                "weight_scale_second" in pergroup_name
                or "weight_offset_second" in pergroup_name
                or is_mx_quant_type(self.quant_method)
            ):
                param.input_dim = 1

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if hasattr(self.quant_method, "process_weights_after_loading"):
            self.quant_method.process_weights_after_loading(layer)

    def get_computed_params(self) -> set[str]:
        """Return parameter name patterns that are computed, not loaded.

        These parameters are computed during process_weights_after_loading
        rather than loaded from checkpoint:
        - weight_offset: Zero for symmetric quantization
        - quant_bias: Computed from weight statistics
        - deq_scale: Computed as input_scale * weight_scale
        - weight_scale: May be computed or have default values for some models
        """
        return {"weight_offset", "quant_bias", "deq_scale", "weight_scale"}

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if isinstance(layer, RowParallelLinear):
            if layer.prefix.find("o_proj") != -1 and oproj_tp_enable():
                tp_rank = get_otp_group().rank_in_group
            elif layer.prefix.find("down_proj") != -1 and mlp_tp_enable():
                tp_rank = get_mlp_tp_group().rank_in_group
            else:
                tp_rank = get_tensor_model_parallel_rank()
        else:
            tp_rank = 0
        return self.quant_method.apply(layer, x, bias, tp_rank)


class AscendKVCacheMethod(BaseKVCacheMethod):
    """KVCache method for Ascend quantization.

    This wrapper class delegates to the actual attention quantization scheme.

    Args:
        scheme: The attention quantization scheme instance.
    """

    def __init__(self, scheme: AscendAttentionScheme) -> None:
        self.quant_method = scheme

    def create_weights(self, layer: torch.nn.Module) -> None:
        # Different from linear method, there are no weight processing/slicing
        # steps for attention in vllm. So the whole process of create weights
        # is hidden into the specific quant method.
        self.quant_method.create_weights(layer)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        self.quant_method.process_weights_after_loading(layer)

    def apply(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache,
        attn_metadata,
        attn_type,
        scale,
        output,
    ) -> torch.Tensor:
        return self.quant_method.apply(layer, query, key, value, kv_cache, attn_metadata, attn_type, scale, output)


class AscendFusedMoEMethod(FusedMoEMethodBase):
    """FusedMoE method for Ascend quantization.

    This wrapper class delegates to the actual MoE quantization scheme.

    Args:
        scheme: The MoE quantization scheme instance.
        moe_config: The FusedMoE configuration.
    """

    def __init__(self, scheme: AscendMoEScheme, moe_config: FusedMoEConfig, tid2eid=None) -> None:
        super().__init__(moe_config)
        self.quant_method = scheme
        self.tid2eid = tid2eid

    @property
    def is_monolithic(self) -> bool:
        return False

    def maybe_make_prepare_finalize(self, routing_tables=None):
        # Ascend uses its own MoE communication and forward_impl path.
        # Do not let upstream modular-kernel initialization replace it.
        return None

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        config = _omoe_cpu_expert_config(layer)
        if config is not None:
            if getattr(self.quant_method, "quant_type", None) != QuantType.W8A8:
                raise NotImplementedError("O-MoE CPU expert loading currently supports only W8A8")
            if hasattr(layer, "_omoe_cpu_weight_schema"):
                raise RuntimeError("O-MoE checkpoint parameters must be created exactly once")
            if any(
                type(value) is not int or value <= 0
                for value in (num_experts, hidden_size, intermediate_size_per_partition)
            ):
                raise ValueError("O-MoE requires nonempty, positive checkpoint expert dimensions")
            layer.enable_omoe = True
            layer._omoe_cpu_weights_ready = False
            layer._omoe_cpu_weight_schema = OMoECPUWeightSchema(
                num_experts, hidden_size, intermediate_size_per_partition, params_dtype
            )
            # Keep full, nonzero standard-EP checkpoint shapes and the original
            # loader. It remains responsible for global/local IDs and TP slices.
            with torch.device("cpu"):
                self._create_weight_parameters(
                    layer, num_experts, hidden_size, intermediate_size_per_partition, params_dtype, **extra_weight_attrs
                )
            validate_omoe_cpu_weights(layer)
            return
        self._create_weight_parameters(
            layer, num_experts, hidden_size, intermediate_size_per_partition, params_dtype, **extra_weight_attrs
        )

    def _create_weight_parameters(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        weight_param = self.quant_method.get_weight(
            num_experts, intermediate_size_per_partition, hidden_size, params_dtype
        )
        for param_key, param_value in weight_param.items():
            param = torch.nn.Parameter(param_value, requires_grad=False)
            layer.register_parameter(param_key, param)
            set_weight_attrs(param, extra_weight_attrs)

        extra_weight_attrs.update({"quant_method": FusedMoeWeightScaleSupported.CHANNEL.value})
        per_group_param = ["weight_scale_second", "weight_offset_second", "scale_bias"] + (
            ["weight_scale", "weight_offset"]
            if hasattr(self.quant_method, "group_size") and self.quant_method.group_size > 0
            else []
        )
        dynamic_quant_param = self.quant_method.get_dynamic_quant_param(
            num_experts, intermediate_size_per_partition, hidden_size, params_dtype
        )
        for param_key, param_value in dynamic_quant_param.items():
            param = torch.nn.Parameter(param_value, requires_grad=False)
            layer.register_parameter(param_key, param)
            set_weight_attrs(param, extra_weight_attrs)
            if any(fields in param_key for fields in per_group_param):
                param.quant_method = FusedMoeWeightScaleSupported.GROUP.value

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool = False,
        num_experts: int = -1,
        expert_map: torch.Tensor | None = None,
        topk_group: int | None = None,
        num_expert_group: int | None = None,
        custom_routing_function: Callable | None = None,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
        e_score_correction_bias: torch.Tensor | None = None,
        is_prefill: bool = True,
        enable_force_load_balance: bool = False,
        log2phy: torch.Tensor | None = None,
        global_redundant_expert_num=0,
        pertoken_scale: torch.Tensor | None = None,
        activation: str = "silu",
        apply_router_weight_on_input: bool = False,
        mc2_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.quant_method.apply(
            layer=layer,
            x=x,
            router_logits=router_logits,
            top_k=top_k,
            renormalize=renormalize,
            use_grouped_topk=use_grouped_topk,
            num_experts=num_experts,
            expert_map=expert_map,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            custom_routing_function=custom_routing_function,
            scoring_func=scoring_func,
            routed_scaling_factor=routed_scaling_factor,
            e_score_correction_bias=e_score_correction_bias,
            is_prefill=is_prefill,
            enable_force_load_balance=enable_force_load_balance,
            log2phy=log2phy,
            global_redundant_expert_num=global_redundant_expert_num,
            pertoken_scale=pertoken_scale,
            activation=activation,
            apply_router_weight_on_input=apply_router_weight_on_input,
            mc2_mask=mc2_mask,
            tid2eid=self.tid2eid,
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if getattr(layer, "enable_omoe", False) is True:
            validate_omoe_cpu_weights(layer)
            # The ND expert kernel implements symmetric W8A8 only. Validate
            # all small quantization parameters before changing any storage;
            # unsupported checkpoints must not be silently misinterpreted.
            for name in ("w13_weight_offset", "w2_weight_offset"):
                if torch.any(getattr(layer, name) != 0):
                    raise NotImplementedError(f"O-MoE requires zero {name} for symmetric W8A8")
            for name in ("w13_weight_scale", "w2_weight_scale"):
                if not torch.isfinite(getattr(layer, name)).all():
                    raise NotImplementedError(f"O-MoE requires finite {name} for W8A8")
            if layer._omoe_cpu_weights_ready:
                return
            # The loader context patch keeps these parameters on CPU. Never
            # invoke the original scheme's whole-layer NPU/NZ conversion.
            layer.w13_weight.data = layer.w13_weight.data.transpose(1, 2).contiguous()
            layer.w2_weight.data = layer.w2_weight.data.transpose(1, 2).contiguous()
            for name in ("w13_weight_scale", "w2_weight_scale"):
                param = getattr(layer, name)
                param.data = param.data.reshape(param.shape[0], -1).to(torch.float32)
            for name in ("w13_weight_offset", "w2_weight_offset"):
                param = getattr(layer, name)
                param.data = param.data.reshape(param.shape[0], -1)
            layer.w13_weight_scale_fp32 = layer.w13_weight_scale.data
            layer._omoe_cpu_weights_ready = True
            validate_omoe_cpu_weights(layer)
            return
        if hasattr(self.quant_method, "process_weights_after_loading"):
            self.quant_method.process_weights_after_loading(layer)

    def get_fused_moe_quant_config(self, layer: torch.nn.Module):
        pass

    @property
    def supports_eplb(self):
        supports_eplb = getattr(self.quant_method, "supports_eplb", False)
        return supports_eplb


class AscendEmbeddingMethod(AscendLinearMethod):
    """Embedding method for Ascend quantization.

    This is essentially the same as AscendLinearMethod, just with a different name
    for clarity when used with VocabParallelEmbedding layers.

    Args:
        scheme: The quantization scheme instance.
    """

    def __init__(self, scheme: AscendLinearScheme) -> None:
        self.quant_method = scheme
