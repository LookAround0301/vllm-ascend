# SPDX-License-Identifier: Apache-2.0
"""Prepared, descriptor-addressed W8A8 expert computation without pool ownership.

Prepare on the stream where the supplied weight views are ready. Keep their
contents, shapes, and storage unchanged while this object is in use. The cached
descriptor owns no memory: the prepared object retains every referenced tensor.
Routing values must come from a valid dispatch (nonnegative group lengths whose
sum is M). This module checks metadata but never downloads device routing values.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass, field
from math import isfinite, prod
from pathlib import Path
from typing import Any, Literal

import torch

DESCRIPTOR_COLUMNS = 8
TILING_STORAGE_BYTES = 1024
MAX_HIDDEN_SIZE = 8192
MAX_INTERMEDIATE_SIZE = 4096
DEFAULT_SMALL_TILING = (16, 128)
DEFAULT_LARGE_TILING = (128, 256)
SMALL_TILING_MAX_ROWS = 16


@dataclass(frozen=True, slots=True)
class DiscreteExpertWeights:
    """One logical expert; its position in the collection is the compute group.

    ``up=None`` means ``gate`` contains fused gate||up columns [H, 2I], with
    fused channel scales [2I]. Otherwise gate/up are separate [H, I] matrices.
    Down always has logical shape [I, H]. All scales use FP32; conversion from
    checkpoint scales is an explicit preparation operation, never a forward
    side effect. The declared logical shapes also apply to NZ-format tensors.
    """

    expert_id: int
    gate: torch.Tensor
    down: torch.Tensor
    gate_scale: torch.Tensor
    down_scale: torch.Tensor
    up: torch.Tensor | None = None
    up_scale: torch.Tensor | None = None

    @property
    def is_fused(self) -> bool:
        return self.up is None


@dataclass(frozen=True, slots=True)
class DiscreteMoEWeights:
    """Expert tensor references in dispatch group order, without pool ownership.

    Logical expert IDs are metadata, not tensor-list or scale-row indices.
    Empty and mixed fused/split collections are supported. Tensor extents and
    layouts are checked once when preparing the raw-address descriptor.
    """

    experts: tuple[DiscreteExpertWeights, ...]
    hidden_size: int
    intermediate_size: int
    layout: Literal["nd", "nz"] = "nd"  # codespell:ignore nd

    @property
    def expert_ids(self) -> tuple[int, ...]:
        return tuple(expert.expert_id for expert in self.experts)


@dataclass(frozen=True, slots=True)
class PreparedDiscreteMoEWeights:
    """Ready weight references, address table, tiling and preparation fence.

    Allocator ``record_stream`` protects referenced PyTorch allocations, not
    an external pool's right to overwrite memory. Such reuse remains the
    caller's responsibility. Ordinary compute on ``creation_stream`` does not
    iterate over experts; a different stream waits for preparation and records
    all referenced allocations on that consuming stream.

    Page-backed callers may retain complete arenas in tensor_refs instead of
    individual expert views; weights then contains only the matrix geometry
    and descriptor rows determine the number of compute groups.

    Graph replay, when supported by the surrounding execution path, must keep
    this object and its weights alive for the graph's entire lifetime: future
    replays do not execute the Python stream-recording code again.
    """

    weights: DiscreteMoEWeights
    descriptor: torch.Tensor
    tiling: torch.Tensor
    device: torch.device
    creation_stream: Any = field(repr=False)
    ready_event: Any = field(repr=False)
    tensor_refs: tuple[torch.Tensor, ...] = field(repr=False)
    operator: Callable = field(repr=False)
    output_operator: Callable = field(repr=False)
    creation_stream_key: tuple[int, int, int] | None = field(default=None, repr=False)
    stream_metadata_getter: Callable | None = field(default=None, repr=False)
    base_tiling_config: tuple[int, int] = field(default=DEFAULT_SMALL_TILING, repr=False)
    large_tiling: torch.Tensor | None = field(default=None, repr=False)

    def select_tiling(self, rows: int) -> torch.Tensor:
        """Select already-prepared metadata by total dispatched row count."""
        if self.large_tiling is not None and rows > SMALL_TILING_MAX_ROWS:
            return self.large_tiling
        return self.tiling


def _require_npu_device(device: torch.device) -> None:
    if device.type != "npu" or device.index is None:
        raise ValueError("discrete W8A8 requires an indexed NPU device")


def _resolve_device(weights: DiscreteMoEWeights, device) -> torch.device:
    inferred = weights.experts[0].gate.device if weights.experts else None
    if device is None:
        if inferred is None:
            raise ValueError("empty expert collections require an explicit device")
        result = inferred
    else:
        result = torch.device(device)
        if result.type == "npu" and result.index is None:
            result = torch.device("npu", torch.npu.current_device())
        if inferred is not None and result != inferred:
            raise ValueError("device must match the expert weight device")
    _require_npu_device(result)
    return result


def _load_discrete_library(library_path):
    # Cold-path import: do not load hardware registration in CPU metadata tools.
    from vllm_ascend.utils import enable_custom_op

    if not enable_custom_op():
        raise RuntimeError("discrete W8A8 requires vLLM Ascend custom operators")
    namespace = torch.ops._C_ascend
    compute = getattr(namespace, "discrete_moe_w8a8", None)
    prepare = getattr(namespace, "prepare_discrete_moe_w8a8_tiling", None)
    output = getattr(namespace, "discrete_moe_w8a8_output", None)
    requested = Path(library_path).expanduser().resolve() if library_path is not None else None
    if compute is not None or prepare is not None or output is not None:
        if compute is None or prepare is None or output is None:
            raise RuntimeError("incomplete discrete W8A8 operator registration; restart with a consistent build")
        if requested is not None and str(requested) not in torch.ops.loaded_libraries:
            raise RuntimeError("discrete W8A8 is already registered; cannot load a second operator library")
        return prepare, compute, output
    path = requested or Path(__file__).resolve().parents[2] / "libvllm_ascend_discrete_moe_C.so"
    if not path.is_file():
        raise FileNotFoundError(f"discrete W8A8 operator library is missing: {path}; build the discrete target first")
    torch.ops.load_library(str(path))
    prepare = getattr(namespace, "prepare_discrete_moe_w8a8_tiling", None)
    compute = getattr(namespace, "discrete_moe_w8a8", None)
    output = getattr(namespace, "discrete_moe_w8a8_output", None)
    if prepare is None or compute is None or output is None:
        raise RuntimeError(f"library did not register the required discrete W8A8 operators: {path}")
    return prepare, compute, output


def _descriptor_rows(weights: DiscreteMoEWeights) -> list[list[int]]:
    """Read host tensor metadata only; never gather or repack a weight matrix."""
    h, i = weights.hidden_size, weights.intermediate_size
    rows = []
    device = weights.experts[0].gate.device if weights.experts else None
    for group, expert in enumerate(weights.experts):
        # Raw pointers carry no tensor metadata. Check their complete extents
        # here before encoding addresses, rather than during object creation.
        n = 2 * i if expert.is_fused else i
        matrices = [("gate", expert.gate, (h, n)), ("down", expert.down, (i, h))]
        scales = [("gate_scale", expert.gate_scale, (n,)), ("down_scale", expert.down_scale, (h,))]
        if not expert.is_fused:
            matrices.append(("up", expert.up, (h, i)))
            scales.append(("up_scale", expert.up_scale, (i,)))
        for name, tensor, shape in matrices:
            label = f"group {group} {name}"
            _validate_tensor(tensor, shape, torch.int8, device, label, check_format=False)
            _check_device_format(tensor, weights.layout, label)
        for name, tensor, shape in scales:
            _validate_tensor(tensor, shape, torch.float32, device, f"group {group} {name}")
        if expert.is_fused:
            up_offset = h * i if weights.layout == "nz" else i
            up_address = expert.gate.data_ptr() + up_offset
            up_scale_address = expert.gate_scale.data_ptr() + i * expert.gate_scale.element_size()
            row_stride = 2 * i
        else:
            up_address = expert.up.data_ptr()
            up_scale_address = expert.up_scale.data_ptr()
            row_stride = i
        rows.append(
            [
                expert.gate.data_ptr(),
                up_address,
                expert.down.data_ptr(),
                expert.gate_scale.data_ptr(),
                up_scale_address,
                expert.down_scale.data_ptr(),
                row_stride,
                row_stride,
            ]
        )
    return rows


def _weight_refs(weights: DiscreteMoEWeights) -> tuple[torch.Tensor, ...]:
    return tuple(
        tensor
        for expert in weights.experts
        for tensor in (expert.gate, expert.up, expert.down, expert.gate_scale, expert.up_scale, expert.down_scale)
        if tensor is not None
    )


def _upload_descriptor(rows: list[list[int]], device: torch.device) -> torch.Tensor:
    return torch.tensor(rows, dtype=torch.int64, device="cpu").reshape(-1, DESCRIPTOR_COLUMNS).to(device)


def prepare_discrete_moe_weights(
    weights: DiscreteMoEWeights, library_path=None, tile_m: int | None = None, tile_n: int | None = None,
    *, device=None, tiling_source: PreparedDiscreteMoEWeights | None = None,
) -> PreparedDiscreteMoEWeights:
    """Validate and prepare once; all scale conversion must already be complete.

    ``device`` is only required for an empty expert collection. The explicit
    optional library path is for isolated builds and never changes in forward.
    With neither tile argument specified, prepare 16/128 and 128/256 metadata
    once; total dispatched rows <=16 selects the smaller tile. An explicit
    tile argument disables automatic selection and the other argument, when
    omitted, retains its original 16/128 default.
    Tiling depends on matrix geometry, not expert addresses or group count;
    a compatible tiling_source reuses it across new descriptors and layers.
    There is no TensorList fallback and no per-token weight format conversion.
    """
    if weights.layout not in ("nd", "nz"):  # codespell:ignore nd
        raise ValueError("layout must be 'nd' or 'nz'")  # codespell:ignore nd
    for name, value, upper in (
        ("hidden_size", weights.hidden_size, MAX_HIDDEN_SIZE),
        ("intermediate_size", weights.intermediate_size, MAX_INTERMEDIATE_SIZE),
    ):
        if not 0 < value <= upper or value % 32:
            raise ValueError(f"{name} must be positive, divisible by 32 and at most {upper}")
    automatic_tiling = tile_m is None and tile_n is None
    tile_m = DEFAULT_SMALL_TILING[0] if tile_m is None else tile_m
    tile_n = DEFAULT_SMALL_TILING[1] if tile_n is None else tile_n
    base_tiling_config = (tile_m, tile_n)
    for name, value, multiple, upper in (("tile_m", tile_m, 16, 128), ("tile_n", tile_n, 32, 256)):
        if type(value) is not int or not 0 < value <= upper or value % multiple:
            raise ValueError(f"{name} must be a multiple of {multiple} in [{multiple}, {upper}]")
    resolved_device = _resolve_device(weights, device)
    with torch.npu.device(resolved_device.index):
        # An empty collection has not allocated a device tensor yet. Direct
        # AscendC shared-library constructors require an initialized runtime.
        torch.npu.init()
        prepare_op, compute_op, output_op = _load_discrete_library(library_path)
        creation_stream = torch.npu.current_stream(resolved_device.index)
        import torch_npu

        # This is the metadata getter used by public current_stream in the
        # supported torch_npu release. Retain only its capability, never a
        # cached answer: each compute must still query the calling thread's
        # actual current stream for this device.
        stream_metadata_getter = getattr(torch_npu._C, "_npu_getCurrentStream", None)
        creation_stream_key = None
        if callable(stream_metadata_getter):
            creation_stream_key = (
                creation_stream.stream_id,
                creation_stream.device_index,
                creation_stream.device_type,
            )
        else:
            stream_metadata_getter = None
        descriptor = _upload_descriptor(_descriptor_rows(weights), resolved_device)
        _validate_tensor(
            descriptor, (len(weights.experts), DESCRIPTOR_COLUMNS), torch.int64, resolved_device, "descriptor"
        )
        anchor = weights.experts[0].gate if weights.experts else descriptor

        def prepare_tiling(config):
            result = prepare_op(anchor, weights.hidden_size, weights.intermediate_size, weights.layout == "nz", *config)
            _validate_tensor(result, (TILING_STORAGE_BYTES,), torch.uint8, resolved_device, "tiling")
            return result

        if (
            tiling_source is not None
            and tiling_source.device == resolved_device
            and tiling_source.weights.hidden_size == weights.hidden_size
            and tiling_source.weights.intermediate_size == weights.intermediate_size
            and tiling_source.weights.layout == weights.layout
            and tiling_source.base_tiling_config == base_tiling_config
            and (tiling_source.large_tiling is not None) == automatic_tiling
        ):
            if creation_stream != tiling_source.creation_stream:
                creation_stream.wait_event(tiling_source.ready_event)
            tiling, large_tiling = tiling_source.tiling, tiling_source.large_tiling
        else:
            tiling = prepare_tiling(base_tiling_config)
            large_tiling = prepare_tiling(DEFAULT_LARGE_TILING) if automatic_tiling else None
        refs = _weight_refs(weights) + (descriptor, tiling)
        if large_tiling is not None:
            refs += (large_tiling,)
        for tensor in refs:
            tensor.record_stream(creation_stream)
        ready_event = torch.npu.Event()
        ready_event.record(creation_stream)
    return PreparedDiscreteMoEWeights(
        weights,
        descriptor,
        tiling,
        resolved_device,
        creation_stream,
        ready_event,
        refs,
        compute_op,
        output_op,
        creation_stream_key,
        stream_metadata_getter,
        base_tiling_config,
        large_tiling,
    )


def _check_device_format(tensor: torch.Tensor, layout: str, name: str) -> None:
    if tensor.device.type == "cpu":
        # ND layout keeps its lowercase external spelling; exemptions below are word-local.
        if layout != "nd":  # codespell:ignore nd
            raise ValueError("NZ weights require real NPU tensors, not reshaped CPU tensors")
        return
    if tensor.device.type != "npu":
        raise ValueError(f"{name}: only CPU reference or NPU tensors are supported")

    # Lazy import keeps CPU-only metadata tools independent of the NPU runtime.
    import torch_npu

    actual = torch_npu.get_npu_format(tensor)
    expected = (torch_npu.Format.FRACTAL_NZ,) if layout == "nz" else (0, 2)
    if actual not in expected:
        raise ValueError(f"{name}: declared {layout} layout does not match NPU format {actual}")
    if layout == "nz":
        k, n = tensor.shape
        # Direct pointers may select a whole expert from a batched NZ storage,
        # but a same-numel logical reshape must not reinterpret that storage.
        try:
            physical = tuple(torch.ops._C_ascend.get_npu_storage_shape(tensor))
        except AttributeError as exc:
            raise RuntimeError("NZ layout validation requires initialized vLLM Ascend custom ops") from exc
        if physical[-4:] != (n // 32, k // 16, 16, 32):
            raise ValueError(f"{name}: physical NZ shape {physical} does not match logical shape {(k, n)}")
        offset, matrix_size = tensor.storage_offset(), k * n
        if offset % matrix_size or offset + matrix_size > prod(physical):
            raise ValueError(f"{name}: NZ view must select a complete aligned matrix within storage")


def _validate_tensor(tensor, shape, dtype, device, name, *, check_format=True):
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    if tuple(tensor.shape) != shape or tensor.device != device:
        raise ValueError(f"{name} must have shape {shape} on {device}")
    if tensor.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    if check_format:
        # ND layout keeps its lowercase external spelling; exemptions below are word-local.
        _check_device_format(tensor, "nd", name)  # codespell:ignore nd


def apply_discrete_moe_mlp(
    prepared: PreparedDiscreteMoEWeights,
    hidden_states: torch.Tensor,
    group_list: torch.Tensor,
    *,
    group_list_type: int = 0,
    dynamic_scale: torch.Tensor | None = None,
    swiglu_limit: float = 10.0,
    return_intermediates: bool = False,
):
    """Compute W8A8 MLP in dispatch order; routing weights belong to combine.

    Type 0 is cumulative group ends; type 1 is group counts. Values must be
    nonnegative and describe exactly the M supplied rows. Valid dispatch is a
    precondition: no ``item``/``tolist`` or CPU synchronization is inserted to
    inspect routing. The low-level kernel additionally guards invalid ranges.
    An INT8 input requires FP32 per-row scales; BF16 is dynamically quantized.
    Dispatched inputs must already be ready on the current consumer stream;
    only weight preparation has an automatically managed ready fence.
    ``return_intermediates`` returns the low-level six-tensor numerical trace.
    Otherwise only BF16 output is returned; the output-only operator reuses
    dead INT32 projection storage within this invocation, never across calls.
    """
    if not isinstance(prepared, PreparedDiscreteMoEWeights):
        raise TypeError("prepared must be PreparedDiscreteMoEWeights")
    if type(group_list_type) is not int or group_list_type not in (0, 1):
        raise ValueError("group_list_type must be 0 (cumulative) or 1 (counts)")
    if type(swiglu_limit) not in (int, float) or not isfinite(swiglu_limit) or swiglu_limit < 0:
        raise ValueError("swiglu_limit must be finite and nonnegative")
    if not isinstance(hidden_states, torch.Tensor) or hidden_states.dim() != 2:
        raise ValueError("hidden_states must have shape [M, H]")
    if hidden_states.dtype not in (torch.int8, torch.bfloat16):
        raise TypeError("discrete W8A8 accepts only INT8 or BF16 hidden_states")
    rows = hidden_states.size(0)
    h, i = prepared.weights.hidden_size, prepared.weights.intermediate_size
    # Page-backed callers retain whole arenas and encode addresses directly;
    # the descriptor is the authoritative compute group count.
    groups = prepared.descriptor.shape[0]
    # The C++ entry checks ND format for every tensor it consumes. Only BF16
    # needs the Python format check before the separate quantization op.
    # Cached descriptor/tiling metadata was also validated during preparation;
    # checking it twice per forward adds overhead without extending safety.
    _validate_tensor(
        hidden_states,
        (rows, h),
        hidden_states.dtype,
        prepared.device,
        "hidden_states",
        check_format=hidden_states.dtype == torch.bfloat16,
    )
    _validate_tensor(
        group_list, (groups,), torch.int64, prepared.device, "group_list", check_format=group_list_type == 1
    )
    if rows and not groups:
        raise ValueError("nonempty hidden_states require at least one expert group")
    if hidden_states.dtype == torch.int8:
        _validate_tensor(dynamic_scale, (rows,), torch.float32, prepared.device, "dynamic_scale", check_format=False)
    elif dynamic_scale is not None:
        raise ValueError("BF16 hidden_states must not supply dynamic_scale")
    # The direct INT8/cumulative path is device-guarded by its C++ entry.
    # Keep a Python guard only for quantization/cumsum performed before it.
    quantized_internally = hidden_states.dtype == torch.bfloat16
    device_index = prepared.device.index
    context = torch.npu.device(device_index) if quantized_internally or group_list_type else nullcontext()
    with context:
        get_stream_metadata = prepared.stream_metadata_getter
        if get_stream_metadata is not None and get_stream_metadata(device_index) == prepared.creation_stream_key:
            stream = prepared.creation_stream
        else:
            # Different streams and older torch_npu versions keep the public
            # path. Getter runtime errors must propagate, not hide behind it.
            stream = torch.npu.current_stream(device_index)
        if stream != prepared.creation_stream:
            stream.wait_event(prepared.ready_event)
            for tensor in prepared.tensor_refs:
                tensor.record_stream(stream)
        if quantized_internally:
            # Keep the original BF16 allocation alive on this stream even
            # after the Python variable is replaced by quantized outputs.
            hidden_states.record_stream(stream)
            if rows:
                import torch_npu

                hidden_states, dynamic_scale = torch_npu.npu_dynamic_quant(hidden_states, dst_type=torch.int8)
            else:
                hidden_states = torch.empty((0, h), dtype=torch.int8, device=prepared.device)
                dynamic_scale = torch.empty((0,), dtype=torch.float32, device=prepared.device)
        if group_list_type == 1:
            # The original dispatch counts are an asynchronous cumsum input.
            group_list.record_stream(stream)
            group_list = group_list.cumsum(dim=0)
        # External inputs may have been allocated on another stream. Fresh
        # quantization/cumsum outputs were allocated on this guarded consumer
        # stream and are captured by C++ until all stages are submitted.
        # Recording them again would add needless allocator end-of-life events.
        if not quantized_internally:
            hidden_states.record_stream(stream)
            dynamic_scale.record_stream(stream)
        if group_list_type == 0:
            group_list.record_stream(stream)
        operator = prepared.operator if return_intermediates else prepared.output_operator
        result = operator(
            hidden_states,
            dynamic_scale,
            prepared.descriptor,
            group_list,
            prepared.select_tiling(rows),
            h,
            i,
            prepared.weights.layout == "nz",
            float(swiglu_limit),
        )
    return result


__all__ = [
    "DiscreteExpertWeights",
    "DiscreteMoEWeights",
    "PreparedDiscreteMoEWeights",
    "prepare_discrete_moe_weights",
    "apply_discrete_moe_mlp",
]
