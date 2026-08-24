#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Time MoE expert-weight H2D with MemFabric local DRAM at real FP8 byte size.

Mirrors vLLM Ascend's single-card expert-offload data flow (MemFabric-backed
host weights/scales -> NPU expert slots) for DeepSeek-V4-Flash (FP8 / e4m3,
``w8a8_dynamic`` path). Sizes buffers to the REAL per-expert byte count read
from the checkpoint and reports H2D transfer time / bandwidth through
``offload.sparse_copy`` — the same primitive
``MemFabricLocalH2DTransport.copy_batch`` launches on the load stream.

Real DeepSeek-V4-Flash expert tensors (hidden=4096, moe_intermediate=2048,
weight_block_size=[128,128], one expert per layer, w1/w2/w3 stored separate):
    w1.weight (2048,2048)  I8        = 4.000 MiB   ┐
    w3.weight (2048,2048)  I8        = 4.000 MiB   ├ w13 = 8.000 MiB
    w2.weight (4096,1024)  I8        = 4.000 MiB   ┘
    w1.scale  (2048,128)   F8_E8M0   = 0.250 MiB   ┐
    w3.scale  (2048,128)   F8_E8M0   = 0.250 MiB   ├ scales = 0.750 MiB
    w2.scale  (4096,64)    F8_E8M0   = 0.250 MiB   ┘
    => one expert total = 12.750 MiB

``sparse_copy`` moves raw bytes; NZ/transpose are equal-length relayouts so
bytes are conserved. Defaults reproduce V4-Flash exactly; override on the CLI
for other models. Use --no-scales to time the three weights only (12.0 MiB).

Run on an Ascend host after sourcing the CANN and MemFabric environments:

    python3 script/test_memfabric_local_h2d.py --device-id 0

     # 默认就是 V4-Flash 单专家(12.75 MiB)
  python3 /home/g00619970/moeoffload_multi/vllm-ascend/script/test_memfabric_local_h2d.py --device-id 0

  # 只测三个权重(不含 scale):
  python3 /home/g00619970/moeoffload_multi/vllm-ascend/script/test_memfabric_local_h2d.py --device-id 0 --no-scales

  # 测多专家批量传输:
  python /home/g00619970/moeoffload_multi/vllm-ascend/script/test_memfabric_local_h2d.py --device-id 0 --num-experts 6
"""

import argparse
import statistics
import sys
import time

import torch


ONE_GIB = 1 << 30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--num-experts", type=int, default=1,
                        help="experts in one sparse_copy batch")
    parser.add_argument("--warmup", type=int, default=5,
                        help="untimed sparse_copy iterations before timing")
    parser.add_argument("--iters", type=int, default=20,
                        help="timed sparse_copy iterations")
    parser.add_argument("--reserve-gib", type=int, default=4)
    parser.add_argument("--log-level", type=int, default=3)
    parser.add_argument("--no-scales", action="store_true",
                        help="time weights only (omit the ~6% scales)")
    return parser.parse_args()


def _import_npu_dependencies():
    try:
        import torch_npu  # noqa: F401
        import memfabric_hybrid as mf
        from memfabric_hybrid import offload
    except ImportError as exc:
        raise RuntimeError(
            "torch_npu and memfabric_hybrid are required; source their "
            "environment setup scripts before running this timing test"
        ) from exc
    return mf, offload


def main() -> int:
    args = parse_args()
    if min(args.num_experts, args.warmup, args.iters, args.reserve_gib) <= 0:
        raise ValueError("num_experts, warmup, iters and reserve size "
                         "must be positive")

    mf, offload = _import_npu_dependencies()
    mf.set_log_level(args.log_level)
    torch.npu.set_device(args.device_id)

    config = offload.OffloadConfig()
    config.device_id = args.device_id
    config.reserve_size = args.reserve_gib * ONE_GIB
    config.alloc_size = config.reserve_size
    ret = offload.initialize(config)
    if ret != 0:
        raise RuntimeError(f"offload.initialize failed: ret={ret}")

    # DeepSeek-V4-Flash per-expert tensors (exact byte count). Weights are
    # FP8 (e4m3) stored as I8; scales are block-wise F8_E8M0 (one byte each).
    # sparse_copy moves raw bytes, so int8/uint8 tensors of these shapes
    # reproduce the real per-expert H2D byte count (NZ/transpose conserve bytes).
    weight_dtype = torch.int8
    scale_dtype = torch.uint8  # F8_E8M0 == one byte; uint8 is the storage type
    tensors = [
        ("w1", (2048, 2048), weight_dtype),
        ("w3", (2048, 2048), weight_dtype),
        ("w2", (4096, 1024), weight_dtype),
    ]
    if not args.no_scales:
        tensors += [
            ("w1_scale", (2048, 128), scale_dtype),
            ("w3_scale", (2048, 128), scale_dtype),
            ("w2_scale", (4096, 64), scale_dtype),
        ]

    host_weights: list[torch.Tensor] = []
    device_slots: list[torch.Tensor] = []
    labels: list[str] = []
    try:
        for expert_id in range(args.num_experts):
            for weight_id, (label, shape, dtype) in enumerate(tensors):
                # A deterministic fill so the one-shot correctness check has
                # something to compare against. Value does not affect timing
                # (sparse_copy only moves bytes).
                value = expert_id * 8 + weight_id + 1
                host = offload.empty(shape, dtype=dtype).fill_(value)
                slot = torch.empty(shape,
                                   dtype=dtype,
                                   device=f"npu:{args.device_id}")
                host_weights.append(host)
                device_slots.append(slot)
                labels.append(label)

        src_ptrs = torch.tensor([item.data_ptr() for item in host_weights],
                                 dtype=torch.int64,
                                 device=f"npu:{args.device_id}")
        dst_ptrs = torch.tensor([item.data_ptr() for item in device_slots],
                                 dtype=torch.int64,
                                 device=f"npu:{args.device_id}")
        lengths = torch.tensor(
            [item.numel() * item.element_size() for item in host_weights],
            dtype=torch.int32,
            device=f"npu:{args.device_id}",
        )
        count = torch.tensor(len(host_weights),
                             dtype=torch.int32,
                             device=f"npu:{args.device_id}")

        def _sparse_copy() -> int:
            return offload.sparse_copy(src_ptrs, dst_ptrs, lengths, count,
                                       device_slots[0].device)

        # One correctness self-check so the timing below is meaningful.
        ret = _sparse_copy()
        if ret != 0:
            raise RuntimeError(f"offload.sparse_copy failed: ret={ret}")
        torch.npu.synchronize()
        for index, (slot, item) in enumerate(zip(device_slots, host_weights)):
            reference = torch.full_like(slot, int(item.flatten()[0]))
            if not torch.equal(slot, reference):
                raise AssertionError(
                    f"H2D verification failed for buffer {index} "
                    f"({labels[index]})")

        # Warm up the sparse_copy path (first calls pay allocation / launch
        # overhead unrelated to steady-state transfer).
        for _ in range(args.warmup):
            ret = _sparse_copy()
            if ret != 0:
                raise RuntimeError(
                    f"offload.sparse_copy failed (warmup): ret={ret}")
            torch.npu.synchronize()

        # Timed loop: each iteration is one sparse_copy + a stream sync, so the
        # measurement covers the full host->device transfer latency for this
        # batch, not just the async enqueue.
        samples_ms: list[float] = []
        for _ in range(args.iters):
            t0 = time.perf_counter()
            ret = _sparse_copy()
            if ret != 0:
                raise RuntimeError(
                    f"offload.sparse_copy failed (timed): ret={ret}")
            torch.npu.synchronize()
            t1 = time.perf_counter()
            samples_ms.append((t1 - t0) * 1e3)

        total_bytes = sum(item.numel() * item.element_size()
                          for item in host_weights)
        avg_ms = statistics.mean(samples_ms)
        min_ms = min(samples_ms)
        max_ms = max(samples_ms)
        med_ms = statistics.median(samples_ms)
        # Bandwidth from min (least noise) and median samples.
        bw_min_gbs = total_bytes / (min_ms * 1e-3) / ONE_GIB
        bw_med_gbs = total_bytes / (med_ms * 1e-3) / ONE_GIB

        print(
            "memfabric local H2D timing: "
            f"device={args.device_id}, experts={args.num_experts}, "
            f"model=DeepSeek-V4-Flash (FP8), "
            f"buffers={len(host_weights)}",
            flush=True)
        per_label_bytes: dict[str, int] = {}
        for item, lbl in zip(host_weights, labels):
            per_label_bytes[lbl] = (per_label_bytes.get(lbl, 0)
                                    + item.numel() * item.element_size())
        for lbl in ("w1", "w3", "w2", "w1_scale", "w3_scale", "w2_scale"):
            if lbl in per_label_bytes:
                b = per_label_bytes[lbl]
                print(f"  {lbl:10s}: {b:>10d} bytes "
                      f"({b / (1 << 20):7.3f} MiB)", flush=True)
        w13 = per_label_bytes.get("w1", 0) + per_label_bytes.get("w3", 0)
        print(
            f"  total/expert: {total_bytes // max(args.num_experts, 1)} bytes "
            f"({total_bytes / max(args.num_experts, 1) / (1 << 20):.3f} MiB)  "
            f"[w13(w1+w3)={w13 / (1 << 20):.3f} MiB]",
            flush=True)
        print(
            f"  per sparse_copy (iters={args.iters}, warmup={args.warmup}): "
            f"min={min_ms:.3f} ms  median={med_ms:.3f} ms  "
            f"avg={avg_ms:.3f} ms  max={max_ms:.3f} ms",
            flush=True)
        print(
            f"  bandwidth: {bw_min_gbs:.2f} GiB/s (from min)  "
            f"{bw_med_gbs:.2f} GiB/s (from median)",
            flush=True)
        if args.num_experts == 1:
            print(
                f"  => one V4-Flash expert transfer time "
                f"= {med_ms:.3f} ms (median)",
                flush=True)
    finally:
        offload.uninitialize()

    return 0


if __name__ == "__main__":
    sys.exit(main())
