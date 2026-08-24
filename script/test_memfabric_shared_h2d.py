#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Smoke-test multi-card MoE expert-weight H2D from MemFabric shared DRAM.

Every rank contributes a MemFabric-backed expert shard. The ranks exchange
their SHARED GVA pointers once, then every NPU pulls another rank's
discontiguous w13/w2 buffers with ``offload.sparse_copy``.

Run on one Ascend node after sourcing the CANN and MemFabric environments:

    torchrun --standalone --nproc-per-node=2 \
        script/test_memfabric_shared_h2d.py
"""

import argparse
import os
import sys

import torch
import torch.distributed as dist


ONE_GIB = 1 << 30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-experts", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=1024)
    parser.add_argument("--intermediate-size", type=int, default=256)
    parser.add_argument("--reserve-gib", type=int, default=1)
    parser.add_argument("--log-level", type=int, default=3)
    return parser.parse_args()


def _import_npu_dependencies():
    try:
        import torch_npu  # noqa: F401
        import memfabric_hybrid as mf
        from memfabric_hybrid import offload
    except ImportError as exc:
        raise RuntimeError(
            "torch_npu and memfabric_hybrid are required; source their "
            "environment setup scripts before running this smoke test"
        ) from exc
    return mf, offload


def _rank_env() -> tuple[int, int, int]:
    missing = [name for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE")
               if name not in os.environ]
    if missing:
        raise RuntimeError(
            "launch this script with torchrun; missing " + ", ".join(missing))
    return (int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"]),
            int(os.environ["WORLD_SIZE"]))


def _pattern(expert_id: int, weight_id: int) -> float:
    return float(expert_id * 2 + weight_id + 1)


def main() -> int:
    args = parse_args()
    if min(args.num_experts, args.hidden_size, args.intermediate_size,
           args.reserve_gib) <= 0:
        raise ValueError("shape, expert count, and reserve size must be positive")

    rank, local_rank, world_size = _rank_env()
    if world_size < 2:
        raise ValueError("shared DRAM test requires at least two ranks")
    if args.num_experts % world_size != 0:
        raise ValueError("num-experts must be divisible by world size")

    mf, offload = _import_npu_dependencies()
    mf.set_log_level(args.log_level)
    torch.npu.set_device(local_rank)

    config = offload.OffloadConfig()
    config.device_id = local_rank
    config.reserve_size = args.reserve_gib * ONE_GIB
    config.alloc_size = config.reserve_size
    config.world_size = world_size
    config.rank_id = rank
    config.scene = offload.Scene.SHARED
    ret = offload.initialize(config)
    if ret != 0:
        raise RuntimeError(f"rank {rank}: offload.initialize failed: ret={ret}")

    process_group_initialized = False
    cpu_group = None
    host_weights: list[torch.Tensor] = []
    device_slots: list[torch.Tensor] = []
    try:
        dist.init_process_group(backend="hccl")
        process_group_initialized = True
        cpu_group = dist.new_group(backend="gloo")
        shapes = (
            (args.hidden_size, 2 * args.intermediate_size),  # w13
            (args.intermediate_size, args.hidden_size),  # w2
        )
        experts_per_rank = args.num_experts // world_size
        shard_start = rank * experts_per_rank
        local_ptrs = []
        for expert_id in range(shard_start,
                               shard_start + experts_per_rank):
            for weight_id, shape in enumerate(shapes):
                value = _pattern(expert_id, weight_id)
                host = offload.empty(
                    shape, dtype=torch.bfloat16).fill_(value)
                host_weights.append(host)
                local_ptrs.append(host.data_ptr())

        gathered_ptrs = [None] * world_size
        dist.all_gather_object(
            gathered_ptrs, local_ptrs, group=cpu_group)
        source_rank = (rank + 1) % world_size
        source_start = source_rank * experts_per_rank
        expected: list[float] = []
        lengths_cpu: list[int] = []
        for expert_id in range(source_start,
                               source_start + experts_per_rank):
            for weight_id, shape in enumerate(shapes):
                slot = torch.empty(shape,
                                   dtype=torch.bfloat16,
                                   device=f"npu:{local_rank}")
                device_slots.append(slot)
                expected.append(_pattern(expert_id, weight_id))
                lengths_cpu.append(slot.numel() * slot.element_size())

        buffer_count = len(device_slots)
        src_ptrs = torch.tensor(gathered_ptrs[source_rank],
                                dtype=torch.int64,
                                device=f"npu:{local_rank}")

        dst_ptrs = torch.tensor([item.data_ptr() for item in device_slots],
                                 dtype=torch.int64,
                                 device=f"npu:{local_rank}")
        lengths = torch.tensor(lengths_cpu,
                               dtype=torch.int32,
                               device=f"npu:{local_rank}")
        count = torch.tensor(buffer_count,
                             dtype=torch.int32,
                             device=f"npu:{local_rank}")
        ret = offload.sparse_copy(src_ptrs, dst_ptrs, lengths, count,
                                  device_slots[0].device)
        if ret != 0:
            raise RuntimeError(
                f"rank {rank}: offload.sparse_copy failed: ret={ret}")
        torch.npu.synchronize()

        for index, (slot, value) in enumerate(zip(device_slots, expected)):
            reference = torch.full_like(slot, value)
            if not torch.equal(slot, reference):
                raise AssertionError(
                    f"rank {rank}: H2D verification failed for buffer {index}")
        dist.barrier()
    finally:
        if process_group_initialized:
            dist.barrier()
        offload.uninitialize()
        if process_group_initialized:
            dist.destroy_process_group()

    total_bytes = sum(lengths_cpu)
    print(
        "memfabric shared H2D OK: "
        f"rank={rank}/{world_size}, device={local_rank}, "
        f"source_rank={source_rank}, experts={experts_per_rank}, "
        f"buffers={len(device_slots)}, "
        f"bytes={total_bytes}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
