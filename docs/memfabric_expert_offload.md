# MemFabric expert H2D

The second-stage MemFabric integration replaces the expert-offload Host memory
allocator and H2D copy implementation while leaving routing, cache eviction,
expert placement, and NPU weight layout unchanged. Single-card uses LOCAL DRAM;
multi-card expert parallelism uses SHARED DRAM.

## Configuration

Install MemFabric Hybrid release 1.2 in the serving environment and source its
runtime environment before starting vLLM. Select the backend through
`additional_config.expert_offload_config`:

```json
{
  "expert_offload_config": {
    "expert_offload": true,
    "enable_multi_card": true,
    "shard_per_rank": true,
    "h2d_backend": "memfabric",
    "memfabric_pool_size_gib": 512,
    "memfabric_log_level": 3
  }
}
```

`memfabric_pool_size_gib` is the DRAM contribution of each serving process. In
single-card mode it is the LOCAL pool size. In multi-card mode every EP rank
contributes this amount to SHARED DRAM, so the aggregate physical pool is
`EP world size x memfabric_pool_size_gib`. Size each contribution to hold that
rank's expert shard and quantization metadata. The default backend is `torch`,
so existing configurations continue to use pinned CPU tensors and `copy_()`.

## Data path

- Expert weights and quantization metadata are allocated with
  `memfabric_hybrid.offload.empty`.
- Decode misses, expert prefetch, hot-expert preload, and single-card prefill
  are submitted through `memfabric_hybrid.offload.sparse_copy`.
- Copies are launched on the manager's current NPU load stream and synchronized
  at the existing stream boundaries. No device-wide synchronization is added.
- MemFabric is uninitialized when the transport is closed or the process exits.

In multi-card mode, each rank loads only its checkpoint shard. After weight
format conversion, the ranks exchange their SHARED GVA source pointers once
over the EP CPU group. Decode, prefetch, and hot preload can then fetch a peer
rank's expert directly with `sparse_copy`, allowing the placement planner to
balance experts across ranks without duplicating Host weights. Standard EP
prefill continues to load each rank's own static shard.

The integration pads odd sparse-copy descriptor batches with a zero-byte entry
because the release 1.2 kernel processes descriptors as two equal halves.

## Multi-card requirements

- `enable_multi_card` and `shard_per_rank` must both be `true`.
- All EP ranks must enter MemFabric initialization and shutdown collectively.
- `memfabric_pool_size_gib` is per rank, not the total across all ranks.
- Release 1.2 SHARED mode currently targets a single node and uses its internal
  configuration store for rank discovery.

With `moe_offload_debug=true`, successful startup prints
`backend=memfabric mode=shared` followed by the number of published SHARED
source pointers. Decode requests that fetch peer-owned experts print
`MEMFABRIC-SHARED-H2D` with separate `remote` and `local` load counts. NPU
profiling should contain `acc_sparse_copy`.
