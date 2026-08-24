"""Host-to-device transfer abstractions for expert offload."""

import atexit
import threading
from dataclasses import dataclass
from typing import Any, Protocol

import torch


ONE_GIB = 1 << 30
UINT32_MAX = (1 << 32) - 1


@dataclass(frozen=True)
class H2DCopyTask:
    """One byte-preserving host-to-device copy.

    ``source`` and ``destination`` intentionally accept both tensors and
    untyped-storage slices. Expert weights use storage slices to preserve their
    packed/NZ byte layout, while quantization attributes use shaped tensors.
    ``nbytes`` is carried explicitly so pointer-based backends can consume the
    same task without inferring sizes from tensor metadata.
    """

    source: Any
    destination: Any
    nbytes: int
    name: str = ""
    non_blocking: bool = True

    def __post_init__(self) -> None:
        if self.nbytes <= 0:
            raise ValueError(f"H2D copy size must be positive; got {self.nbytes}")


@dataclass(frozen=True)
class HostPointerSource:
    """Pointer-only view of a peer rank's MemFabric SHARED allocation."""

    pointer: int

    def data_ptr(self) -> int:
        return self.pointer


class ExpertH2DTransport(Protocol):
    """Backend contract for a batch of expert H2D copies."""

    def allocate_host_tensor(self, shape, dtype) -> torch.Tensor:
        """Allocate a CPU tensor suitable as this backend's H2D source."""
        ...

    def copy_batch(self, tasks: list[H2DCopyTask]) -> None:
        """Enqueue all tasks on the caller-selected device stream."""
        ...

    def synchronize(self, stream) -> None:
        """Wait for submitted copies and release descriptor resources."""
        ...

    def close(self) -> None:
        """Release backend resources."""
        ...


class TorchCopyH2DTransport:
    """Existing expert-offload H2D behavior implemented with ``copy_``."""

    supports_remote_sources = False

    def allocate_host_tensor(self, shape, dtype) -> torch.Tensor:
        return torch.empty(shape, dtype=dtype, device="cpu", pin_memory=True)

    def copy_batch(self, tasks: list[H2DCopyTask]) -> None:
        for task in tasks:
            task.destination.copy_(task.source,
                                   non_blocking=task.non_blocking)

    @staticmethod
    def synchronize(stream) -> None:
        stream.synchronize()

    @staticmethod
    def close() -> None:
        return


class MemFabricLocalH2DTransport:
    """MemFabric LOCAL DRAM allocator and sparse-copy H2D backend.

    MemFabric launches ``sparse_copy`` on torch-npu's current stream. The
    manager selects its load stream before calling ``copy_batch`` and later
    calls ``synchronize`` on that same stream, preserving graph host-callback
    ordering without a device-wide synchronization.
    """

    supports_remote_sources = False

    def __init__(
        self,
        device_id: int,
        pool_size_gib: int,
        log_level: int = 3,
        *,
        mf_module=None,
        offload_module=None,
    ) -> None:
        if pool_size_gib <= 0:
            raise ValueError(
                "MemFabric pool size must be positive; "
                f"got {pool_size_gib} GiB")
        if mf_module is None or offload_module is None:
            try:
                import memfabric_hybrid as mf
                from memfabric_hybrid import offload
            except ImportError as exc:
                raise RuntimeError(
                    "h2d_backend='memfabric' requires memfabric_hybrid; "
                    "install it and source the MemFabric environment first"
                ) from exc
            mf_module = mf
            offload_module = offload

        self.device_id = device_id
        self.device = torch.device(f"npu:{device_id}")
        self._offload = offload_module
        self._closed = False
        self._inflight_descriptors = []
        self._descriptor_lock = threading.Lock()

        mf_module.set_log_level(log_level)
        config = offload_module.OffloadConfig()
        config.device_id = device_id
        config.reserve_size = pool_size_gib * ONE_GIB
        config.alloc_size = config.reserve_size
        ret = offload_module.initialize(config)
        if ret != 0:
            raise RuntimeError(
                "MemFabric LOCAL initialization failed: "
                f"device={device_id}, pool_size_gib={pool_size_gib}, ret={ret}")
        atexit.register(self.close)

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("MemFabric H2D transport is already closed")

    def allocate_host_tensor(self, shape, dtype) -> torch.Tensor:
        self._ensure_open()
        return self._offload.empty(shape, dtype=dtype)

    @staticmethod
    def _int32_bits(value: int) -> int:
        if not 0 <= value <= UINT32_MAX:
            raise ValueError(
                "MemFabric sparse-copy length exceeds uint32: "
                f"{value} bytes")
        return value if value < (1 << 31) else value - (1 << 32)

    def copy_batch(self, tasks: list[H2DCopyTask]) -> None:
        self._ensure_open()
        if not tasks:
            return

        src_values = [task.source.data_ptr() for task in tasks]
        dst_values = [task.destination.data_ptr() for task in tasks]
        length_values = [self._int32_bits(task.nbytes) for task in tasks]

        # Release 1.2's sparse-copy kernel splits the descriptor list into two
        # equal halves. Pad odd batches with a zero-byte no-op descriptor so no
        # real task is dropped.
        if len(tasks) % 2:
            src_values.append(src_values[-1])
            dst_values.append(dst_values[-1])
            length_values.append(0)

        src_ptrs = torch.tensor(src_values,
                                dtype=torch.int64,
                                device=self.device)
        dst_ptrs = torch.tensor(dst_values,
                                dtype=torch.int64,
                                device=self.device)
        lengths = torch.tensor(length_values,
                               dtype=torch.int32,
                               device=self.device)
        count = torch.tensor(len(src_values),
                             dtype=torch.int32,
                             device=self.device)
        with self._descriptor_lock:
            self._ensure_open()
            ret = self._offload.sparse_copy(src_ptrs, dst_ptrs, lengths,
                                            count, self.device)
            if ret != 0:
                raise RuntimeError(
                    "MemFabric sparse_copy failed: "
                    f"device={self.device_id}, tasks={len(tasks)}, ret={ret}")
            self._inflight_descriptors.append(
                (src_ptrs, dst_ptrs, lengths, count))

    def synchronize(self, stream) -> None:
        with self._descriptor_lock:
            stream.synchronize()
            self._inflight_descriptors.clear()

    def close(self) -> None:
        with self._descriptor_lock:
            if self._closed:
                return
            self._offload.uninitialize()
            self._inflight_descriptors.clear()
            self._closed = True
        atexit.unregister(self.close)


class MemFabricSharedH2DTransport(MemFabricLocalH2DTransport):
    """Multi-card SHARED DRAM backend with one contribution per EP rank."""

    supports_remote_sources = True

    def __init__(
        self,
        device_id: int,
        pool_size_gib: int,
        world_size: int,
        rank_id: int,
        log_level: int = 3,
        *,
        mf_module=None,
        offload_module=None,
    ) -> None:
        if world_size <= 1:
            raise ValueError(
                "MemFabric SHARED world_size must be greater than one")
        if not 0 <= rank_id < world_size:
            raise ValueError(
                f"Invalid MemFabric SHARED rank: {rank_id}/{world_size}")
        if pool_size_gib <= 0:
            raise ValueError(
                "MemFabric pool size must be positive; "
                f"got {pool_size_gib} GiB")
        if mf_module is None or offload_module is None:
            try:
                import memfabric_hybrid as mf
                from memfabric_hybrid import offload
            except ImportError as exc:
                raise RuntimeError(
                    "h2d_backend='memfabric' requires memfabric_hybrid; "
                    "install it and source the MemFabric environment first"
                ) from exc
            mf_module = mf
            offload_module = offload

        self.device_id = device_id
        self.device = torch.device(f"npu:{device_id}")
        self._offload = offload_module
        self._closed = False
        self._inflight_descriptors = []
        self._descriptor_lock = threading.Lock()
        self.world_size = world_size
        self.rank_id = rank_id

        mf_module.set_log_level(log_level)
        config = offload_module.OffloadConfig()
        config.device_id = device_id
        config.reserve_size = pool_size_gib * ONE_GIB
        config.alloc_size = config.reserve_size
        config.world_size = world_size
        config.rank_id = rank_id
        config.scene = offload_module.Scene.SHARED
        ret = offload_module.initialize(config)
        if ret != 0:
            raise RuntimeError(
                "MemFabric SHARED initialization failed: "
                f"device={device_id}, rank={rank_id}/{world_size}, "
                f"pool_size_gib={pool_size_gib}, ret={ret}")
        atexit.register(self.close)


def create_h2d_transport(
    backend: str,
    *,
    device_id: int,
    memfabric_pool_size_gib: int,
    memfabric_log_level: int,
    enable_multi_card: bool = False,
    world_size: int = 1,
    rank_id: int = 0,
) -> ExpertH2DTransport:
    if backend == "torch":
        return TorchCopyH2DTransport()
    if backend == "memfabric":
        if enable_multi_card:
            return MemFabricSharedH2DTransport(
                device_id=device_id,
                pool_size_gib=memfabric_pool_size_gib,
                world_size=world_size,
                rank_id=rank_id,
                log_level=memfabric_log_level,
            )
        return MemFabricLocalH2DTransport(
            device_id=device_id,
            pool_size_gib=memfabric_pool_size_gib,
            log_level=memfabric_log_level,
        )
    raise ValueError(f"Unsupported expert-offload H2D backend: {backend}")
