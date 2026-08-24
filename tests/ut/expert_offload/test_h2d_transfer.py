from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest

from vllm_ascend.expert_offload.h2d_transfer import (
    H2DCopyTask,
    MemFabricLocalH2DTransport,
    MemFabricSharedH2DTransport,
    ONE_GIB,
    TorchCopyH2DTransport,
    create_h2d_transport,
)


def test_torch_transport_preserves_task_order_and_non_blocking_flags():
    first_destination = MagicMock()
    second_destination = MagicMock()
    tasks = [
        H2DCopyTask(
            source="w13-source",
            destination=first_destination,
            nbytes=16,
            name="w13",
        ),
        H2DCopyTask(
            source="scale-source",
            destination=second_destination,
            nbytes=4,
            name="scale",
            non_blocking=False,
        ),
    ]

    TorchCopyH2DTransport().copy_batch(tasks)

    assert first_destination.copy_.call_args_list == [
        call("w13-source", non_blocking=True)]
    assert second_destination.copy_.call_args_list == [
        call("scale-source", non_blocking=False)]


def test_torch_transport_accepts_an_empty_batch():
    TorchCopyH2DTransport().copy_batch([])


def test_torch_transport_allocates_pinned_host_tensor():
    expected = object()
    with patch(
        "vllm_ascend.expert_offload.h2d_transfer.torch.empty",
        return_value=expected,
    ) as empty:
        result = TorchCopyH2DTransport().allocate_host_tensor((2, 3), "dtype")

    assert result is expected
    empty.assert_called_once_with(
        (2, 3), dtype="dtype", device="cpu", pin_memory=True)


class _FakeDescriptor:

    def __init__(self, values, dtype, device):
        self.values = values
        self.dtype = dtype
        self.device = device

    def data_ptr(self):
        return id(self)


def _make_memfabric_transport(pool_size_gib=2):
    mf_module = MagicMock()
    offload_module = MagicMock()
    offload_module.OffloadConfig.side_effect = SimpleNamespace
    offload_module.initialize.return_value = 0
    offload_module.sparse_copy.return_value = 0
    offload_module.uninitialize.return_value = 0
    with (
        patch(
            "vllm_ascend.expert_offload.h2d_transfer.torch.device",
            return_value="npu:fake",
        ),
        patch(
            "vllm_ascend.expert_offload.h2d_transfer.atexit.register",
        ),
    ):
        transport = MemFabricLocalH2DTransport(
            device_id=3,
            pool_size_gib=pool_size_gib,
            log_level=1,
            mf_module=mf_module,
            offload_module=offload_module,
        )
    return transport, mf_module, offload_module


def test_memfabric_transport_initializes_local_pool_and_allocates_from_it():
    transport, mf_module, offload_module = _make_memfabric_transport()
    expected = object()
    offload_module.empty.return_value = expected

    result = transport.allocate_host_tensor((4, 8), "dtype")

    mf_module.set_log_level.assert_called_once_with(1)
    config = offload_module.initialize.call_args.args[0]
    assert config.device_id == 3
    assert config.reserve_size == 2 * ONE_GIB
    assert config.alloc_size == 2 * ONE_GIB
    assert result is expected
    offload_module.empty.assert_called_once_with((4, 8), dtype="dtype")


def test_memfabric_shared_transport_initializes_one_contribution_per_rank():
    mf_module = MagicMock()
    offload_module = MagicMock()
    offload_module.OffloadConfig.side_effect = SimpleNamespace
    offload_module.Scene.SHARED = "shared"
    offload_module.initialize.return_value = 0
    with (
        patch(
            "vllm_ascend.expert_offload.h2d_transfer.torch.device",
            return_value="npu:fake",
        ),
        patch(
            "vllm_ascend.expert_offload.h2d_transfer.atexit.register",
        ),
    ):
        transport = MemFabricSharedH2DTransport(
            device_id=2,
            pool_size_gib=4,
            world_size=8,
            rank_id=2,
            log_level=1,
            mf_module=mf_module,
            offload_module=offload_module,
        )

    config = offload_module.initialize.call_args.args[0]
    assert config.device_id == 2
    assert config.reserve_size == 4 * ONE_GIB
    assert config.alloc_size == 4 * ONE_GIB
    assert config.world_size == 8
    assert config.rank_id == 2
    assert config.scene == "shared"
    assert transport.supports_remote_sources is True


def test_memfabric_shared_transport_validates_rank_topology():
    with pytest.raises(ValueError, match="world_size"):
        MemFabricSharedH2DTransport(0, 1, 1, 0)
    with pytest.raises(ValueError, match="Invalid.*rank"):
        MemFabricSharedH2DTransport(0, 1, 4, 4)


def test_memfabric_transport_pads_odd_sparse_copy_batch_and_retires_descriptors():
    transport, _, offload_module = _make_memfabric_transport()
    sources = [MagicMock(), MagicMock(), MagicMock()]
    destinations = [MagicMock(), MagicMock(), MagicMock()]
    for index, source in enumerate(sources):
        source.data_ptr.return_value = 100 + index
    for index, destination in enumerate(destinations):
        destination.data_ptr.return_value = 200 + index
    tasks = [
        H2DCopyTask(source, destination, index + 1)
        for index, (source, destination) in enumerate(
            zip(sources, destinations))
    ]
    descriptors = []

    def make_descriptor(values, dtype, device):
        descriptor = _FakeDescriptor(values, dtype, device)
        descriptors.append(descriptor)
        return descriptor

    with patch(
        "vllm_ascend.expert_offload.h2d_transfer.torch.tensor",
        side_effect=make_descriptor,
    ):
        transport.copy_batch(tasks)

    assert descriptors[0].values == [100, 101, 102, 102]
    assert descriptors[1].values == [200, 201, 202, 202]
    assert descriptors[2].values == [1, 2, 3, 0]
    assert descriptors[3].values == 4
    offload_module.sparse_copy.assert_called_once_with(
        *descriptors, "npu:fake")
    assert len(transport._inflight_descriptors) == 1

    stream = MagicMock()
    transport.synchronize(stream)

    stream.synchronize.assert_called_once_with()
    assert transport._inflight_descriptors == []


def test_memfabric_transport_preserves_uint32_length_bits():
    assert MemFabricLocalH2DTransport._int32_bits((1 << 31) - 1) == \
        (1 << 31) - 1
    assert MemFabricLocalH2DTransport._int32_bits(1 << 31) == -(1 << 31)
    assert MemFabricLocalH2DTransport._int32_bits((1 << 32) - 1) == -1
    with pytest.raises(ValueError, match="exceeds uint32"):
        MemFabricLocalH2DTransport._int32_bits(1 << 32)


def test_memfabric_transport_reports_sparse_copy_failure():
    transport, _, offload_module = _make_memfabric_transport()
    offload_module.sparse_copy.return_value = 9
    task = H2DCopyTask(MagicMock(), MagicMock(), 4)

    with (
        patch(
            "vllm_ascend.expert_offload.h2d_transfer.torch.tensor",
            side_effect=lambda values, dtype, device: _FakeDescriptor(
                values, dtype, device),
        ),
        pytest.raises(RuntimeError, match="sparse_copy failed.*ret=9"),
    ):
        transport.copy_batch([task])

    assert transport._inflight_descriptors == []


def test_memfabric_transport_close_is_idempotent():
    transport, _, offload_module = _make_memfabric_transport()

    with patch(
        "vllm_ascend.expert_offload.h2d_transfer.atexit.unregister",
    ):
        transport.close()
        transport.close()

    offload_module.uninitialize.assert_called_once_with()
    with pytest.raises(RuntimeError, match="already closed"):
        transport.allocate_host_tensor((1,), "dtype")


def test_memfabric_transport_reports_initialization_failure():
    mf_module = MagicMock()
    offload_module = MagicMock()
    offload_module.OffloadConfig.side_effect = SimpleNamespace
    offload_module.initialize.return_value = 7

    with (
        patch(
            "vllm_ascend.expert_offload.h2d_transfer.torch.device",
            return_value="npu:fake",
        ),
        pytest.raises(RuntimeError, match="initialization failed.*ret=7"),
    ):
        MemFabricLocalH2DTransport(
            device_id=0,
            pool_size_gib=1,
            mf_module=mf_module,
            offload_module=offload_module,
        )


def test_h2d_transport_factory_rejects_unknown_backend():
    with pytest.raises(ValueError, match="Unsupported"):
        create_h2d_transport(
            "unknown",
            device_id=0,
            memfabric_pool_size_gib=0,
            memfabric_log_level=3,
        )


@pytest.mark.parametrize("nbytes", [0, -1])
def test_h2d_copy_task_requires_positive_size(nbytes):
    with pytest.raises(ValueError, match="copy size must be positive"):
        H2DCopyTask(source=object(), destination=object(), nbytes=nbytes)
