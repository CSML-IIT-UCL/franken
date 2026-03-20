from functools import partial
import os
import platform
import pytest
import torch.distributed
from torch.multiprocessing import Process, Pipe, SimpleQueue

from franken.data import Configuration, FrankenAtomsDataset
from franken.datasets.registry import DATASET_REGISTRY


class ThrowingProcess(Process):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pconn, self._cconn = Pipe()
        self._exception = None

    def run(self):
        try:
            super().run()
            self._cconn.send(None)
        except Exception as e:
            self._cconn.send(e)
            raise e

    @property
    def exception(self):
        if self._pconn.poll():
            self._exception = self._pconn.recv()
        return self._exception


def init_processes(rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    loopback_address = "lo" if platform.system() == "Linux" else "lo0"
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '26515'
    os.environ['GLOO_SOCKET_IFNAME'] = loopback_address
    torch.distributed.init_process_group(backend, rank=rank, world_size=size)
    fn()


def init_distributed_cpu(num_proc, run_fn):
    processes = []
    for rank in range(num_proc):
        p = ThrowingProcess(target=init_processes, args=(rank, num_proc, run_fn))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
        if p.exception:
            error = p.exception
            raise error


def mocked_dataset(num_atoms, dtype, device, num_configs: int = 1):
    data = []
    for _ in range(num_configs):
        data.append(Configuration(
            torch.randn(num_atoms, 3, dtype=dtype),
            torch.randint(1, 100, (num_atoms,)),
            torch.tensor(num_atoms),
        ).to(device))
    return data


def distributed_length_inner_fn(num_samples):
    data_path = DATASET_REGISTRY.get_path("test", "long", None, False)
    dataset = FrankenAtomsDataset(
        data_path,
        split="train",
        num_random_subsamples=num_samples,
        subsample_rng=None,
        gnn_config=None,
    )
    assert len(dataset) == num_samples
    dataloader = dataset.get_dataloader(True)
    rank = torch.distributed.get_rank()
    ws = torch.distributed.get_world_size()
    assert len(dataloader) == (len(dataset) // ws) + int(len(dataset) % ws > rank)


@pytest.mark.parametrize("num_samples", [1, 7, 19])
@pytest.mark.parametrize("num_procs", [1, 4])
def test_distributed_dataloader_length(num_samples, num_procs):
    fn_ = partial(distributed_length_inner_fn, num_samples)
    init_distributed_cpu(num_procs, fn_)


def distributed_order_inner_fn(num_samples, num_procs, ids_queue):
    data_path = DATASET_REGISTRY.get_path("test", "long", None, False)
    dataset = FrankenAtomsDataset(
        data_path,
        split="train",
        num_random_subsamples=num_samples,
        subsample_rng=None,
        gnn_config=None
    )
    assert len(dataset) == num_samples
    dataloader = dataset.get_dataloader(True)
    rank = torch.distributed.get_rank()
    dl_elements = [el for el in dataloader]
    dl_id = 0
    for i in range(rank, num_samples, num_procs):
        expected = dataset[i]
        assert isinstance(expected, tuple)
        torch.testing.assert_close(
            dl_elements[dl_id][0].atom_pos, expected[0].atom_pos
        )
        torch.testing.assert_close(
            dl_elements[dl_id][1].forces, expected[1].forces
        )
        dl_id += 1
        ids_queue.put(i)
    assert dl_id == len(dl_elements)


def test_distributed_dataloader_order():
    num_samples = 7
    num_procs = 3
    ids_queue = SimpleQueue()
    fn_ = partial(distributed_order_inner_fn, num_samples, num_procs, ids_queue)
    init_distributed_cpu(num_procs, fn_)
    # Assert all IDs were processed - only once
    all_ids = []
    while not ids_queue.empty():
        all_ids.append(ids_queue.get())
    assert sorted(all_ids) == list(range(num_samples))
