from typing import Optional, Iterator

from torch.utils.data import Sampler, Dataset
import torch.distributed as dist


def calc_distributed_indices(num_samples: int, world_size: int, rank: int) -> list[int]:
    assert rank < world_size
    indices = list(range(num_samples))
    # subsample
    indices = indices[rank:num_samples:world_size]
    return indices


class SimpleUnevenDistributedSampler(Sampler):
    def __init__(
        self,
        dataset: Dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
    ) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas - 1}]"
            )
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.total_size = len(self.dataset)  # type: ignore[arg-type]
        # num_samples indicates the number of samples for the current process
        self.num_samples = len(range(self.rank, self.total_size, self.num_replicas))

    def __iter__(self) -> Iterator:
        indices = calc_distributed_indices(
            num_samples=self.total_size, world_size=self.num_replicas, rank=self.rank
        )
        assert len(indices) == self.num_samples
        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples
