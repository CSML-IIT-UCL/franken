from typing import List, Protocol, runtime_checkable

import metatomic.torch
import torch

from franken.data.base import Configuration


@runtime_checkable
class AtomisticModelWrapper(Protocol):
    def descriptors(self, data: Configuration) -> torch.Tensor: ...

    def feature_dim(self) -> int: ...

    def cutoff_radius(self) -> float: ...

    def num_interaction_layers(self) -> int: ...

    def supported_atomic_types(self) -> torch.Tensor: ...

    def franken_train(self) -> None: ...

    def franken_val(self) -> None: ...

    def get_neighbors(self, partial_config: Configuration) -> Configuration: ...


@runtime_checkable
class MetatomicModelWrapper(AtomisticModelWrapper, Protocol):
    def requested_neighbor_lists(self) -> List[metatomic.torch.NeighborListOptions]: ...
