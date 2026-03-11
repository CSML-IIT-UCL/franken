from typing import List, Optional, Tuple
import warnings

import metatrain.pet
import torch
import metatomic.torch
import metatrain.utils.io
from metatrain.pet.modules.adaptive_cutoff import get_adaptive_cutoffs
from metatrain.pet.modules.nef import (
    compute_reversed_neighbor_list,
    edge_array_to_nef,
    get_corresponding_edges,
    get_nef_indices,
)
from metatrain.pet.modules.utilities import cutoff_func_bump, cutoff_func_cosine

from franken.data import Configuration


def systems_to_batch(
    positions: torch.Tensor,
    edge_index: torch.Tensor,
    cell: torch.Tensor,
    cell_shifts: torch.Tensor,
    species: torch.Tensor,
    options: metatomic.torch.NeighborListOptions,
    species_to_species_index: torch.Tensor,
    cutoff_function: str,
    cutoff_width: float,
    num_neighbors_adaptive: Optional[float] = None,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """
    FRANKEN: Adapted this function from `metatrain.pet.modules.structures.systems_to_batch`
        such that: 1. system_indices, sample_labels are not used - they caused errors with forward autograd
        and are not used for feature computation. 2. input is taken as a config, and no concatenation is needed.
    Converts a list of systems to a batch required for the PET model.

    :param systems: List of systems to convert to a batch.
    :param options: Options for the neighbor list.
    :param all_species_list: List of all atomic species in the dataset.
    :param species_to_species_index: Mapping from atomic species to species indices.
    :param cutoff_function: Type of the smoothing function at the cutoff.
    :param cutoff_width: Width of the cutoff function for a cutoff mask.
    :param num_neighbors_adaptive: Optional maximum number of neighbors per atom.
        If provided, the adaptive cutoff scheme will be used for each atom to
        approximately select this number of neighbors.
    :return: A tuple containing the batch tensors.
        The batch consists of the following tensors:
        - `element_indices_nodes`: The atomic species of the central atoms
        - `element_indices_neighbors`: The atomic species of the neighboring atoms
        - `edge_vectors`: The cartesian edge vectors between the central atoms and their
            neighbors
        - `edge_distances`: The distances between the central atoms and their neighbors
        - `padding_mask`: A padding mask indicating which neighbors are real, and which
            are padded
        - `reverse_neighbor_index`: The reversed neighbor list for each central atom
        - `cutoff_factors`: The cutoff function values for each edge
        - `system_indices`: The system index for each atom in the batch
        - `sample_labels`: Labels indicating the system and atom indices for each atom

    """
    centers = edge_index[:, 0]
    neighbors = edge_index[:, 1]
    cells = cell.unsqueeze(0)  # Franken: unsqueeze needed to 'fake' multiple systems

    # somehow the backward of this operation is very slow at evaluation,
    # where there is only one cell, therefore we simplify the calculation
    # for that case
    cell_contributions = cell_shifts.to(cells.dtype) @ cells[0]
    edge_vectors = positions[neighbors] - positions[centers] + cell_contributions
    edge_distances = torch.norm(edge_vectors, dim=-1) + 1e-15

    num_nodes = len(positions)

    if num_neighbors_adaptive is not None:
        with torch.profiler.record_function("PET::get_adaptive_cutoffs"):
            # Adaptive cutoff scheme to approximately select `num_neighbors_adaptive`
            # neighbors for each atom
            atomic_cutoffs = get_adaptive_cutoffs(
                centers,
                edge_distances,
                num_neighbors_adaptive,
                num_nodes,
                options.cutoff,
                cutoff_width=cutoff_width,
            )
            # Symmetrize the cutoffs between pairs of atoms (PET needs this symmetry
            # due to its corresponding edge indexing ij -> ji)
            pair_cutoffs = (atomic_cutoffs[centers] + atomic_cutoffs[neighbors]) / 2.0
        with torch.profiler.record_function("PET::adaptive_cutoff_masking"):
            # Apply cutoff mask
            cutoff_mask = edge_distances <= pair_cutoffs

            pair_cutoffs = pair_cutoffs[cutoff_mask]
            centers = centers[cutoff_mask]
            neighbors = neighbors[cutoff_mask]
            edge_vectors = edge_vectors[cutoff_mask]
            cell_shifts = cell_shifts[cutoff_mask]
            edge_distances = edge_distances[cutoff_mask]
    else:
        pair_cutoffs = options.cutoff * torch.ones(
            len(centers), device=positions.device, dtype=positions.dtype
        )

    num_neighbors = torch.bincount(centers, minlength=num_nodes)
    # this logic shouldn't be needed thanks to `minlength` above, but just to be safe:
    max_edges_per_node = (
        int(torch.max(num_neighbors)) if num_neighbors.numel() > 0 else 0
    )

    # uncomment these to print out stats on the adaptive cutoff behavior
    # print("adaptive_cutoffs", *pair_cutoffs.tolist())
    # print("num_neighbors", *num_neighbors.tolist())

    if cutoff_function.lower() == "bump":
        # use bump switching function for adaptive cutoff
        cutoff_factors = cutoff_func_bump(edge_distances, pair_cutoffs, cutoff_width)
    elif cutoff_function.lower() == "cosine":
        # backward-compatible cosine swithcing for fixed cutoff
        cutoff_factors = cutoff_func_cosine(edge_distances, pair_cutoffs, cutoff_width)
    else:
        raise ValueError(
            f"Unknown cutoff function type: {cutoff_function}. "
            f"Supported types are 'Cosine' and 'Bump'."
        )

    # Convert to NEF (Node-Edge-Feature) format:
    nef_indices, nef_to_edges_neighbor, nef_mask = get_nef_indices(
        centers, num_nodes, max_edges_per_node
    )

    # Element indices
    element_indices_nodes = species_to_species_index[species]
    element_indices_neighbors = element_indices_nodes[neighbors]

    # Send everything to NEF:
    edge_vectors = edge_array_to_nef(edge_vectors, nef_indices)
    edge_distances = torch.sqrt(torch.sum(edge_vectors**2, dim=2) + 1e-15)
    element_indices_neighbors = edge_array_to_nef(
        element_indices_neighbors, nef_indices
    )
    cutoff_factors = edge_array_to_nef(cutoff_factors, nef_indices, nef_mask, 0.0)

    corresponding_edges = get_corresponding_edges(
        torch.concatenate(
            [centers.unsqueeze(-1), neighbors.unsqueeze(-1), cell_shifts],
            dim=-1,
        )
    )

    # These are the two arrays we need for message passing with edge reversals,
    # if indexing happens in a two-dimensional way:
    # edges_ji = edges_ij[reversed_neighbor_list, neighbors_index]
    reversed_neighbor_list = compute_reversed_neighbor_list(
        nef_indices, corresponding_edges, nef_mask
    )
    neighbors_index = edge_array_to_nef(neighbors, nef_indices).to(torch.int64)

    # Here, we compute the array that allows indexing into a flattened
    # version of the edge array (where the first two dimensions are merged):
    reverse_neighbor_index = (
        neighbors_index * neighbors_index.shape[1] + reversed_neighbor_list
    )
    # At this point, we have `reverse_neighbor_index[~nef_mask] = 0`, which however
    # creates too many of the same index which slows down backward enormously.
    # (See see https://github.com/pytorch/pytorch/issues/41162)
    # We therefore replace the padded indices with a sequence of unique indices.
    reverse_neighbor_index[~nef_mask] = torch.arange(
        int(torch.sum(~nef_mask)), device=reverse_neighbor_index.device
    )

    return (
        element_indices_nodes,
        element_indices_neighbors,
        edge_vectors,
        edge_distances,
        nef_mask,
        reverse_neighbor_index,
        cutoff_factors,
    )


class PETModelWrapper(torch.nn.Module):
    def __init__(self, base_model: torch.nn.Module, gnn_backbone_id):
        super().__init__()
        self.base_model = self.get_pet_model(base_model)
        self.gnn_backbone_id = gnn_backbone_id
        # Save useful hyperparameters here
        self.cutoff = self.base_model.cutoff
        self.num_layers = (
            self.base_model.num_gnn_layers
        )  # NOTE: this doesn't consider long-range
        self.atomic_types = torch.tensor(
            self.base_model.atomic_types, dtype=torch.int64
        )

    def init_args(self):
        return {
            "gnn_backbone_id": self.gnn_backbone_id,
        }

    def get_pet_model(self, base_model: torch.nn.Module) -> metatrain.pet.PET:
        # extract the underlying PET model. Wrapped under two layers:
        # base_model is `metatomic.torch.AtomisticModel` wrapper
        # an inner LLPR (for uncertainty quantification) wrapper is optional
        llpr_model = base_model.module
        if hasattr(llpr_model, "model"):
            pet_model: torch.nn.Module = (
                llpr_model.model
            )  # pyright: ignore[reportAssignmentType, reportAttributeAccessIssue]
        else:
            pet_model = llpr_model
        return pet_model

    def descriptors(self, data: Configuration) -> torch.Tensor:
        nl_options = self.base_model.requested_neighbor_lists()[0]

        # Make torch jit script happy by having everything in local variables
        edge_index = data.edge_index
        cell = data.cell
        cell_shifts = data.unit_shifts
        assert cell_shifts is not None
        assert edge_index is not None
        assert cell is not None
        species = data.atomic_numbers
        # **Stage 0: Input Preparation**
        (
            element_indices_nodes,
            element_indices_neighbors,
            edge_vectors,
            edge_distances,
            padding_mask,
            reverse_neighbor_index,
            cutoff_factors,
        ) = systems_to_batch(
            data.atom_pos,
            edge_index,
            cell,
            cell_shifts,
            species,
            nl_options,
            self.base_model.species_to_species_index,  # pyright: ignore[reportArgumentType]
            self.base_model.cutoff_function,
            self.base_model.cutoff_width,
            self.base_model.num_neighbors_adaptive,
        )
        # Franken: use_manual_attention switches FlashAttention off. It is required for forward autograd!
        use_manual_attention = True
        # **Stage 1: Feature Computation via GNN Layers**
        featurizer_inputs: dict[str, torch.Tensor] = dict(
            element_indices_nodes=element_indices_nodes,
            element_indices_neighbors=element_indices_neighbors,
            edge_vectors=edge_vectors,
            edge_distances=edge_distances,
            reverse_neighbor_index=reverse_neighbor_index,
            padding_mask=padding_mask,
            cutoff_factors=cutoff_factors,
        )
        node_features_list, edge_features_list = self.base_model._calculate_features(
            featurizer_inputs,
            use_manual_attention=use_manual_attention,
        )  # pyright: ignore[reportCallIssue]
        return node_features_list[0]

    def feature_dim(self) -> int:
        dim: int = self.base_model.d_node
        return dim

    def cutoff_radius(self) -> float:
        return self.cutoff

    def num_interaction_layers(self) -> int:
        return self.num_layers

    def supported_atomic_types(self) -> torch.Tensor:
        return self.atomic_types

    def requested_neighbor_lists(self) -> List[metatomic.torch.NeighborListOptions]:
        return self.base_model.requested_neighbor_lists()

    @staticmethod
    def load_from_checkpoint(
        trainer_ckpt, gnn_backbone_id: str, map_location=None
    ) -> "PETModelWrapper":
        with warnings.catch_warnings():
            warnings.filterwarnings(
                action="ignore",
                message="PET assumes that Cartesian tensors of rank 2 are stress-like",
            )
            loaded_model = metatrain.utils.io.load_model(trainer_ckpt)
            loaded_model = loaded_model.export()  # no metadata?
        return PETModelWrapper(base_model=loaded_model, gnn_backbone_id=gnn_backbone_id)
