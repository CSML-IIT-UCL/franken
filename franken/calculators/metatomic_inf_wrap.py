import argparse
import os
from typing import Dict, List, Optional

import torch
from metatrain.utils.sum_over_atoms import sum_over_atoms
import metatensor.torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import (
    AtomisticModel,
    ModelCapabilities,
    ModelMetadata,
    ModelOutput,
    System,
)

from franken.backbones.wrappers.base import MetatomicModelWrapper
from franken.data.base import Configuration
from franken.rf.model import FrankenPotential


class MetatomicInferenceWrapper(torch.nn.Module):
    def __init__(self, franken_model: FrankenPotential):
        super().__init__()
        self.model = franken_model
        self.model.gnn.franken_val()

    def concatenate_structures(
        self,
        systems: List[System],
    ) -> tuple[Configuration, torch.Tensor]:
        """
        Concatenate a list of systems into a single batch.

        :param systems: List of systems to concatenate.
        :param neighbor_list_options: Options for the neighbor list.
        :return: A tuple containing the concatenated positions, centers, neighbors,
            species, cells, cell shifts, system indices, and sample labels.
        """

        positions: List[torch.Tensor] = []
        centers: List[torch.Tensor] = []
        neighbors: List[torch.Tensor] = []
        species: List[torch.Tensor] = []
        cell_shifts: List[torch.Tensor] = []
        shifts: List[torch.Tensor] = []
        cells: List[torch.Tensor] = []
        system_indices: List[torch.Tensor] = []
        atom_indices: List[torch.Tensor] = []
        pbcs: List[torch.Tensor] = []
        node_counter = 0

        for i, system in enumerate(systems):
            known_neighbor_lists = system.known_neighbor_lists()
            if len(known_neighbor_lists) != 1:
                raise NotImplementedError(
                    f"Requested {len(known_neighbor_lists)} neighbor lists. We only support 1."
                )
            neighbor_list = system.get_neighbor_list(known_neighbor_lists[0])
            nl_values = neighbor_list.samples.values

            centers_values = nl_values[:, 0]
            neighbors_values = nl_values[:, 1]
            cell_shifts_values = nl_values[:, 2:]

            system_size = len(system)
            positions.append(system.positions)
            species.append(system.types)
            pbcs.append(system.pbc)

            centers.append(centers_values + node_counter)
            neighbors.append(neighbors_values + node_counter)
            cell_shifts.append(cell_shifts_values)
            cells.append(system.cell)
            shifts.append(cell_shifts_values.to(system.cell.dtype) @ system.cell)

            node_counter += system_size
            system_indices.append(
                torch.full((system_size,), i, device=system.positions.device)
            )
            atom_indices.append(
                torch.arange(system_size, device=system.positions.device)
            )

        batch_ids = torch.cat(system_indices)
        return Configuration(
            atom_pos=torch.cat(positions),
            edge_index=torch.stack([torch.cat(centers), torch.cat(neighbors)], dim=1),
            natoms=torch.bincount(batch_ids, minlength=len(systems)).to(
                dtype=torch.int64
            ),
            atomic_numbers=torch.cat(species),
            cell=torch.stack(cells, dim=0),
            unit_shifts=torch.cat(cell_shifts),
            shifts=torch.cat(shifts),
            batch_ids=batch_ids,
            pbc=torch.stack(pbcs),
        ), torch.cat(atom_indices)

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:
        if list(outputs.keys()) != ["energy"]:
            raise ValueError(
                "this model can only compute 'energy', but `outputs` contains other "
                f"keys: {', '.join(outputs.keys())}"
            )

        device = systems[0].positions.device
        # Concatenate systems into a single Configuration object
        concat_data, atom_indices = self.concatenate_structures(systems)
        batch_ids = concat_data.batch_ids
        assert batch_ids is not None
        # Compute energy with underlying model. This is per-system energy
        energy = self.model.energy(None, concat_data)  # [M, N]
        # Convert it to per-atom energy
        energy = energy / concat_data.natoms[None, ...]  # [M, N]
        energy = torch.gather(energy, dim=1, index=batch_ids[None, ...])  # [M, A]

        # Build the weird output format required by metatomic
        # 1. Build sample labels
        sample_values = torch.stack([batch_ids, atom_indices], dim=1)
        sample_labels = Labels(names=["system", "atom"], values=sample_values)
        # 2. Energy block
        energy_block = TensorBlock(
            values=energy.reshape(-1, 1),
            samples=sample_labels,
            components=torch.jit.annotate(List[Labels], []),
            properties=Labels("energy", torch.tensor([[0]], device=device)),
        )
        out_tmap = {
            "energy": TensorMap(
                keys=Labels("_", torch.tensor([[0]], device=device)),
                blocks=[energy_block],
            )
        }

        # Output filtering. Copied from PET model in metatomic
        # If selected atoms request is provided, we slice the atomic predictions
        # tensor maps to get the predictions for the selected atoms only.
        if selected_atoms is not None:
            for output_name, tmap in out_tmap.items():
                out_tmap[output_name] = metatensor.torch.slice(
                    tmap, axis="samples", selection=selected_atoms
                )

        # If per-atom predictions are requested, we return the atomic predictions
        # tensor maps. Otherwise, we sum the atomic predictions over the atoms
        # to get the final per-structure predictions for each requested output.
        for output_name, atomic_property in out_tmap.items():
            if outputs[output_name].per_atom:
                out_tmap[output_name] = atomic_property
            else:
                out_tmap[output_name] = sum_over_atoms(atomic_property)

        return out_tmap


def create_metatomic(
    model_path: str, rf_weight_id: int | None, dtype: torch.dtype
) -> str:
    """Compile a franken model into a metatomic model wrapper

    Args:
        model_path (str):
            path to the franken model checkpoint.
        rf_weight_id (int | None):
            ID of the random feature weights. Can generally be left to ``None`` unless
            the checkpoint contains multiple trained models.

    Returns:
        str: the path where the metatomic model was saved to.
    """
    franken_model = FrankenPotential.load(
        model_path,
        map_location=torch.device("cpu"),
        rf_weight_id=rf_weight_id,
    )
    if not isinstance(franken_model.gnn, MetatomicModelWrapper):
        raise NotImplementedError(
            f"GNN underlying the franken model ({franken_model.gnn_config.path_or_id}) is not compatible with Metatomic."
        )
    franken_model = franken_model.to(device="cpu", dtype=dtype)
    mta_wrapper = MetatomicInferenceWrapper(franken_model)

    base_metadata = getattr(franken_model.gnn, "metadata", {})
    metadata = ModelMetadata(
        name="franken",
        description="Franken model wrapping a pretrained deep-learning atomistic potential",
        authors=["Giacomo Meanti"],
        references={
            "model": ["https://arxiv.org/abs/2505.05652"],
        }
        | base_metadata,
    )
    outputs = {
        "energy": ModelOutput(quantity="energy", unit="eV", per_atom=False),
    }
    capabilities = ModelCapabilities(
        outputs=outputs,
        atomic_types=franken_model.gnn.supported_atomic_types().tolist(),
        interaction_range=franken_model.gnn.cutoff_radius()
        * franken_model.gnn.num_interaction_layers(),
        length_unit="angstrom",
        supported_devices=["cuda", "cpu"],
        dtype="float32" if dtype == torch.float32 else "float64",
    )
    wrapper = AtomisticModel(
        mta_wrapper.eval(),  # pyright: ignore[reportArgumentType]
        metadata,
        capabilities,
    )
    save_path = f"{os.path.splitext(model_path)[0]}-metatomic.pt"
    print(f"Saving metatomic model to '{save_path}'")
    wrapper.save(save_path)  # pyright: ignore[reportCallIssue]
    return save_path


def build_arg_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Convert a franken model to for use with Metatomic-compatible calculators. "
            "This includes calculators based on LAMMPS and ASE. "
            "The wrapped model can be based on MACE or PET based GNNs."
        ),
    )
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to the model to be converted to LAMMPS",
    )
    parser.add_argument(
        "--rf_weight_id",
        type=int,
        help="Head of the model to be converted to LAMMPS",
        default=None,
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "float64"],
        help="Data-type in which the model will run",
        required=True,
    )
    return parser


def wrap_metatomic_cli():
    parser = build_arg_parser()
    args = parser.parse_args()
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    create_metatomic(args.model_path, args.rf_weight_id, dtype)


if __name__ == "__main__":
    wrap_metatomic_cli()


# For sphinx docs
get_parser_fn = lambda: build_arg_parser()  # noqa: E731
