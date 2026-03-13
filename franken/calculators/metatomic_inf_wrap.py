import argparse
import os
from typing import Dict, List, Optional

import metatensor
import metatrain
import metatrain.utils
import metatrain.utils.sum_over_atoms
import torch
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

        # we don't want to worry about selected_atoms yet
        # if selected_atoms is not None:
        #     raise NotImplementedError("selected_atoms is not implemented")

        # if outputs["energy"].per_atom:
        #     raise NotImplementedError("per atom energy is not implemented")

        # Build sample labels.
        #  This was originally part of `concatenate_structures` in PET code
        system_indices_lst: List[torch.Tensor] = []
        atom_indices_lst: List[torch.Tensor] = []
        for i, system in enumerate(systems):
            system_size = len(system)
            system_indices_lst.append(
                torch.full((system_size,), i, device=system.positions.device)
            )
            atom_indices_lst.append(
                torch.arange(system_size, device=system.positions.device)
            )
        system_indices = torch.cat(system_indices_lst)
        atom_indices = torch.cat(atom_indices_lst)
        sample_values = torch.stack([system_indices, atom_indices], dim=1)
        sample_labels = Labels(
            names=["system", "atom"],
            values=sample_values,
        )

        device = systems[0].positions.device
        energy_lst: List[torch.Tensor] = []
        # torch.zeros(
        #     (len(systems), 1), dtype=systems[0].positions.dtype, device=device
        # )
        for i, system in enumerate(systems):
            known_neighbor_lists = system.known_neighbor_lists()
            if len(known_neighbor_lists) != 1:
                raise NotImplementedError(
                    f"Requested {len(known_neighbor_lists)} neighbor lists. We only support 1."
                )
            neighbor_list = system.get_neighbor_list(known_neighbor_lists[0])
            nl_values = neighbor_list.samples.values
            franken_data = Configuration(
                atom_pos=system.positions,
                atomic_numbers=system.types,
                natoms=torch.tensor(
                    len(system.types), dtype=torch.int32, device=device
                ).view(1),
                pbc=system.pbc,
                cell=system.cell,
                unit_shifts=nl_values[:, 2:],
                edge_index=nl_values[:, :2],
            )
            # Don't compute_forces. This will be done in the metatomic calculator.
            # in theory we could set the `explicit_gradients` capability in the model
            # but it doesn't seem to be actually used anywhere in the calculators (which
            # rely on performing autograd themselves)
            sys_energy, _ = self.model(franken_data, compute_forces=False)
            node_energy = sys_energy.repeat(system.positions.shape[0]).div(
                system.positions.shape[0]
            )
            energy_lst.append(node_energy)

        energy = torch.cat(energy_lst, 0)

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

        # If selected atoms request is provided, we slice the atomic predictions
        # tensor maps to get the predictions for the selected atoms only.
        if selected_atoms is not None:
            for output_name, tmap in out_tmap.items():
                out_tmap[output_name] = metatensor.slice(
                    tmap, axis="samples", selection=selected_atoms
                )

        # If per-atom predictions are requested, we return the atomic predictions
        # tensor maps. Otherwise, we sum the atomic predictions over the atoms
        # to get the final per-structure predictions for each requested output.
        for output_name, atomic_property in out_tmap.items():
            if outputs[output_name].per_atom:
                out_tmap[output_name] = atomic_property
            else:
                out_tmap[output_name] = metatrain.utils.sum_over_atoms.sum_over_atoms(
                    atomic_property
                )

        return out_tmap

        # # add metadata to the output
        # block = TensorBlock(
        #     values=energy,
        #     samples=Labels(
        #         "system", torch.arange(len(systems), device=device).reshape(-1, 1)
        #     ),
        #     components=[],
        #     properties=Labels("energy", torch.tensor([[0]], device=device)),
        # )
        # return {
        #     "energy": TensorMap(
        #         keys=Labels("_", torch.tensor([[0]], device=device)), blocks=[block]
        #     )
        # }


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
        supported_devices=["cpu", "cuda"],
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
