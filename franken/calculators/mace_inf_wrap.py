import logging
import os
import argparse
from typing import Optional
from ase.data import chemical_symbols
import torch
from e3nn.util import jit

from franken.backbones.wrappers.base import AtomisticModelWrapper
import franken.data.base
from franken.data.base import Configuration, LAMMPSMLIAPData
from franken.rf.model import FrankenPotential

try:
    from lammps.mliap.mliap_unified_abc import MLIAPUnified
except ImportError:

    class MLIAPUnified:
        def __init__(self):
            pass


logger = logging.getLogger("franken")


@jit.compile_mode("script")
class MaceInferenceWrapper(torch.nn.Module):
    def __init__(
        self,
        franken_model: FrankenPotential,
    ):
        """Wraps Franken with a MACE-compatible GNN for inference. This is compatible only with MACE-LAMMPS calculator.

        This model can be jitted and exported for use with the MACE LAMMPS fork.

        Args:
            franken_model (FrankenPotential): The base franken model used in this MD calculator

        Note:
            The backbone underlying the franken model must be a MACE model. This is because we
            are re-using the LAMMPS interface developed by the MACE authors.
        """
        super().__init__()

        self.model = franken_model
        self.model.gnn.franken_val()
        # the following buffers are all required by the LAMMPS-MACE implementation
        self.register_buffer("atomic_numbers", self.model.gnn.supported_atomic_types())
        self.register_buffer("r_max", torch.tensor(self.model.gnn.cutoff_radius()))
        self.register_buffer(
            "num_interactions", torch.tensor(self.model.gnn.num_interaction_layers())
        )
        # this attribute is used for dtype detection in LAMMPS-MACE.
        # See: https://github.com/ACEsuit/lammps/blob/mace/src/ML-MACE/pair_mace.cpp#314
        # we setup a dummy attribute
        dtype = torch.float64  # only supported dtype in Kokkos
        if hasattr(self.model.gnn, "node_embedding"):
            self.model.node_embedding = self.model.gnn.node_embedding
        else:
            setattr(self.model, "node_embedding", type("Attr", (), {})())
            setattr(self.model.node_embedding, "linear", type("Attr", (), {})())
            setattr(self.model.node_embedding.linear, "weight", type("Attr", (), {})())
            setattr(self.model.node_embedding.linear.weight, "dtype", dtype)
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(
        self,
        data: dict[str, torch.Tensor],
        local_or_ghost: torch.Tensor,
        compute_virials: bool = False,
    ) -> dict[str, torch.Tensor | None]:
        """Compute energies and forces of a given configuration.

        This module is meant to be used in conjunction with LAMMPS,
        and this function should not be called directly. The format of
        the input data is designed to work with the MACE-LAMMPS fork.

        Warning:
            Stresses and virials are not supported by franken. Since they
            are required to be set by LAMMPS, this function sets them to tensors
            of the appropriate shape filled with zeros. Make sure that
            the chosen MD method does not depend on these quantities.
        """
        # node_attrs is a one-hot representation of the atom types. atom_nums should be the actual atomic numbers!
        # we rely on correct sorting. This is the same as in MACE.
        atom_nums = self.atomic_numbers[  # pyright: ignore[reportIndexIssue]
            torch.argmax(data["node_attrs"], dim=1)
        ]

        franken_data = Configuration(
            atom_pos=data["positions"].double(),
            atomic_numbers=atom_nums,
            natoms=torch.tensor(
                len(atom_nums), dtype=torch.int32, device=atom_nums.device
            ).view(1),
            edge_index=data["edge_index"].transpose(0, 1),
            shifts=data["shifts"].double(),
            unit_shifts=data["unit_shifts"],
            cell=data["cell"].double(),
        )
        targets = [
            franken.data.base.ENERGY_KEY,
            franken.data.base.FORCES_KEY,
        ]
        if compute_virials:
            targets.append(franken.data.base.STRESS_KEY)

        preds = self.model(targets, franken_data)
        forces = preds[franken.data.base.FORCES_KEY].squeeze(0)
        energy = preds[franken.data.base.ENERGY_KEY].squeeze(0)
        # Kokkos doesn't like total_energy_local and only looks at node_energy.
        # We hack around this:
        node_energy = energy.repeat(len(atom_nums)).div(len(atom_nums))
        virials: Optional[torch.Tensor] = None
        if compute_virials:
            virials = preds[franken.data.base.STRESS_KEY]
        return {
            "total_energy_local": energy,
            "node_energy": node_energy,
            "forces": forces,
            "virials": virials,
        }


class LAMMPSMLIAPInferenceWrapper(MLIAPUnified):
    def __init__(self, model: FrankenPotential):
        super().__init__()
        # TODO: Must assert that FrankenPotential is a MACE potential.
        self.model = model
        self.step = 0
        self.initialized = False
        self.device = torch.device("cpu")
        self.allow_cpu = self._get_env_bool("MACE_ALLOW_CPU", False)
        self.force_cpu = self._get_env_bool("MACE_FORCE_CPU", False)
        self.dtype = model.gnn.r_max.dtype
        # Attributes required by interface:
        self.ndescriptors = 1
        self.nparams = 1
        # Half of radial cutoff
        self.rcutfac = 0.5 * float(model.gnn.r_max)
        self.element_types = [chemical_symbols[s] for s in model.gnn.atomic_numbers]

    def _initialize_device(self, data):
        using_kokkos = "kokkos" in data.__class__.__module__.lower()
        if using_kokkos and not self.force_cpu:
            device = torch.as_tensor(data.elems).device
            if device.type == "cpu" and not self.allow_cpu:
                raise ValueError(
                    "GPU requested but tensor is on CPU. Set MACE_ALLOW_CPU=true to allow CPU computation."
                )
        else:
            device = torch.device("cpu")

        self.device = device
        self.model.to(device)
        self.initialized = True
        logger.info(f"FrankenPotential model initialized on device: {device}")

    def compute_forces(self, data):
        natoms = data.nlocal
        ntotal = data.ntotal
        nghosts = ntotal - natoms
        npairs = data.npairs
        species = torch.as_tensor(data.elems, dtype=torch.int64)

        if not self.initialized:
            self._initialize_device(data)

        self.step += 1
        if natoms == 0 or npairs <= 1:
            return
        franken_data = self._prepare_batch(data, natoms, nghosts, species)

        frk_out = self.model(
            franken_data,
            targets=[
                franken.data.base.EDGE_FORCES_KEY,
                franken.data.base.PER_ATOM_ENERGY_KEY,
            ],
        )

        if self.device.type != "cpu":
            torch.cuda.synchronize()

        self._update_lammps_data(
            data,
            frk_out[franken.data.base.PER_ATOM_ENERGY_KEY],
            frk_out[franken.data.base.EDGE_FORCES_KEY],
            natoms,
        )

    def _prepare_batch(self, data, natoms, nghosts, species):
        """Prepare the input batch for the FrankenPotential model."""
        # NOTE: atom_pos is unused here, but we can't set it to None.
        return Configuration(
            atom_pos=torch.zeros(natoms, 3, dtype=self.dtype, device=self.device),
            atomic_numbers=species.to(self.device),
            natoms=torch.as_tensor(natoms, dtype=torch.int32, device=self.device).view(
                1
            ),
            edge_index=torch.stack(
                [
                    torch.as_tensor(data.pair_j, dtype=torch.int64, device=self.device),
                    torch.as_tensor(data.pair_i, dtype=torch.int64, device=self.device),
                ],
                dim=1,
            ),
            batch_ids=torch.zeros(natoms, dtype=torch.int64, device=self.device),
            edge_vectors=torch.as_tensor(
                data.rij, dtype=self.dtype, device=self.device
            ),
            lammps_mliap_data=LAMMPSMLIAPData(
                natoms=natoms, nghosts=nghosts, class_data=data
            ),
        )

    def _update_lammps_data(self, data, atom_energies, pair_forces, natoms):
        """Update LAMMPS data structures with computed energies and forces."""
        if self.dtype == torch.float32:
            pair_forces = pair_forces.double()
        eatoms = torch.as_tensor(data.eatoms)
        atom_energies_real = atom_energies[:natoms].detach()
        eatoms.copy_(atom_energies_real)
        data.energy = atom_energies_real.sum().item()
        data.update_pair_forces_gpu(pair_forces)

    def compute_descriptors(self, data):
        """Required by interface but unused"""
        pass

    def compute_gradients(self, data):
        """Required by interface but unused"""
        pass


def init_wrapper(
    model_path: str,
    rf_weight_id: int | None,
    backbone_path_or_id: str | None = None,
    mliap: bool = False,
) -> str:
    """Compile a franken model into a wrapped model ready for use with MACE-LAMMPS

    Args:
        model_path (str):
            path to the franken model checkpoint.
        rf_weight_id (int | None):
            ID of the random feature weights. Can generally be left to ``None`` unless
            the checkpoint contains multiple trained models.
        backbone_path_or_id (str | None):
            Override the backbone checkpoint path stored in the
            Franken checkpoint. This is useful when paths changed
            since training.
        mliap (bool):
            Whether to use the LAMMPS ML-IAP interface instead of the
            custom MACE-LAMMPS distribution.

    Returns:
        str: the path where the LAMMPS-compatible model was saved to.
    """
    franken_model = FrankenPotential.load(
        model_path,
        map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        rf_weight_id=rf_weight_id,
        backbone_path_or_id=backbone_path_or_id,
    )

    if not isinstance(franken_model.gnn, AtomisticModelWrapper):
        raise NotImplementedError(
            f"GNN underlying the franken model ({franken_model.gnn_config.path_or_id}) is not compatible with MACE-LAMMPS."
        )

    if mliap:
        # TODO: MACE automatically converts model to cuEquivariance here. We could
        # also do it (conditionally on model type?)
        lammps_model = LAMMPSMLIAPInferenceWrapper(franken_model)
        save_path = f"{os.path.splitext(model_path)[0]}-mliap_lammps.pt"
        torch.save(lammps_model, save_path)
    else:
        # NOTE: Kokkos is hardcoded to double and will silently corrupt data
        # if the model does not use dtype double.
        franken_model = franken_model.double().to("cpu")
        lammps_model = MaceInferenceWrapper(franken_model)
        lammps_model_compiled = jit.compile(lammps_model)

        save_path = f"{os.path.splitext(model_path)[0]}-lammps.pt"
        logger.info(f"Saving compiled model to '{save_path}'")
        lammps_model_compiled.save(save_path)  # pyright: ignore[reportCallIssue]
    return save_path


def build_arg_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Convert a franken model to be able to use it with the MACE fork of LAMMPS. "
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
        "--backbone_path_or_id",
        type=str,
        help=(
            "Override the backbone checkpoint path stored in the Franken checkpoint."
        ),
        default=None,
    )
    parser.add_argument(
        "--mliap",
        action="store_true",
        help="Use the ML-IAP interface instead of the custom MACE-LAMMPS interface.",
    )
    return parser


def wrap_mace_cli():
    parser = build_arg_parser()
    args = parser.parse_args()

    init_wrapper(
        args.model_path,
        args.rf_weight_id,
        args.backbone_path_or_id,
        mliap=args.mliap,
    )


if __name__ == "__main__":
    wrap_mace_cli()


# For sphinx docs
get_parser_fn = lambda: build_arg_parser()  # noqa: E731
