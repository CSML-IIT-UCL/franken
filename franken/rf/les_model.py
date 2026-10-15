"""Franken latent Ewalds summation model"""

import logging
import os
from typing import Literal, Mapping, Optional, Union

import torch

from franken.config import BackboneConfig, LESConfig, RFConfig
from franken.data import Configuration
import franken.data.base
from franken.rf.model import FrankenPotential
from franken.utils.derivatives import forces_bwdad, forces_stress_bwdad

logger = logging.getLogger("franken")


class LESFrankenPotential(FrankenPotential):
    def __init__(
        self,
        gnn_config: BackboneConfig,
        rf_config: RFConfig,
        les_config: LESConfig,
        is_periodic: bool | None = None,
        jac_chunk_size: Union[int, Literal["auto"]] = "auto",
        scale_by_Z: bool = True,
        num_species: int = 1,
        atomic_energies: Optional[Mapping[int, torch.Tensor | float]] = None,
    ):
        try:
            from les import Les
        except ImportError as exc:
            raise ImportError(
                "LESFrankenPotential requires the external LES package. "
                "Install it with pip install 'franken[les]'."
            ) from exc
        super(LESFrankenPotential, self).__init__(
            gnn_config=gnn_config,
            rf_config=rf_config,
            jac_chunk_size=jac_chunk_size,
            scale_by_Z=scale_by_Z,
            num_species=num_species,
            atomic_energies=atomic_energies,
        )
        self.les_config = les_config
        self.is_periodic = is_periodic
        # checks on hidden_dim (keep them?)
        hidden_dim = les_config.hidden_dim
        if isinstance(hidden_dim, int):
            hidden_dim = (hidden_dim,)
        if len(hidden_dim) == 1:
            hidden_dim = hidden_dim * (les_config.n_layers - 1)
        if len(hidden_dim) != les_config.n_layers - 1:
            raise ValueError("hidden_dim must contain n_layers - 1 dimensions")

        # Initialize LES
        self.les = Les(
            les_arguments={
                "use_atomwise": True,
                "n_layers": les_config.n_layers,
                "n_hidden": list(hidden_dim),
                "add_linear_nn": les_config.add_linear_nn,
                "output_scaling_factor": les_config.les_output_scale,
                "dl": les_config.dl,
                "sigma": les_config.sigma,
                "is_periodic": is_periodic,
                "N_max": les_config.N_max,
            }
        )
        # Initialize the LES MLP immediately to avoid annoying downstream issues
        # Upstream builds its MLP on the first call
        self.les(
            positions=torch.randn(5, 3),
            cell=torch.randn(1, 3, 3),
            desc=torch.randn(5, self.gnn.feature_dim()),
        )

    @property
    @torch.jit.unused
    def hyperparameters(self):
        hps = super().hyperparameters
        hps["les"] = self.les_config.to_ckpt()
        hps["les"]["is_periodic"] = self.is_periodic
        return hps

    def save(self, path: os.PathLike | str, multi_weights: torch.Tensor | None = None):
        # TODO: We probably want to drop multi-weight support for LES model
        if multi_weights is not None:
            assert torch.is_tensor(multi_weights)
            assert multi_weights.ndim <= 2
            assert multi_weights.shape[-1] == self.rf.weights.shape[-1]

        ckpt = {
            "jac_chunk_size": self.jac_chunk_size,
            "multi_weights": multi_weights,
            "num_species": self.num_species,
            "is_periodic": self.is_periodic,
            "rf": {
                "config": self.rf_config.to_ckpt(),
                "state_dict": self.rf.state_dict(),
            },
            "input_scaler": {
                "config": self.input_scaler.init_args(),
                "state_dict": self.input_scaler.state_dict(),
            },
            "energy_shift": self.energy_shift.state_dict(),
            "gnn": {
                "config": self.gnn_config.to_ckpt(),
            },
            "les": {
                "config": self.les_config.to_ckpt(),
                "state_dict": self.les.state_dict(),
            },
        }
        torch.save(ckpt, path)

    @classmethod
    def _from_checkpoint(
        cls,
        ckpt,
        map_location=None,
        rf_weight_id: int | None = None,
        backbone_path_or_id: str | None = None,
    ):
        rf_cfg = RFConfig.from_ckpt(ckpt["rf"]["config"])
        gnn_cfg = BackboneConfig.from_ckpt(ckpt["gnn"]["config"])
        les_cfg = LESConfig.from_ckpt(ckpt["les"]["config"])
        if backbone_path_or_id is not None:
            logger.warning(
                f"The backbone path/id changed from {gnn_cfg.path_or_id} to {backbone_path_or_id}. If this refers to a different backbone, unexpected results may occur."
            )
            gnn_cfg.path_or_id = backbone_path_or_id
        model = cls(
            gnn_config=gnn_cfg,
            rf_config=rf_cfg,
            les_config=les_cfg,
            jac_chunk_size=ckpt["jac_chunk_size"],
            num_species=ckpt["num_species"],
            is_periodic=ckpt["is_periodic"],
            **ckpt["input_scaler"]["config"],
        )
        model.rf.load_state_dict(ckpt["rf"]["state_dict"])
        model.input_scaler.load_state_dict(ckpt["input_scaler"]["state_dict"])
        model.energy_shift.load_state_dict(ckpt["energy_shift"])
        model.les.load_state_dict(ckpt["les"]["state_dict"])

        if (
            ckpt["multi_weights"] is not None
        ):  # TODO: We probably want to drop multi-weight support for LES model
            if rf_weight_id is None:
                raise ValueError(
                    f"The checkpoint contains {ckpt['multi_weights'].shape[0]}, select which one to load by specifying rf_weight_id"
                )
            assert rf_weight_id < ckpt["multi_weights"].shape[0]
            model.rf.weights.copy_(
                ckpt["multi_weights"][rf_weight_id].reshape_as(model.rf.weights)
            )

        if map_location is not None:
            return model.to(map_location)
        else:
            return model

    def _get_les_outputs(
        self,
        atom_pos: torch.Tensor,
        descriptors: torch.Tensor,
        data: Configuration,
    ) -> dict[str, torch.Tensor]:
        cell = data.cell
        if cell is None:
            cell = torch.zeros(len(data.natoms), 3, 3)
        if cell.dim() == 2:
            cell = cell.unsqueeze(0)
        les_out_internal = self.les(
            positions=atom_pos,
            cell=cell,
            desc=descriptors,
            batch=data.batch_ids,
        )
        # epsilon_0 = 0.00552635  # e^2 eV^-1 A^-1
        q_normalisation_factor = 1  # (2 * epsilon_0) ** 0.5
        les_out = {
            franken.data.base.ENERGY_TARGET_KEY: les_out_internal["E_lr"].reshape(
                1, -1
            ),
            franken.data.base.LES_CHARGES_TARGET_KEY: les_out_internal["latent_charges"]
            * q_normalisation_factor,
        }
        return les_out

    def _energy_aux(
        self,
        atom_pos: torch.Tensor,
        displacement: torch.Tensor | None,
        data: Configuration,
        weights: torch.Tensor | None,
    ):
        # weights: [num weights(M), num features(F)]
        if weights is None:
            weights = self.rf.weights
        gnn_descriptors = self.descriptors(atom_pos, displacement, data)
        random_features = self.rfs_from_descriptor(gnn_descriptors, data)
        random_features = random_features.to(dtype=weights.dtype)  # [N, F]

        natoms = data.natoms.to(dtype=weights.dtype).view(-1)  # [N]
        rff_energies = torch.matmul(random_features, weights.T).T  # [M, N]
        rff_energies = natoms[None, :] * rff_energies
        les_out = self._get_les_outputs(atom_pos, gnn_descriptors, data)
        tot_energy = rff_energies + les_out[franken.data.base.ENERGY_TARGET_KEY]
        les_out[franken.data.base.ENERGY_TARGET_KEY] = tot_energy
        return tot_energy.sum(1), les_out

    def _les_energy_aux(
        self,
        atom_pos: torch.Tensor,
        displacement: torch.Tensor | None,
        data: Configuration,
    ):
        gnn_descriptors = self.descriptors(atom_pos, displacement, data)
        les_outputs = self._get_les_outputs(atom_pos, gnn_descriptors, data)
        return les_outputs[franken.data.base.ENERGY_TARGET_KEY].sum(1), les_outputs

    def _predict(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        weights: torch.Tensor | None,
        data: Configuration,
        targets: list[str],
        mode: str,
    ) -> dict[str, torch.Tensor]:
        if mode != "torch.autograd":
            raise RuntimeError(
                f"LESFrankenPotential only supports 'torch.autograd' differential mode. Found {mode}"
            )
        if weights is None:
            weights = self.rf.weights
        compute_force = franken.data.base.FORCES_TARGET_KEY in targets
        compute_stress = franken.data.base.STRESS_TARGET_KEY in targets
        if compute_stress:
            return forces_stress_bwdad(
                data, fn=self._energy_aux, is_training=self.training, weights=weights
            )
        elif compute_force:
            return forces_bwdad(
                data, fn=self._energy_aux, is_training=self.training, weights=weights
            )
        else:
            _, results = self._energy_aux(data.atom_pos, None, data, weights)  # [M, N]
            return results

    def predict_les(
        self,
        data: Configuration,
        targets: list[str],
    ) -> dict[str, torch.Tensor]:
        """Predict using **only** the LES module.

        The random-features contribution is not computed.
        """
        compute_force = franken.data.base.FORCES_TARGET_KEY in targets
        compute_stress = franken.data.base.STRESS_TARGET_KEY in targets
        if compute_stress:
            return forces_stress_bwdad(
                data, fn=self._les_energy_aux, is_training=self.training
            )
        elif compute_force:
            return forces_bwdad(
                data, fn=self._les_energy_aux, is_training=self.training
            )
        else:
            _, results = self._les_energy_aux(data.atom_pos, None, data)  # [M, N]
            return results

    @torch.jit.export
    def supported_targets(self) -> list[str]:
        """Returns the list of target keys which are supported by the model.

        Requesting a target which is unsupported from the ``forward`` or ``predict``
        methods will simply not return any result for that target, likely leading
        to downstream errors.
        """
        return [
            franken.data.base.ENERGY_TARGET_KEY,
            franken.data.base.FORCES_TARGET_KEY,
            franken.data.base.STRESS_TARGET_KEY,
            franken.data.base.LES_CHARGES_TARGET_KEY,
        ]
