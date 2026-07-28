"""Franken model"""

import logging
import os
from typing import Literal, Mapping, Optional, Union

import torch

from franken.config import BackboneConfig, LESConfig, RFConfig
from franken.data import Configuration
import franken.data.base
from franken.les.les_head import initialize_les
from franken.rf.model import FrankenPotential
from franken.utils.derivatives import forces_bwdad, forces_stress_bwdad

logger = logging.getLogger("franken")


class LESFrankenPotential(FrankenPotential):
    def __init__(
        self,
        gnn_config: BackboneConfig,
        rf_config: RFConfig,
        les_config: LESConfig,
        jac_chunk_size: Union[int, Literal["auto"]] = "auto",
        scale_by_Z: bool = True,
        num_species: int = 1,
        atomic_energies: Optional[Mapping[int, torch.Tensor | float]] = None,
    ):
        super(LESFrankenPotential, self).__init__(
            gnn_config=gnn_config,
            rf_config=rf_config,
            jac_chunk_size=jac_chunk_size,
            scale_by_Z=scale_by_Z,
            num_species=num_species,
            atomic_energies=atomic_energies,
        )
        self.les_config = les_config
        self.les = initialize_les(
            les_config=les_config, feature_dim=self.gnn.feature_dim()
        )

    @property
    @torch.jit.unused
    def hyperparameters(self):
        hps = super().hyperparameters
        hps["les"] = self.les_config.to_ckpt()
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
    def load(
        cls,
        path,
        map_location=None,
        rf_weight_id: int | None = None,
        backbone_path_or_id: str | None = None,
    ):
        ckpt = torch.load(path, map_location=map_location, weights_only=False)

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
        les_energies = self.les(gnn_descriptors, atom_pos, data)
        energies = rff_energies + les_energies

        return energies.sum(1), energies

    def _les_energy_aux(
        self,
        atom_pos: torch.Tensor,
        displacement: torch.Tensor | None,
        data: Configuration,
    ):
        gnn_descriptors = self.descriptors(atom_pos, displacement, data)
        les_energies = self.les(gnn_descriptors, atom_pos, data)
        return les_energies, les_energies

    def _predict(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        weights: torch.Tensor | None,
        data: Configuration,
        targets: list[str],
        is_training: bool,
    ) -> dict[str, torch.Tensor]:
        if weights is None:
            weights = self.rf.weights
        compute_force = franken.data.base.FORCES_TARGET_KEY in targets
        compute_stress = franken.data.base.STRESS_TARGET_KEY in targets
        if compute_stress:
            return forces_stress_bwdad(
                data, fn=self._energy_aux, is_training=is_training, weights=weights
            )
        elif compute_force:
            return forces_bwdad(
                data, fn=self._energy_aux, is_training=is_training, weights=weights
            )
        else:
            _, energy = self._energy_aux(data.atom_pos, None, data, weights)  # [M, N]
            return {franken.data.base.ENERGY_TARGET_KEY: energy}

    def predict_les(
        self,
        data: Configuration,
        targets: list[str],
        is_training: bool = False,
    ) -> dict[str, torch.Tensor]:
        compute_force = franken.data.base.FORCES_TARGET_KEY in targets
        compute_stress = franken.data.base.STRESS_TARGET_KEY in targets
        if compute_stress:
            return forces_stress_bwdad(
                data, fn=self._les_energy_aux, is_training=is_training
            )
        elif compute_force:
            return forces_bwdad(data, fn=self._les_energy_aux, is_training=is_training)
        else:
            _, energy = self._les_energy_aux(data.atom_pos, None, data)  # [M, N]
            return {franken.data.base.ENERGY_TARGET_KEY: energy}

    def predict(
        self,
        targets: list[str],
        data: Configuration,
        weights: torch.Tensor | None = None,
        differential_mode: str = "torch.autograd",
        add_energy_shift: bool = True,
        is_training: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Infer energy, forces and other quantities for an atomic system with a learned RF model.

        The parameter `weights` can be used to specified the model's coefficients. Otherwise the ones stored in
        :attr:`FrankenPotential.rf.weights` will be used instead.

        Args:
            targets: the target quantities to compute. For example, :code:`"energy"`, :code:`"forces"`
                or :code:`"stress"`. To see all available quantities, check :attr:`"franken.data.base.TargetType"`.
            weights: weights of the random feature model. Defaults to None, in which case
                the weights set in :attr:`FrankenPotential.rf` will be used instead.
            add_energy_shift: whether to add the energy shift to the energy.

        Returns:
            A dictionary mapping requested targets to the computed values.
            Each requested target has a first dimension which depends on the number of models present
            in the current weights. The second dimension depends on the number of separate systems present
            in the data. Further dimensions depend on the specific target. For example,
            forces have size `[num_models, num_systems * num_atoms_per_system, 3]`; stress tensors instead
            have size `[num_models, num_systems, 3, 3]` and energy tensors have size `[num_models, num_systems]`.

        Note:
            The `"torch.func"` strategy for differentiation is not supported for torch-jitted models.
            Use `"torch.autograd"` if the model has been processed by :code:`torch.jit.script`.
        """
        natoms = torch.atleast_1d(data.natoms)
        out = self._predict(weights, data, targets, is_training=is_training)

        if add_energy_shift and franken.data.base.ENERGY_TARGET_KEY in targets:
            out[franken.data.base.ENERGY_TARGET_KEY] = out[
                franken.data.base.ENERGY_TARGET_KEY
            ] + self.energy_shift(
                data.atomic_numbers,
                batch_ids=data.batch_ids,
                num_systems=int(natoms.numel()),
            )
        return out  # ([M, N], [M, A, 3])

    def forward(
        self,
        targets: list[str],
        data: Configuration,
        weights: torch.Tensor | None = None,
        add_energy_shift: bool = True,
    ) -> dict[str, torch.Tensor]:
        """
        See docstring of :meth:`~franken.rf.model.FrankenPotential.predict`.

        This function defaults to using the 'torch.autograd' strategy which allows the model
        to be jit-compiled.
        """
        out = self.predict(
            targets=targets,
            data=data,
            weights=weights,
            differential_mode=(
                "torch.autograd" if not self.force_func_grad else "torch.func"
            ),
            add_energy_shift=add_energy_shift,
        )
        return {k: v.squeeze(0) for k, v in out.items()}
