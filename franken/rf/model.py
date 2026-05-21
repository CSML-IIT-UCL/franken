"""Franken model"""

import logging
import os
from typing import Callable, List, Literal, Mapping, Optional, Union

import torch

from franken.config import (
    BackboneConfig,
    RFConfig,
)
from franken.backbones.utils import load_checkpoint
from franken.data import Configuration
import franken.data.base
from franken.data.base import TargetType
from franken.rf.atomic_energies import AtomicEnergiesShift
from franken.rf.heads import initialize_rf
from franken.rf.scaler import FeatureScaler
from franken.utils.jac import jacfwd, tune_jacfwd_chunksize

logger = logging.getLogger("franken")


def prep_with_displacement(
    data: Configuration, atom_pos: torch.Tensor, displacement: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    data_batch_ids = data.batch_ids
    batch_ids = (
        data_batch_ids
        if data_batch_ids is not None
        else torch.zeros(atom_pos.shape[0], dtype=torch.int32, device=atom_pos.device)
    )
    num_systems = data.natoms.numel()
    data_cell = data.cell
    cell = (
        data_cell
        if data_cell is not None
        else torch.zeros(
            num_systems * 3, 3, dtype=atom_pos.dtype, device=atom_pos.device
        )
    )
    unit_shifts = data.unit_shifts
    assert unit_shifts is not None
    edge_index = data.edge_index
    assert edge_index is not None
    sender = edge_index[:, 0]
    symmetric_displacement = 0.5 * (
        displacement + displacement.transpose(-1, -2)
    )  # From https://github.com/mir-group/nequip
    atom_pos = atom_pos + torch.einsum(
        "be,bec->bc", atom_pos, symmetric_displacement[batch_ids]
    )
    # deal with the case of 2d cell with a single batch
    cell = cell.view(-1, 3, 3)
    cell = cell + torch.matmul(cell, symmetric_displacement)
    shifts = torch.einsum(
        "be,bec->bc",
        unit_shifts,
        cell[batch_ids[sender]],
    )
    return atom_pos, shifts


def virial_to_stress(virial: torch.Tensor, data: Configuration) -> torch.Tensor:
    cell = data.cell
    assert cell is not None
    cell = cell.view(-1, 3, 3)
    volume = torch.linalg.det(cell).abs().unsqueeze(-1)
    stress = virial / volume.view(-1, 1, 1)
    stress = torch.where(torch.abs(stress) < 1e10, stress, torch.zeros_like(stress))
    return -stress


class FrankenPotential(torch.nn.Module):
    """Maps atomic configurations into high-dimensional vectors through a neural network and random features.

    This class provides both functions to map atom configurations with GNN + random features through methods
    :meth:`~franken.rf.model.FrankenPotential.feature_map` and :meth:`~franken.rf.model.FrankenPotential.grad_feature_map`,
    and to perform inference on the potential function of atom configurations given a learned
    linear model (:meth:`~franken.rf.model.FrankenPotential.energy_and_forces`).

    Args:
        gnn_config: Configuration object for the GNN. Sets the ID and other backbone parameters.
        rf_config: Configuration object for the random features (kernel-approximation). Sets all relevant kernel parameters,
            as well as the number of random features.
        jac_chunk_size: Force calculation requires computing Jacobians through the GNN. Since this is
            memory intensive, we can do this in batches across atoms of each configuration. When dealing with large systems,
            the chunk size becomes important to ensure no out-of-memory errors occur. This can either be set manually or be
            set automatically (by passing `"auto"`) which will attempt to determine the largest chunk that fits in memory.
            Defaults to "auto".
        scale_by_Z: Whether features should be scaled indipendently for each different species. Defaults to True.
        num_species: The number of species that this model will be trained on. Defaults to 1.
        atomic_energies: A precomputed dictionary mapping atomic numbers to the average energy of that species.
            Can be safely left to its default of None in most cases.

    Attributes:
        gnn (torch.nn.Module): The graph neural network which is used for feature extraction, loaded from
            a pretrained checkpoint.
        rf (torch.nn.Module): Random-features module :class:`franken.rf.heads.RandomFeaturesHead`.

    Note:
        The automatic Jacobian chunking is known to be error-prone. If you encounter out-of-memory errors
        with this option active, try manually setting the `jac_chunk_size` parameter.
    """

    def __init__(
        self,
        gnn_config: BackboneConfig,
        rf_config: RFConfig,
        jac_chunk_size: Union[int, Literal["auto"]] = "auto",
        scale_by_Z: bool = True,
        num_species: int = 1,
        atomic_energies: Optional[Mapping[int, torch.Tensor | float]] = None,
    ):
        super(FrankenPotential, self).__init__()
        # stored here as simpler to access than through self.gnn
        self.gnn_config = gnn_config
        self.rf_config = rf_config
        self.jac_chunk_size = jac_chunk_size
        self.num_species = num_species

        # Caches for jacobian function
        self.jac_cache: dict[str, Callable] = {}

        # Initialize `gnn`, `rf`, `input_scaler`, `energy_shift` submodules
        self.gnn = load_checkpoint(gnn_config)

        rf_feature_dim = self.gnn.feature_dim()
        self.rf = initialize_rf(rf_config, rf_feature_dim=rf_feature_dim)

        self.input_scaler = FeatureScaler(
            input_dim=self.rf.input_dim,
            statistics=None,
            scale_by_Z=scale_by_Z,
            num_species=num_species,
        )
        self.energy_shift = AtomicEnergiesShift(
            num_species=num_species, atomic_energies=atomic_energies
        )
        self.force_func_grad = False

    @property
    @torch.jit.unused
    def hyperparameters(self):
        hps = {
            "franken": self.gnn_config.to_ckpt(),
            "random_features": self.rf_config.to_ckpt(),
            "input_scaler": self.input_scaler.init_args(),
        }
        return hps

    def save(self, path: os.PathLike | str, multi_weights: torch.Tensor | None = None):
        if multi_weights is not None:
            assert torch.is_tensor(multi_weights)
            assert multi_weights.ndim <= 2
            assert multi_weights.shape[-1] == self.rf.weights.shape[-1]

        ckpt = {
            "jac_chunk_size": self.jac_chunk_size,
            "multi_weights": multi_weights,
            "num_species": self.num_species,
        }

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
        }
        torch.save(ckpt, path)

    @classmethod
    def load(
        cls,
        path,
        map_location=None,
        rf_weight_id: int | None = None,
    ):
        ckpt = torch.load(path, map_location=map_location, weights_only=False)

        rf_cfg = RFConfig.from_ckpt(ckpt["rf"]["config"])
        gnn_cfg = BackboneConfig.from_ckpt(ckpt["gnn"]["config"])
        model = cls(
            gnn_config=gnn_cfg,
            rf_config=rf_cfg,
            jac_chunk_size=ckpt["jac_chunk_size"],
            num_species=ckpt["num_species"],
            **ckpt["input_scaler"]["config"],
        )
        model.rf.load_state_dict(ckpt["rf"]["state_dict"])
        model.input_scaler.load_state_dict(ckpt["input_scaler"]["state_dict"])
        model.energy_shift.load_state_dict(ckpt["energy_shift"])

        if ckpt["multi_weights"] is not None:
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

    @torch.jit.unused
    @torch.no_grad
    def _compute_forces_stresses(
        self,
        data: Configuration,
        fmaps_func: bool,
        weights: torch.Tensor | None,
        cache_key: str,
    ) -> dict[str, torch.Tensor]:
        num_systems = data.natoms.numel()
        displacement = torch.zeros(
            (num_systems, 3, 3),
            dtype=data.atom_pos.dtype,
            device=data.atom_pos.device,
        )
        if fmaps_func:
            args = [data.atom_pos, displacement, data]
        else:
            args = [data.atom_pos, displacement, data, weights]
        if (jacfn := self.jac_cache.get(cache_key)) is None:
            func = self._feature_map_aux if fmaps_func else self._energy_aux
            jac_chunk_size = self.get_jacobian_chunk_size(func, args, argnums=(0, 1))
            jacfn = jacfwd(
                func, argnums=(0, 1), has_aux=True, chunk_size=jac_chunk_size
            )
            self.jac_cache[cache_key] = jacfn
        (force_fm, virial_fm), energy_fm = jacfn(*args)
        stress_fm = virial_to_stress(-virial_fm, data)
        return {
            franken.data.base.FORCES_TARGET_KEY: -force_fm,
            franken.data.base.STRESS_TARGET_KEY: stress_fm,
            franken.data.base.ENERGY_TARGET_KEY: energy_fm,
        }

    @torch.jit.unused
    @torch.no_grad
    def _compute_forces(
        self,
        data: Configuration,
        fmaps_func: bool,
        weights: torch.Tensor | None,
        cache_key: str,
    ) -> dict[str, torch.Tensor]:
        if fmaps_func:
            args = [data.atom_pos, None, data]
        else:
            args = [data.atom_pos, None, data, weights]
        if (jacfn := self.jac_cache.get(cache_key)) is None:
            func = self._feature_map_aux if fmaps_func else self._energy_aux
            jac_chunk_size = self.get_jacobian_chunk_size(func, args, argnums=0)
            jacfn = jacfwd(func, argnums=0, has_aux=True, chunk_size=jac_chunk_size)
            self.jac_cache[cache_key] = jacfn
        force_fm, energy_fm = jacfn(*args)
        return {
            franken.data.base.FORCES_TARGET_KEY: -force_fm,
            franken.data.base.ENERGY_TARGET_KEY: energy_fm,
        }

    def _compute_forces_stresses_ag(
        self, data: Configuration, fmaps_func: bool, weights: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        n_systems = data.natoms.numel()
        displacement = torch.zeros(
            (n_systems, 3, 3),
            dtype=data.atom_pos.dtype,
            device=data.atom_pos.device,
        ).requires_grad_(True)
        data.atom_pos.requires_grad_(True)
        if fmaps_func:
            _, energy = self._feature_map_aux(data.atom_pos, displacement, data)
        else:
            _, energy = self._energy_aux(
                data.atom_pos, displacement, data, weights=weights
            )
        n_sols = energy.shape[0]
        n_atoms = data.atom_pos.shape[0]
        force_lst, virial_lst = [], []
        for i in range(n_sols):  # each model (M) independently
            cur_energy: torch.Tensor = energy[i]
            # NOTE: complex type annotation required by jit.script
            grad_out: List[Optional[torch.Tensor]] = [torch.ones_like(cur_energy)]
            grads = torch.autograd.grad(
                outputs=[cur_energy],
                inputs=[data.atom_pos, displacement],
                grad_outputs=grad_out,  # type: ignore
                retain_graph=i < n_sols - 1,
            )
            g0 = grads[0]
            assert g0 is not None
            force_lst.append(g0)
            g1 = grads[1]
            assert g1 is not None
            virial_lst.append(g1)
        force = -torch.stack(force_lst, 0).view(n_sols, n_atoms, 3)
        virial = -torch.stack(virial_lst, 0).view(n_sols, n_systems, 3, 3)
        stress = virial_to_stress(virial, data)
        return {
            franken.data.base.FORCES_TARGET_KEY: force.detach(),
            franken.data.base.STRESS_TARGET_KEY: stress.detach(),
            franken.data.base.ENERGY_TARGET_KEY: energy.detach(),
        }

    def _compute_forces_ag(
        self, data: Configuration, fmaps_func: bool, weights: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        data.atom_pos.requires_grad_(True)
        if fmaps_func:
            _, energy = self._feature_map_aux(data.atom_pos, None, data)
        else:
            _, energy = self._energy_aux(data.atom_pos, None, data, weights=weights)
        n_sols = energy.shape[0]
        n_atoms = data.atom_pos.shape[0]
        force_lst = []
        for i in range(n_sols):  # each model (M) independently
            cur_energy: torch.Tensor = energy[i]
            # NOTE: complex type annotation required by jit.script
            grad_out: List[Optional[torch.Tensor]] = [torch.ones_like(cur_energy)]
            grads = torch.autograd.grad(
                outputs=[cur_energy],
                inputs=[data.atom_pos],
                grad_outputs=grad_out,  # type: ignore
                retain_graph=i < n_sols - 1,
            )
            grad = grads[0]
            assert grad is not None
            force_lst.append(grad)
        force = -torch.stack(force_lst, 0).view(n_sols, n_atoms, 3)
        return {
            franken.data.base.FORCES_TARGET_KEY: force.detach(),
            franken.data.base.ENERGY_TARGET_KEY: energy.detach(),
        }

    def _feature_map_aux(
        self,
        atom_pos: torch.Tensor,
        displacement: torch.Tensor | None,
        data: Configuration,
    ):
        old_atom_pos = data.atom_pos

        if displacement is not None:
            atom_pos, shifts = prep_with_displacement(data, atom_pos, displacement)
            data.shifts = shifts
        data.atom_pos = atom_pos

        gnn_descriptors = self.gnn.descriptors(data)
        normalized_descriptors = self.input_scaler(
            gnn_descriptors,
            atomic_numbers=data.atomic_numbers,
        )
        random_features = self.rf.feature_map(
            normalized_descriptors,
            atomic_numbers=data.atomic_numbers,
            batch_ids=data.batch_ids,
        )
        data.atom_pos = old_atom_pos
        # the differentiable output is summed over structures
        return random_features.sum(0), random_features

    def grad_feature_map(
        self,
        data: Configuration,
        targets: list[TargetType],
    ) -> dict[TargetType, torch.Tensor]:
        """Compute the feature map for this configuration and
        its gradient with respect to atomic positions

        Returns:
            forces_fmap : Tensor of size [n_random_features, n_atoms * 3]
            energy_fmap : Tensor of size [n_structures, n_random_features]
        """
        compute_force = franken.data.base.FORCES_TARGET_KEY in targets
        compute_stress = franken.data.base.STRESS_TARGET_KEY in targets
        out: dict[TargetType, torch.Tensor]
        if compute_force and compute_stress:
            out = self._compute_forces_stresses(
                data, True, weights=None, cache_key="force_stress_fmap"
            )
        elif compute_force:
            out = self._compute_forces(data, True, weights=None, cache_key="force_fmap")
        else:
            _, energy_fm = self._feature_map_aux(data.atom_pos, None, data)
            out = {franken.data.base.ENERGY_TARGET_KEY: energy_fm}
        # Fix shapes
        if franken.data.base.ENERGY_TARGET_KEY in out:
            # Sys, F -> F, Sys
            out[franken.data.base.ENERGY_TARGET_KEY] = out[
                franken.data.base.ENERGY_TARGET_KEY
            ].transpose(0, 1)
        if franken.data.base.FORCES_TARGET_KEY in out:
            # F, S*A, 3 -> F, S*A*3
            out_f = out[franken.data.base.FORCES_TARGET_KEY]
            out[franken.data.base.FORCES_TARGET_KEY] = out_f.reshape(out_f.shape[0], -1)
        if franken.data.base.STRESS_TARGET_KEY in out:
            # F, S, 3, 3 -> F, S*3*3
            out_s = out[franken.data.base.STRESS_TARGET_KEY]
            out[franken.data.base.STRESS_TARGET_KEY] = out_s.reshape(out_s.shape[0], -1)
        return out

    def _energy_aux(
        self,
        atom_pos: torch.Tensor,
        displacement: torch.Tensor | None,
        data: Configuration,
        weights: torch.Tensor | None,
    ):
        if weights is None:
            weights = self.rf.weights
        # weights: [num weights(M), num features(F)]
        _, feature_map = self._feature_map_aux(atom_pos, displacement, data)
        feature_map = feature_map.to(dtype=weights.dtype)  # [N, F]
        natoms = data.natoms.to(dtype=weights.dtype).view(-1)  # [N]
        energies = torch.matmul(feature_map, weights.T).T  # [M, N]
        energies = natoms[None, :] * energies
        return energies.sum(1), energies

    def _predict(
        self,
        weights: torch.Tensor | None,
        data: Configuration,
        targets: list[str],
        mode: str,  # Literal["torch.func", "torch.autograd"],
    ) -> dict[str, torch.Tensor]:
        """Computes the gradient of the :meth:`~franken.rf.model.FrankenPotential.energy` acting on a configuration.

        The gradient is equivalent to the negative force acting on
        the configuration.

        if `weights` is not provided, the weights stored in the :attr:`FrankenPotential.rf`
        random features object will be used instead.

        This function returns a tuple: `energy_gradient`, `energy`.

        .. note::
            This function uses the :mod:`torch.func` package to compute the gradient,
            which is particularly useful when computing the energy gradient with
            multiple linear models. In this case `weights` can be a matrix whose
            first dimension is the number of linear models.
            When computing the gradient for a single linear model, use the
            :meth:`~franken.rf.model.FrankenPotential.grad_energy_autograd` method instead
            for better performance.

        Args:
            weights (Tensor or None): the linear coefficients to compute the energy
            data: the molecular configuration whose energy and gradient to compute.

        See also :meth:`~franken.rf.model.FrankenPotential.grad_energy_autograd`.

        """
        compute_force = franken.data.base.FORCES_TARGET_KEY in targets
        compute_stress = franken.data.base.STRESS_TARGET_KEY in targets
        if compute_force and compute_stress:
            if mode == "torch.func":
                return self._compute_forces_stresses(
                    data,
                    fmaps_func=False,
                    weights=weights,
                    cache_key="force_stress_energy",
                )
            elif mode == "torch.autograd":
                return self._compute_forces_stresses_ag(data, False, weights)
            else:
                raise ValueError(mode)
        elif compute_force:
            if mode == "torch.func":
                return self._compute_forces(
                    data, fmaps_func=False, weights=weights, cache_key="force_energy"
                )
            elif mode == "torch.autograd":
                return self._compute_forces_ag(data, False, weights)
            else:
                raise ValueError(mode)
        else:
            _, energy = self._energy_aux(data.atom_pos, None, data, weights)  # [M, N]
            return {franken.data.base.ENERGY_TARGET_KEY: energy}

    def get_jacobian_chunk_size(
        self, func, func_inputs, argnums: int | tuple[int, int] = 0
    ) -> int:
        if hasattr(self, "_auto_jac_chunk_size"):
            return self._auto_jac_chunk_size
        else:
            jac_chunk_size = self.jac_chunk_size
            if jac_chunk_size == "auto":
                # TODO: We can probably cache the value for different functions to avoid multiple tuner runs.
                jac_chunk_size = tune_jacfwd_chunksize(
                    test_sample=func_inputs,
                    func=func,
                    argnums=argnums,
                    has_aux=True,
                )
                self._auto_jac_chunk_size = jac_chunk_size
                logger.info(
                    f"jacobian chunk size automatically set to {self._auto_jac_chunk_size}"
                )
            assert isinstance(jac_chunk_size, int)
            return jac_chunk_size

    def predict(
        self,
        targets: list[str],
        data: Configuration,
        weights: torch.Tensor | None = None,
        forces_mode: str = "torch.autograd",  # Literal["torch.autograd", "torch.func"] = "torch.autograd",
        add_energy_shift: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Infer energy and forces of an atomic configuration using a learned random-features model.

        The parameter `weights` can be used to specified the model's weights. Otherwise the weights stored in
        :attr:`FrankenPotential.rf.weights` will be used instead.

        The different values of `forces_mode` correspond to different ways of differentiating
        through the model to obtain the forces acting on the atoms:

        * :code:`"torch.func"` is best for when `weights` contains multiple linear models on which to perform inference at the same time (in that case `weights` should be a matrix of shape `[n_linear_models, model_size]`).

        * :code:`"torch.autograd"` is best for when a single linear model is used (i.e. when `weights` is a vector of shape `[model_size]`)

        * :code:`"no_forces"` can be used if forces are not required.

        Args:
            weights: weights of the random feature model. Defaults to None, in which case the weights set in :attr:`FrankenPotential.rf` will be used instead.

            forces_mode: how to compute the model's forces. Defaults to :code:`"torch.autograd"`.

            add_energy_shift: whether to add the energy shift to the energy.

        Returns:
            A tuple containing a tensor representing the potential energy (this is scalar, unless doing inference with multiple
            models when it can be vector-valued), and another optional tensor representing the forces acting on each atom of the
            given configuration. If multiple models are given, the forces will have shape :code:`[n_linear_models, n_atoms, 3]`,
            otherwise they will have shape :code:`[n_atoms, 3]`.
        """
        natoms = torch.atleast_1d(data.natoms)
        out = self._predict(weights, data, targets, forces_mode)

        if add_energy_shift and franken.data.base.ENERGY_TARGET_KEY in targets:
            out[franken.data.base.ENERGY_TARGET_KEY] = out[
                franken.data.base.ENERGY_TARGET_KEY
            ] + self.energy_shift(
                data.atomic_numbers,
                batch_ids=data.batch_ids,
                num_systems=int(natoms.numel()),
            )
        return out  # ([M, N], [M, A, 3])

    def predict_from_fmaps(
        self,
        data: Configuration,
        fmaps: dict[str, torch.Tensor],
        weights: torch.Tensor | None = None,
        add_energy_shift: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Compute energies and forces from pre-computed featuremaps.
        This function does not require calling the underlying GNN.

        Args:
            data : Configuration object describing one or more atomic structures
            energy_fmap : Tensor of size [n_structures, num_random_features] containing the
                original feature map
            forces_fmap : Tensor of size [num_random_features, num_atoms * 3] containing the
                gradients of the original feature map
            weights: Optional tensor of size [num_models, num_random_features]. This contains
                the weights of a trained RF model. If no weights are provided, the weights
                contained in the RF model attached to this class will be used. Energies and forces
                can be computed for multiple models simulatenously by passing in multiple weight
                vectors (arranged in 2D).
        Returns:
            energies : Tensor of size [n_models(M), n_systems(S)]
            forces   : Tensor of size [n_models(M), n_atoms(A), 3]
        """
        if weights is None:
            weights = self.rf.weights
        natoms = torch.atleast_1d(data.natoms)  # [S]
        assert weights.ndim == 2, f"Weights dimensions {weights.shape}"
        for tt, fmap in fmaps.items():
            assert fmap.ndim == 2, f"{tt} map dimensions {fmap.shape}"

        out = {}
        for tt, fmap in fmaps.items():
            out_tt = torch.matmul(weights, fmap)  # [M, S*X]
            if (
                tt == franken.data.base.ENERGY_TARGET_KEY
            ):  # fmap: [F, S], out_tt: [M, S]
                out_tt = natoms[None, :] * out_tt  # [M, S]
            elif (
                tt == franken.data.base.FORCES_TARGET_KEY
            ):  # fmap: [F, S*A*3], out_tt: [M, S*A*3]
                out_tt = out_tt.reshape(out_tt.shape[0], -1, 3)  # [M, S*A, 3]
                if data.batch_ids is None:
                    out_tt = out_tt * natoms
                else:
                    natoms_mul = torch.gather(natoms, 0, data.batch_ids)  # [S*A]
                    out_tt = out_tt * natoms_mul[None, :, None]
            elif (
                tt == franken.data.base.STRESS_TARGET_KEY
            ):  # fmap: [F, S*3*3], out_tt: [M, S*3*3]
                out_tt = out_tt.reshape(out_tt.shape[0], -1, 3, 3)  # [M, S, 3, 3]
                out_tt = natoms[None, :, None, None] * out_tt
            else:
                raise NotImplementedError(tt)
            out[tt] = out_tt

        if add_energy_shift and franken.data.base.ENERGY_TARGET_KEY in fmaps.keys():
            out[franken.data.base.ENERGY_TARGET_KEY] = out[
                franken.data.base.ENERGY_TARGET_KEY
            ] + self.energy_shift(
                data.atomic_numbers,
                batch_ids=data.batch_ids,
                num_systems=int(natoms.numel()),
            )
        return out

    def forward(
        self,
        targets: list[str],
        data: Configuration,
        weights: torch.Tensor | None = None,
        add_energy_shift: bool = True,
    ) -> dict[str, torch.Tensor]:
        """
        See docstring of :meth:`~franken.rf.model.FrankenPotential.energy_and_forces`.

        This function defaults to using the 'torch.autograd' strategy which allows the model
        to be jit-compiled.
        """
        out = self.predict(
            targets=targets,
            data=data,
            weights=weights,
            forces_mode="torch.autograd" if not self.force_func_grad else "torch.func",
            add_energy_shift=add_energy_shift,
        )
        return {k: v.squeeze(0) for k, v in out.items()}
