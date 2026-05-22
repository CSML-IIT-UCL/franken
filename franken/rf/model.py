"""Franken model"""

import logging
import os
from typing import List, Literal, Mapping, Optional, Tuple, Union

import torch

from franken.config import (
    BackboneConfig,
    RFConfig,
)
from franken.backbones.utils import load_checkpoint
from franken.data import Configuration
from franken.rf.atomic_energies import AtomicEnergiesShift
from franken.rf.heads import initialize_rf
from franken.rf.scaler import FeatureScaler
from franken.utils.jac import jacfwd, tune_jacfwd_chunksize

logger = logging.getLogger("franken")


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
        self._grad_energy_jacfn = None
        self._grad_fmap_jacfn = None

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
        backbone_path_or_id: str | None = None,
    ):
        ckpt = torch.load(path, map_location=map_location, weights_only=False)

        rf_cfg = RFConfig.from_ckpt(ckpt["rf"]["config"])
        gnn_cfg = BackboneConfig.from_ckpt(ckpt["gnn"]["config"])
        if backbone_path_or_id is not None:
            logger.warning(
                f"The backbone path/id changed from {gnn_cfg.path_or_id} to {backbone_path_or_id}. If this refers to a different backbone, unexpected results may occur."
            )
            gnn_cfg.path_or_id = backbone_path_or_id
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

    def feature_map(self, data: Configuration):
        """Obtain an embedding of each atom, and map it through random features.

        The RF mapping computes an average so the final feature map is
        per-structure, instead of per-atom. In case data contains multiple
        structures, multiple feature maps are computed.

        Returns:
            feature_map: Tensor of size [n_structures, n_random_features]
        """
        gnn_descriptors = self.gnn.descriptors(data)

        normalized_descriptors = self.input_scaler(
            gnn_descriptors,
            atomic_numbers=data.atomic_numbers,
        )
        return self.rf.feature_map(
            normalized_descriptors,
            atomic_numbers=data.atomic_numbers,
            batch_ids=data.batch_ids,
        )

    def _feature_map_aux(self, atom_pos: torch.Tensor, data: Configuration):
        old_atom_pos = data.atom_pos
        data.atom_pos = atom_pos
        random_features = self.feature_map(data)
        data.atom_pos = old_atom_pos
        # the differentiable output is summed over structures
        return random_features.sum(0), random_features

    def grad_feature_map(
        self, data: Configuration
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the feature map for this configuration and
        its gradient with respect to atomic positions

        Returns:
            forces_fmap : Tensor of size [n_random_features, n_atoms * 3]
            energy_fmap : Tensor of size [n_structures, n_random_features]
        """
        if self._grad_fmap_jacfn is None:
            jac_chunk_size = self.get_jacobian_chunk_size(
                self._feature_map_aux, [data.atom_pos, data], argnums=0
            )
            self._grad_fmap_jacfn = jacfwd(
                self._feature_map_aux,
                argnums=0,
                has_aux=True,
                chunk_size=jac_chunk_size,
            )
        return self._grad_fmap_jacfn(data.atom_pos, data)

    def energy(
        self, weights: Optional[torch.Tensor], data: Configuration
    ) -> torch.Tensor:
        r"""Computes the energy of a configuration, using a linear model.

        if `weights` is not provided, the weights stored in the :attr:`FrankenPotential.rf`
        random features object will be used instead.

        This function returns the energy of the configuration scaled by the number
        of atoms in the configuration itself

        .. math::
            \text{out} = \text{num atoms} \times E(\text{configuration})

        Args:
            weights: The linear coefficients of the energy model.
            configuration: The molecular configuration whose energy to compute.
        Returns:
            energies : A tensor of size [n_models, n_structures].
                ``n_models`` denotes the number of models present in the ``weights`` attribute;
                ``n_structures`` the number of different structures present in the input data.
        """
        if weights is None:
            weights = self.rf.weights
        # weights: [num weights(M), num features(F)]
        feature_map = self.feature_map(data).to(dtype=weights.dtype)  # [N, F]
        natoms = data.natoms.to(dtype=weights.dtype).view(-1)  # [N]
        energies = torch.matmul(feature_map, weights.T).T  # [M, N]
        energies = natoms[None, :] * energies
        return energies

    def _energy_aux(
        self,
        weights: Optional[torch.Tensor],
        atom_pos: torch.Tensor,
        data: Configuration,
    ):
        old_atom_pos = data.atom_pos
        data.atom_pos = atom_pos
        energy = self.energy(weights, data)  # [M, N]
        data.atom_pos = old_atom_pos
        return energy.sum(1), energy

    def grad_energy_func(
        self, weights: Optional[torch.Tensor], data: Configuration
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
        if self._grad_energy_jacfn is None:
            jac_chunk_size = self.get_jacobian_chunk_size(
                self._energy_aux, [weights, data.atom_pos, data], argnums=1
            )
            self._grad_energy_jacfn = jacfwd(
                self._energy_aux, argnums=1, has_aux=True, chunk_size=jac_chunk_size
            )
        out = self._grad_energy_jacfn(
            weights, data.atom_pos, data
        )  # ([M, A, 3], [M, N])
        return out

    def grad_energy_autograd(
        self, weights: Optional[torch.Tensor], data: Configuration
    ) -> Tuple[torch.Tensor | None, torch.Tensor]:
        """Computes the gradient of the :meth:`~franken.rf.model.FrankenPotential.energy` acting on a configuration.

        The gradient is equivalent to the negative force acting on
        the configuration.

        if `weights` is not provided, the weights stored in the :attr:`FrankenPotential.rf`
        random features object will be used instead.

        This function returns a tuple: `energy_gradient`, `energy`. See
        :meth:`~franken.rf.model.FrankenPotential.grad_energy_func` for a discussion on the
        performance characteristics of the two implementations.

        Args:
            weights (Tensor or None): the linear coefficients to compute the energy
            data: the molecular configuration whose energy and gradient to compute.
        """
        # Ensure atom positions require gradients
        data.atom_pos.requires_grad_(True)
        # Compute the energy
        energy = self.energy(weights, data)  # [M, N]
        n_sols, n_sys = energy.shape
        n_atoms = data.atom_pos.shape[0]
        # Compute energy gradients for each model (M) independently.
        gradients: List[torch.Tensor] = []
        for i in range(n_sols):
            cur_energy: torch.Tensor = energy[i]
            # complex type annotation required by the obsolete jit.script system
            grad_out: List[Optional[torch.Tensor]] = [torch.ones_like(cur_energy)]
            grad_i = torch.autograd.grad(
                outputs=[cur_energy],
                inputs=[data.atom_pos],
                grad_outputs=grad_out,
                retain_graph=i < n_sols - 1,
            )[0]
            assert grad_i is not None
            gradients.append(grad_i)
        # Stack gradients along a new dimension
        gradient = torch.stack(gradients, dim=0)
        gradient = gradient.view(n_sols, n_atoms, 3)
        return gradient, energy  # ([M, A, 3], [M, N])

    def get_jacobian_chunk_size(self, func, func_inputs, argnums=0) -> int:
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

    def energy_and_forces(
        self,
        data: Configuration,
        weights: Optional[torch.Tensor] = None,
        forces_mode: str = "torch.autograd",
        add_energy_shift: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
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
        if forces_mode == "torch.func":
            with torch.no_grad():
                grad_energy, energy = self.grad_energy_func(weights, data)
                forces = -grad_energy.detach()
        elif forces_mode == "torch.autograd":
            grad_energy, energy = self.grad_energy_autograd(weights, data)
            assert grad_energy is not None
            forces = -grad_energy.detach()
        elif forces_mode == "no_forces":
            energy = self.energy(weights, data)
            forces = None
        else:
            raise ValueError(f"forces_mode '{forces_mode}' is not valid.")

        if add_energy_shift:
            energy = energy + self.energy_shift(
                data.atomic_numbers,
                batch_ids=data.batch_ids,
                num_systems=int(natoms.numel()),
            )
        return energy, forces  # ([M, N], [M, A, 3])

    def energy_and_forces_from_fmaps(
        self,
        data: Configuration,
        energy_fmap: torch.Tensor,
        forces_fmap: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        add_energy_shift: bool = True,
    ):
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
            energies : Tensor of size [n_models, n_structures]
            forces   : Tensor of size [n_models, n_atoms, 3]
        """
        if weights is None:
            weights = self.rf.weights
        natoms = torch.atleast_1d(data.natoms)
        # Consistency checks
        assert energy_fmap.ndim == 2, f"Energy map dimensions {energy_fmap.shape}"
        assert forces_fmap.ndim == 2, f"Forces map dimensions {forces_fmap.shape}"
        assert weights.ndim == 2, f"Weights dimensions {weights.shape}"
        assert (
            energy_fmap.shape[1] == forces_fmap.shape[0]
        ), f"{forces_fmap.shape=} {energy_fmap.shape=}"
        assert (
            weights.shape[1] == forces_fmap.shape[0]
        ), f"{weights.shape=} {forces_fmap.shape=}"

        energies = natoms[None, :] * torch.matmul(energy_fmap, weights.T).T  # [M, N]
        forces = torch.matmul(weights, forces_fmap)  # [M, A*3]
        forces = forces.view(forces.shape[0], -1, 3)  # [M, A, 3]
        if data.batch_ids is None:
            forces = forces * natoms
        else:
            natoms_mul = torch.gather(natoms, 0, data.batch_ids)
            forces = forces * natoms_mul[None, :, None]

        if add_energy_shift:
            energies = energies + self.energy_shift(
                data.atomic_numbers,
                batch_ids=data.batch_ids,
                num_systems=int(natoms.numel()),
            )
        return energies, forces

    def forward(
        self,
        data: Configuration,
        weights: Optional[torch.Tensor] = None,
        add_energy_shift: bool = True,
        compute_forces: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        See docstring of :meth:`~franken.rf.model.FrankenPotential.energy_and_forces`.

        This function defaults to using the 'torch.autograd' strategy which allows the model
        to be jit-compiled.
        """
        if not compute_forces:
            energy = self.energy(weights, data)
            forces = None
        else:
            grad_energy, energy = self.grad_energy_autograd(weights, data)
            assert grad_energy is not None
            energy = energy.detach()
            forces = -grad_energy.detach()
        if add_energy_shift:
            natoms = torch.atleast_1d(data.natoms)
            energy = energy + self.energy_shift(
                data.atomic_numbers,
                batch_ids=data.batch_ids,
                num_systems=int(natoms.numel()),
            )
        return energy, forces
