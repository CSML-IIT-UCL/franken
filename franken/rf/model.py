"""Franken model"""

import logging
import os
from typing import Literal, Mapping, Optional, Tuple, Union

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
        gnn_backbone_id (str):
        kernel_type (str): The kernel type to use. Must be one of :code:`gaussian`, :code:`poly`, or :code:`linear`.
        rf_config (dict): A dictionary of parameters for the random features head. For each kernel type the default parameters can be found :ref:`below <random_features_params>`.
        interaction_block (int): GNN layer used for feature extraction
        jac_chunk_size (int or "auto"):

    Attributes:
        gnn (torch.nn.Module): The graph neural network which is used for feature extraction, loaded from
            a pretrained checkpoint.
        rf (torch.nn.Module): Random-features module :class:`franken.rf.heads.RandomFeaturesHead`.
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

    def feature_map(self, data: Configuration):
        """Obtain an embedding of each atom, and map it through
        random features. The RF mapping computes an average so
        the final feature map is per-configuration, instead of
        per-atom.
        """
        gnn_descriptors = self.gnn.descriptors(data)

        normalized_descriptors = self.input_scaler(
            gnn_descriptors,
            atomic_numbers=data.atomic_numbers,
        )
        return self.rf.feature_map(
            normalized_descriptors,
            atomic_numbers=data.atomic_numbers,
        )

    def _feature_map_aux(self, atom_pos: torch.Tensor, data: Configuration):
        old_atom_pos = data.atom_pos
        data.atom_pos = atom_pos
        random_features = self.feature_map(data)
        data.atom_pos = old_atom_pos
        return random_features, random_features

    def grad_feature_map(
        self, data: Configuration
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the feature map for this configuration and
        its gradient with respect to atomic positions

        Returns:
         - forces_fmap : Tensor of size [n_random_features, n_atoms * 3]
         - energy_fmap : Tensor of size [n_random_features]
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

        if :attr:`weights` is not provided, the weights stored in the :attr:`self.rf`
        random features object will be used instead.

        This function returns the energy of the configuration scaled by the number
        of atoms in the configuration itself

        .. math::
            \text{out} = \text{num atoms} \times E(\text{configuration})

        Args:
            weights: The linear coefficients of the energy model.
            configuration: The molecular configuration whose energy to compute.
        """
        if weights is None:
            weights = self.rf.weights

        feature_map = self.feature_map(data).to(dtype=weights.dtype)

        # Contract the last dim of weights with the first of feature_map
        energy = data.natoms * torch.tensordot(weights, feature_map, dims=1)

        return energy

    def _energy_aux(
        self,
        weights: Optional[torch.Tensor],
        atom_pos: torch.Tensor,
        data: Configuration,
    ):
        old_atom_pos = data.atom_pos
        data.atom_pos = atom_pos
        energy = self.energy(weights, data)
        data.atom_pos = old_atom_pos
        return energy, energy

    def grad_energy_func(
        self, weights: Optional[torch.Tensor], data: Configuration
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes the gradient of the :meth:`~franken.rf.model.FrankenPotential.energy` acting on a configuration.

        The gradient is equivalent to the negative force acting on
        the configuration.

        if :attr:`weights` is not provided, the weights stored in the :attr:`self.rf`
        random features object will be used instead.

        This function returns a tuple: :attr:`energy_gradient`, :attr:`energy`.

        .. note::
            This function uses the :mod:`torch.func` package to compute the gradient,
            which is particularly useful when computing the energy gradient with
            multiple linear models. In this case :attr:`weights` can be a matrix whose
            first dimension is the number of linear models.
            When computing the gradient for a single linear model, use the
            :meth:`~franken.rf.model.FrankenPotential.grad_energy_autograd` method instead
            for better performance.

        Args:
            weights (Tensor or None): the linear coefficients to compute the energy
            configuration: the molecular configuration whose energy and gradient to compute.

        See also :meth:`~franken.rf.model.FrankenPotential.grad_energy_autograd`.

        """
        if self._grad_energy_jacfn is None:
            jac_chunk_size = self.get_jacobian_chunk_size(
                self._energy_aux, [weights, data.atom_pos, data], argnums=1
            )
            self._grad_energy_jacfn = jacfwd(
                self._energy_aux, argnums=1, has_aux=True, chunk_size=jac_chunk_size
            )
        out = self._grad_energy_jacfn(weights, data.atom_pos, data)
        return out

    def grad_energy_autograd(
        self, weights: Optional[torch.Tensor], data: Configuration
    ) -> Tuple[torch.Tensor | None, torch.Tensor]:
        """Computes the gradient of the :meth:`~franken.rf.model.FrankenPotential.energy` acting on a configuration.

        The gradient is equivalent to the negative force acting on
        the configuration.

        if :attr:`weights` is not provided, the weights stored in the :attr:`self.rf`
        random features object will be used instead.

        This function returns a tuple: :attr:`energy_gradient`, :attr:`energy`. See
        :meth:`~franken.rf.model.FrankenPotential.grad_energy_func` for a discussion on the
        performance characteristics of the two implementations.

        Args:
            weights (Tensor or None): the linear coefficients to compute the energy
            configuration: the molecular configuration whose energy and gradient to compute.
        """
        # Ensure atom positions require gradients
        data.atom_pos.requires_grad_(True)
        # Compute the energy
        energy = self.energy(weights, data)

        if energy.ndim == 0:
            # Scalar case, single gradient
            gradient = torch.autograd.grad(outputs=[energy], inputs=[data.atom_pos])[0]
        else:
            # Vector case, compute gradients for each element independently
            gradients: list[torch.Tensor] = []
            for i in range(energy.shape[0]):
                grad_i = torch.autograd.grad(
                    outputs=[energy[i]],
                    inputs=[data.atom_pos],
                    retain_graph=True,
                )[0]
                assert grad_i is not None
                gradients.append(grad_i)
            # Stack gradients along a new dimension
            gradient = torch.stack(gradients, dim=0)

        return gradient, energy

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

        The parameter :attr:`weights` can be used to specified the model's weights. Otherwise the weights stored in
        :attr:`self.rf.weights` will be used instead.

        The different values of :attr:`forces_mode` correspond to different ways of differentiating
        through the model to obtain the forces acting on the atoms:

        * :code:`"torch.func"` is best for when :attr:`weights` contains multiple linear models on which to perform inference at the same time (in that case :attr:`weights` should be a matrix of shape `[n_linear_models, model_size]`).

        * :code:`"torch.autograd"` is best for when a single linear model is used (i.e. when :attr:`weights` is a vector of shape `[model_size]`)

        * :code:`"no_forces"` can be used if forces are not required.

        Args:
            weights: weights of the random feature model. Defaults to None, in which case the weights set in :attr:`self.rf` will be used instead.

            forces_mode: how to compute the model's forces. Defaults to :code:`"torch.autograd"`.

            add_energy_shift: whether to add the energy shift to the energy.

        Returns:
            energy (Tensor): the scalar (or vector-valued when doing inference with multiple models) valued potential energy.

            forces (Tensor): the forces acting on each atom of the given configuration. If multiple models are given, this will have shape :code:`[n_linear_models, n_atoms, 3]`.
        """
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
            energy = energy + self.energy_shift(data.atomic_numbers)
        return energy.detach(), forces

    def energy_and_forces_from_fmaps(
        self,
        data: Configuration,
        energy_fmap: torch.Tensor,
        forces_fmap: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        add_energy_shift: bool = True,
    ):
        if weights is None:
            weights = self.rf.weights
        energy = torch.tensordot(
            weights,
            data.natoms * energy_fmap,
            dims=([1], [0]),  # type: ignore
        )
        forces = torch.tensordot(
            weights,
            data.natoms * forces_fmap.view(forces_fmap.shape[0], -1, 3),
            dims=([1], [0]),  # type: ignore
        )
        if add_energy_shift:
            energy = energy + self.energy_shift(data.atomic_numbers)
        return energy, forces

    def forward(
        self,
        data: Configuration,
        weights: Optional[torch.Tensor] = None,
        add_energy_shift: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        See docstring of :meth:`~franken.rf.model.FrankenPotential.energy_and_forces`.

        This function defaults to using the 'torch.autograd' strategy which allows the model
        to be jit-compiled.
        """
        grad_energy, energy = self.grad_energy_autograd(weights, data)
        assert grad_energy is not None
        forces = -grad_energy
        if add_energy_shift:
            energy = energy + self.energy_shift(data.atomic_numbers)
        return energy.detach(), forces.detach()
