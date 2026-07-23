import dataclasses

import torch
import torch.nn as nn

from franken.config import LESConfig
import franken.data.base
from franken.data.base import Configuration
from franken.utils.derivatives import _forces_bwdad_helper, _forces_stress_bwdad_helper
from franken.utils.misc import sanitize_init_dict
from franken.les.ewald import Ewald


def initialize_les(les_config: LESConfig, feature_dim: int):
    les_params = sanitize_init_dict(LESHead, dataclasses.asdict(les_config))
    return LESHead(input_dim=feature_dim, **les_params)


class LESHead(nn.Module):
    """
    Predicts atomic charges from GNN embeddings and computes
    electrostatic energy with LES.

    Output:
        total_energy : (1,)
    """

    def __init__(
        self,
        input_dim: int,
        n_layers: int = 3,
        hidden_dim: tuple[int, ...] = (32, 16),
        dl: float = 2.0,
        sigma=1.0,
        les_output_scale: float = 0.1,
        add_linear_nn: bool = True,
    ):
        super().__init__()
        self.les_output_scale = les_output_scale

        # Build the MLP
        if isinstance(hidden_dim, int):
            hidden_dim = (hidden_dim,)
        if len(hidden_dim) == 1:
            n_hidden = [hidden_dim[0]] * (n_layers - 1)
        elif len(hidden_dim) != n_layers - 1:
            raise ValueError(
                f"Expected {n_layers - 1} dimensions for the MLP. Found dimensions {hidden_dim}."
            )
        else:
            n_hidden = list(hidden_dim)
        n_neurons = [input_dim] + n_hidden + [1]
        layers = []
        for i in range(n_layers - 1):
            layers.append(nn.Linear(n_neurons[i], n_neurons[i + 1]))
            layers.append(nn.SiLU(inplace=True))
        layers.append(nn.Linear(n_neurons[-2], n_neurons[-1]))
        self.outnet = nn.Sequential(*layers)
        self.linear_nn = None
        if add_linear_nn:
            self.linear_nn = nn.Linear(input_dim, 1)

        self.ewald = Ewald(
            dl=dl,
            sigma=sigma,
        )

    def predict(
        self,
        atom_features: torch.Tensor,
        data: Configuration,
        displacement: torch.Tensor | None,
        targets: list[str],
    ):
        compute_force = franken.data.base.FORCES_TARGET_KEY in targets
        compute_stress = franken.data.base.STRESS_TARGET_KEY in targets
        energies = self(atom_features, data)
        if compute_stress:
            assert displacement is not None
            forces, stress = _forces_stress_bwdad_helper(energies, displacement, data)
            return {
                franken.data.base.FORCES_TARGET_KEY: forces.detach(),
                franken.data.base.STRESS_TARGET_KEY: stress.detach(),
                franken.data.base.ENERGY_TARGET_KEY: energies.detach(),
            }
        elif compute_force:
            forces, stress = _forces_bwdad_helper(energies, data)
            return {
                franken.data.base.FORCES_TARGET_KEY: forces.detach(),
                franken.data.base.ENERGY_TARGET_KEY: energies.detach(),
            }
        else:
            return {
                franken.data.base.ENERGY_TARGET_KEY: energies.detach(),
            }

    def forward(
        self,
        atom_features: torch.Tensor,
        configuration: Configuration,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        atom_features : Tensor
            Tensor of atomic features obtained through a GNN. Should have shape
            `[n_atoms, hidden_dim]`.
        configuration : Configuration

        Returns
        -------
        energy : tensor shape (1,)
        """
        # predict atomwise contributions
        y = self.outnet(atom_features)
        if self.linear_nn is not None:
            y = y + self.linear_nn(atom_features)
        y = y * self.les_output_scale

        E_lr, q_induced, u_induced = self.ewald(
            q=y,
            r=configuration.atom_pos,
            cell=configuration.cell,
            batch=configuration.batch_ids,
        )
        return E_lr
