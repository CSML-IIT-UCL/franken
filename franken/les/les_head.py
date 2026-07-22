import dataclasses

import torch
import torch.nn as nn

from franken.config import LESConfig
import franken.data.base
from franken.data.base import Configuration
from franken.utils.derivatives import forces_autograd, forces_stress_autograd
from franken.utils.misc import sanitize_init_dict
from les.ewald import Ewald


def initialize_les(les_config: LESConfig, feature_dim: int):
    random_features_params = sanitize_init_dict(LESHead, dataclasses.asdict(les_config))
    return LESHead(input_dim=feature_dim, **random_features_params)


class LESHead(nn.Module):
    """
    Predicts atomic charges from GNN embeddings and computes
    electrostatic energy with LES.

    Output:
        total_energy : (1,)
    """

    def __init__(
        self,
        input_dim,
        hidden_dim=128,
        dl=2.0,
        sigma=1.0,
    ):
        super().__init__()

        self.charge_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

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
            forces, stress = forces_stress_autograd(energies, displacement, data)
            return {
                franken.data.base.FORCES_TARGET_KEY: forces.detach(),
                franken.data.base.STRESS_TARGET_KEY: stress.detach(),
                franken.data.base.ENERGY_TARGET_KEY: energies.detach(),
            }
        elif compute_force:
            forces, stress = forces_autograd(energies, data)
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
        q = self.charge_net(atom_features).squeeze(-1)
        ewald_out = self.ewald(
            q=q,
            r=configuration.atom_pos,
            cell=configuration.cell,
            batch=configuration.batch_ids,
        )
        return ewald_out["pot"]
