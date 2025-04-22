from pathlib import Path
from typing import Literal, Union

import numpy as np
import torch
from ase.calculators.calculator import Calculator, all_changes

from franken.data import BaseAtomsDataset, Configuration
from franken.rf.model import FrankenPotential
from franken.utils.misc import get_device_name


class FrankenCalculator(Calculator):
    """Calculator for ASE with franken models

    Attributes:
        implemented_properties:
            Lists properties which can be asked from this calculator, notably "energy" and "forces".
    """

    implemented_properties = ["energy", "forces"]
    default_parameters = {}
    nolabel = True  # ??

    def __init__(
        self,
        franken_ckpt: Union[FrankenPotential, str, Path],
        device=None,
        rf_weight_id: int | None = None,
        forces_mode: Literal["torch.func", "torch.autograd"] = "torch.autograd",
        **calc_kwargs,
    ):
        """Initialize FrankenCalculator class from a franken model.

        Args:
            franken_ckpt : Path to the franken model.
                This class accepts pre-loaded models, as well as jitted models (with `torch.jit`).
            device : PyTorch device specification for where the model should reside
                (e.g. "cuda:0" for GPU placement or "cpu" for CPU placement).
            rf_weight_id : ID of the random feature weights.
                Can generally be left to ``None`` unless the checkpoint contains multiple trained models.
        """
        # TODO: Remove forces_mode, torch.autograd is always the right way.
        super().__init__(**calc_kwargs)
        self.franken: FrankenPotential
        if isinstance(franken_ckpt, torch.nn.Module):
            self.franken = franken_ckpt
            if device is not None:
                self.franken.to(device)
        else:
            # Handle jitted torchscript archives and normal files
            try:
                self.franken = torch.jit.load(franken_ckpt, map_location=device)
            except RuntimeError as e:
                if "PytorchStreamReader" not in str(e):
                    raise
                self.franken = FrankenPotential.load(  # type: ignore
                    franken_ckpt,
                    map_location=device,
                    rf_weight_id=rf_weight_id,
                )

        self.dataset = BaseAtomsDataset.from_path(
            data_path=None,
            split="md",
            gnn_config=self.franken.gnn_config,
        )
        self.device = (
            device if device is not None else next(self.franken.parameters()).device
        )
        self.forces_mode = forces_mode

    def calculate(
        self,
        atoms=None,
        properties=None,
        system_changes=all_changes,
    ):
        if properties is None:
            properties = self.implemented_properties
        if "forces" not in properties:
            forces_mode = "no_forces"
        else:
            forces_mode = self.forces_mode

        super().calculate(atoms, properties, system_changes)

        # self.atoms is set in the super() call. Unclear why it should be preferred over `atoms`
        config_idx = self.dataset.add_configuration(self.atoms)  # type: ignore
        cpu_data = self.dataset.__getitem__(config_idx, no_targets=True)
        assert isinstance(cpu_data, Configuration)
        data = cpu_data.to(self.device)

        energy, forces = self.franken.energy_and_forces(data, forces_mode=forces_mode)

        if energy.ndim == 0:
            self.results["energy"] = energy.item()
        else:
            self.results["energy"] = np.squeeze(energy.numpy(force=True))
        if "forces" in properties:
            assert forces is not None
            self.results["forces"] = np.squeeze(forces.numpy(force=True))


def calculator_throughput(
    calculator, atoms_list, num_repetitions=1, warmup_configs=5, verbose=True
):
    from time import perf_counter

    hardware = get_device_name(calculator.device)

    _atom_numbers = set(len(atoms) for atoms in atoms_list)
    assert (
        len(_atom_numbers) == 1
    ), f"This function only accepts configurations with the same number of atoms, while found configurations with {_atom_numbers} number of atoms"
    natoms = _atom_numbers.pop()

    assert len(atoms_list) > warmup_configs
    for idx in range(warmup_configs):
        calculator.calculate(atoms_list[idx])
    time_init = perf_counter()
    for _ in range(num_repetitions):
        for atoms in atoms_list:
            calculator.calculate(atoms)
    time = perf_counter() - time_init
    configs_per_sec = (len(atoms_list) * num_repetitions) / time
    results = {
        "throughput": configs_per_sec,
        "atoms": natoms,
        "hardware": hardware,
    }
    if verbose:
        print(
            f"{results['throughput']:.1f} cfgs/sec ({results['atoms']} atoms) | {results['hardware']}"
        )
    return results
