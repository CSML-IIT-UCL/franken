from pathlib import Path
from typing import Union

import torch
from ase.calculators.calculator import Calculator, all_changes

from franken.config import BackboneConfig
from franken.data import FrankenAtomsDataset, Configuration
from franken.data.base import (
    ENERGY_TARGET_KEY,
    FORCES_TARGET_KEY,
    STRESS_TARGET_KEY,
)
from franken.rf.model import FrankenPotential
from franken.utils.misc import get_device_name


class FrankenCalculator(Calculator):
    """Calculator for ASE with franken models

    Attributes:
        implemented_properties:
            Lists properties which can be asked from this calculator, notably "energy", "forces", "stress".
    """

    implemented_properties = ["energy", "forces", "stress"]
    default_parameters = {}
    nolabel = True  # ??

    def __init__(
        self,
        franken_ckpt: Union[FrankenPotential, str, Path],
        device=None,
        rf_weight_id: int | None = None,
        gnn_config: BackboneConfig | None = None,
        dtype: torch.dtype | str | None = None,
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
            gnn_config : Configuration object for the backbone implemented by the Franken checkpoint.
                Normal behavior is to automatically detect the GNN from the loaded franken model.
                This may not be always possible, in particular if the provided model is JIT scripted.
                In those cases passing the correct `gnn_config` is needed.
            dtype : Inference precision, either ``"float32"|torch.float32`` or ``"float64"|torch.float64``.
                ``None`` preserves the model's precision and uses the backbone's
                precision for inputs.
        """
        if isinstance(dtype, str):
            dtype = {"float32": torch.float32, "float64": torch.float64}.get(dtype, dtype)
        if dtype is not None and dtype not in (torch.float32, torch.float64):
            raise ValueError(
                'dtype must be None, torch.float32, torch.float64, "float32", or "float64"'
            )
        super().__init__(**calc_kwargs)
        self.franken: FrankenPotential
        if isinstance(franken_ckpt, torch.nn.Module):
            self.franken = franken_ckpt
            if device is not None:
                self.franken = self.franken.to(device)
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
        # convert franken model to the requested dtype if specified.
        if dtype is not None:
            self.franken = self.franken.to(dtype=dtype)
        self.dtype = next(
            tensor.dtype
            for tensor in (*self.franken.gnn.parameters(), *self.franken.gnn.buffers())
            if tensor.is_floating_point()
        )
        self.franken.gnn.franken_val()

        if hasattr(self.franken, "gnn_config"):
            gnn_config = self.franken.gnn_config
        elif gnn_config is None:
            raise ValueError(
                "Franken model does not have a GNN configuration."
                " This can happen if the model is scripted. "
                "Please pass an explicit gnn_config object instead."
            )

        self.dataset = FrankenAtomsDataset(
            data_path=None,
            split="md",
            gnn_config=gnn_config,
        )
        self.device = (
            device if device is not None else next(self.franken.parameters()).device
        )

    def calculate(
        self,
        atoms=None,
        properties=None,
        system_changes=all_changes,
    ):
        if properties is None:
            properties = self.implemented_properties

        super().calculate(atoms, properties, system_changes)

        # self.atoms is set in the super() call. Unclear why it should be preferred over `atoms`
        config_idx = self.dataset.add_configuration(self.atoms)  # type: ignore
        cpu_data = self.dataset.__getitem__(config_idx, no_targets=True)
        assert isinstance(cpu_data, Configuration)
        data = cpu_data.to(device=self.device, dtype=self.dtype)

        targets = [ENERGY_TARGET_KEY]
        if "forces" in properties:
            targets.append(FORCES_TARGET_KEY)
        if "stress" in properties:
            targets.append(STRESS_TARGET_KEY)
        computed = self.franken(targets, data)

        self.results["energy"] = computed[ENERGY_TARGET_KEY].item()
        if "forces" in properties:
            self.results["forces"] = (
                computed[FORCES_TARGET_KEY].squeeze(0).numpy(force=True)
            )
        if "stress" in properties:
            self.results["stress"] = (
                computed[STRESS_TARGET_KEY].squeeze(0).numpy(force=True)
            )


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
