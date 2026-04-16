# Overview

Franken currently supports multiple deployment and simulation interfaces, depending
on the backbone and the target software.

The table below summarizes the currently supported routes and links to the
dedicated pages.

| Interface | Software | Supported backbones | Usage mode | 
| --- | --- | --- | --- | 
| [ASE](interface_ase.md) | ASE | all | Python: [`FrankenCalculator`](../reference/franken-api/stubs/franken.calculators.FrankenCalculator.rst) | 
| [MACE-LAMMPS](interface_mace_lammps.md) | LAMMPS(+mace) | MACE | CLI: [`franken.wrap_mace_lammps`](../reference/franken-cli/franken.create_lammps_model.rst) | 
| [Metatomic](interface_metatomic.md)  | ASE, LAMMPS(+metatomic) | PET | CLI: [`franken.wrap_metatomic`](../reference/franken-cli/franken.create_lammps_model.rst) | 
| [Torch-sim](interface_torchsim.md) | torch-sim | MACE, PET | Python: [`FrankenTorchSimModel`](../reference/franken-api/stubs/franken.calculators.FrankenTorchSimModel.rst) | 
