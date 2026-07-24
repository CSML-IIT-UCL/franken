# ASE

| **Software** | **Backbones** | **Usage** |
| --- | --- | --- |
| ASE | all | Python: [`FrankenCalculator`](../reference/franken-api/stubs/franken.calculators.FrankenCalculator.rst) |

The native ASE calculator is the simplest way to run single-point evaluations and molecular dynamics with Franken models from Python.

## Load the calculator

```python
from ase.io import read

from franken.calculators import FrankenCalculator

atoms = read("init_structure.xyz")
atoms.calc = FrankenCalculator("path/to/best_ckpt.pt", device="cuda:0")

energy = atoms.get_potential_energy()
forces = atoms.get_forces()
stress = atoms.get_stress()
```

> **Note**: `FrankenCalculator` strictly adheres to [ASE units](https://docs.ase-lib.org/ase/units.html). The predicted `energy` is returned in `eV`, `forces` in `eV/Å`, and `stress` in `eV/Å³`. Similarly, when coupling the calculator with an ASE molecular dynamics engine (like `NPT`), target pressures and parameters should be explicitly converted to these atomic units using the constants provided in `ase.units`.

## Run molecular dynamics in ASE

```python
from ase import units
from ase.io import read
from ase.md.langevin import Langevin

from franken.calculators import FrankenCalculator

atoms = read("init_structure.xyz")
atoms.calc = FrankenCalculator("path/to/best_ckpt.pt", device="cuda:0")

dyn = Langevin(
    atoms,
    timestep=0.5 * units.fs,
    temperature_K=300.0,
    friction=0.01,
)
dyn.run(1000)
```

For a complete end-to-end example, see the
{doc}`molecular dynamics tutorial <../notebooks/molecular_dynamics>`.
