# Using Franken with ASE

This page shows how to use a trained Franken checkpoint through the ASE calculator interface.
This is the simplest way to run single-point evaluations and molecular dynamics from Python.

## Load the calculator

```python
from ase.io import read

from franken.calculators import FrankenCalculator

atoms = read("init_structure.xyz")
atoms.calc = FrankenCalculator("path/to/best_ckpt.pt", device="cuda:0")

energy = atoms.get_potential_energy()
forces = atoms.get_forces()
```

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
