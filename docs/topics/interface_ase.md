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

For an end-to-end molecular dynamics workflow, see the
{doc}`ASE molecular dynamics tutorial <../notebooks/molecular_dynamics>`.

## GP uncertainty

GP uncertainty can be computed for energy and forces.
It requires to have saved the training Cholesky factor.

The supported outputs are `energy_gp_uncertainty` (scalar),
`per_atom_energy_gp_uncertainty` (shape `(natoms,)`), and
`forces_gp_uncertainty` (shape `(natoms, 3)`,).

```python
calc = atoms.calc
calc.calculate(
    atoms,
    properties=["energy", "forces", "per_atom_energy_gp_uncertainty"],
)
atomic_scores = calc.results["per_atom_energy_gp_uncertainty"]

# Request one property through ASE's cache-aware interface.
energy_score = calc.get_property("energy_gp_uncertainty", atoms)
```

Notes: 
* They are dimensionless posterior/prior standard-deviation ratios in `[0, 1]`,
not calibrated errors in energy or force units. 
* Force uncertainty requires feature Jacobians and can be expensive. 
* They are not calculated by default by `calculate(atoms, properties=None)`, only if requested.
