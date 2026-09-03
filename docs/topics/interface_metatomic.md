# Metatomic

| **Software** | **Backbones** | **Usage** |
| --- | --- | --- |
| ASE, LAMMPS(+metatomic) | PET | CLI: [`franken.wrap_metatomic`](../reference/franken-cli/franken.create_lammps_model.rst) |

The Metatomic export turns a Franken-PET checkpoint into a compiled model that can be used from ASE and the Metatomic LAMMPS interface.

## Compile a Franken checkpoint

Export the trained checkpoint with the dedicated CLI:

```bash
franken.wrap_metatomic --model_path path/to/best_ckpt.pt --dtype float64
```

This creates a compiled model named `best_ckpt-metatomic.pt` next to the original
checkpoint.

## Use the compiled model with ASE

```python
import ase.build
import ase.md
import ase.md.velocitydistribution
import ase.units
import numpy as np

from metatomic_ase import MetatomicCalculator

primitive = ase.build.bulk(name="C", crystalstructure="diamond", a=3.567)
atoms = ase.build.make_supercell(primitive, 3 * np.eye(3))
ase.md.velocitydistribution.MaxwellBoltzmannDistribution(atoms, temperature_K=300)

atoms.calc = MetatomicCalculator("path/to/best_ckpt-metatomic.pt", device="cuda")

integrator = ase.md.Langevin(
    atoms,
    timestep=1.0 * ase.units.fs,
    temperature_K=300,
    friction=0.1 / ase.units.fs,
)
integrator.run(100)
```

## Use the compiled model with LAMMPS

### Installation 

The `lammps-metatomic` version can be easily installed with conda. Please refer to the official manual:

https://docs.metatensor.org/metatomic/latest/engines/lammps.html

### Simulation

Once `lammps-metatomic` is installed, the compiled model can be used directly in
LAMMPS:

```text
pair_style metatomic best_ckpt-metatomic.pt
pair_coeff * * 1 8
```

The atomic numbers in `pair_coeff` must match the species ordering in your
LAMMPS data file. A complete minimal input used in CI is:

```text
units           metal
atom_style      atomic
boundary        p p p

read_data       water.lammpsdata

mass 1 1.00794
mass 2 15.9994

pair_style      metatomic best_ckpt-metatomic.pt
pair_coeff      * * 1 8
timestep        0.0005

velocity        all create 300 111 dist gaussian
fix             fix_nve all nve
fix             fix_temp all temp/csvr 300 300 0.1 111
run             10
```
