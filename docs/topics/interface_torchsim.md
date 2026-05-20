# Torch-sim

| **Software** | **Backbones** | **Usage** |
| --- | --- | --- |
| torch-sim | MACE, PET | Python: [`FrankenTorchSimModel`](../reference/franken-api/stubs/franken.calculators.FrankenTorchSimModel.rst) |

This interface enables to use Franken models in [torch-sim](https://github.com/TorchSim/torch-sim).

### What is supported
- Outputs: `energy`, `forces`
- Batched evaluation
- Stress/virials: not supported in this interface

### Installation

 - The Python interface can be installed together with franken via `pip install franken[torch-sim]`.
 - torch-sim does not publish packages on conda-forge - installation must be with pip via `pip install torch-sim-atomistic`.
 - torch-sim requires **at least python 3.12**

## Load

### Load from checkpoint
`FrankenTorchSimModel` can be initialized directly from a Franken checkpoint path (simplest option).

```python
import torch
import torch_sim as ts

from franken.calculators import FrankenTorchSimModel

model = FrankenTorchSimModel(
    "path/to/best_ckpt.pt",
    device="cuda" if torch.cuda.is_available() else "cpu",
    dtype=torch.float32,
    rf_weight_id=None,  # set this when loading a multi-head checkpoint
)
```

### Load from an already-instantiated Franken model
```python
import torch

from franken.calculators import FrankenTorchSimModel
from franken.rf.model import FrankenPotential

franken = FrankenPotential.load("path/to/best_ckpt.pt", map_location="cuda")
model = FrankenTorchSimModel(franken, device="cuda", dtype=torch.float32)
```

## Run 

### Forward pass on a batched state
The interface accepts a torch-sim `SimState` and returns a dictionary with:
- `"energy"`: shape `[n_systems]`
- `"forces"`: shape `[n_atoms, 3]`

```python
import torch
import torch_sim as ts
from ase.build import molecule

from franken.calculators.torchsim_inf_wrap import FrankenTorchSimModel

h2o = molecule("H2O")
h2o.center(vacuum=5.0)
ch4 = molecule("CH4")
ch4.center(vacuum=5.0)

state = ts.io.atoms_to_state(
    [h2o, ch4],
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    dtype=torch.float32,
)

model = FrankenTorchSimModel("path/to/best_ckpt.pt", device=state.device, dtype=state.dtype)
out = model(state)

energy = out["energy"]  # [n_systems]
forces = out["forces"]  # [n_atoms, 3]
```

### Batched MD with `torch_sim.integrate`
This example runs multiple systems in parallel using one batched integration call.

```python
import torch
import torch_sim as ts
from ase.io import read

from franken.calculators import FrankenTorchSimModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model
mace_model = FrankenTorchSimModel(
    "path/to/best_ckpt.pt",
    device=device,
    dtype=torch.float32,
)

# system batch (same structure replicated)
atoms = read("init_structure.xyz")
batch_size = 16
many_atoms = [atoms.copy() for _ in range(batch_size)]
trajectory_files = [f"traj_{i}.h5md" for i in range(batch_size)]

# batched MD
final_state = ts.integrate(
    system=many_atoms,
    model=mace_model,
    n_steps=nsteps,
    timestep=0.001,
    temperature=100,
    integrator=ts.Integrator.nvt_langevin,
    trajectory_reporter=dict(filenames=trajectory_files, state_frequency=10),
)
final_atoms_list = final_state.to_atoms()
```
