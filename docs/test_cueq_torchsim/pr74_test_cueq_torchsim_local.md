# Guide to Installing and Validating `franken` with cuEquivariance and TorchSim in a Local Environment (CUDA 12.6)

## 1. Environment Preparation and PyTorch Installation (CUDA 12.6)

PR #74 on the `feature/mace-default-install` branch requires **Python >= 3.11** and **PyTorch >= 2.5.0**. It is recommended to start from a clean virtual environment (e.g., `conda` or `venv`/`uv`):

```bash
# Create and activate a clean environment with Python 3.12
conda create -n test-franken python=3.12 -y
conda activate test-franken

# alternatively, with uv:
uv venv --python 3.12
source .venv/bin/activate
```

Check your local NVIDIA driver with `nvidia-smi`:

* With **Driver >= 580** (which supports up to CUDA 13.0) and a **Quadro RTX 4000 GPU (8 GB, Turing architecture `sm_75`)**, you can run builds compiled for **CUDA 12.6** without issues.

### Installing PyTorch with CUDA 12.6

Install PyTorch and torchvision from the official `cu126` index:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

### Post-Installation Verification of PyTorch

Check that PyTorch recognizes the local GPU and the CUDA 12.6 build:

```bash
python -c "import torch; print('PyTorch Version:', torch.__version__); print('CUDA Available:', torch.cuda.is_available()); print('Device Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'); print('PyTorch CUDA Version:', torch.version.cuda)"
```

**Expected outcome:**

* `CUDA Available: True`
* `Device Name: Quadro RTX 4000`
* `PyTorch CUDA Version: 12.6`

---

## 2. Installing `franken`, Default MACE, cuEquivariance, and TorchSim

MACE (`mace-torch >= 0.3.17, < 0.4`) is now a core dependency in `pyproject.toml`, and the `[mace]` extra has been removed. To align the local environment with Leonardo, install the package in editable mode including the `[develop,torch-sim,cueq-cu12]` extras and pinning `torch-sim-atomistic==0.6.2`:

```bash
cd /franken

# Preliminary dependency check (--dry-run)
python -m pip install --dry-run -e ".[develop,torch-sim,cueq-cu12]" "torch-sim-atomistic==0.6.2"

# Actual installation
python -m pip install -e ".[develop,torch-sim,cueq-cu12]" "torch-sim-atomistic==0.6.2"
```

### Verifying MACE, cuEquivariance Kernels, and Warp

Check that all modules and native GPU kernels are active:

```bash
python - <<'PY'
import mace
import torch
import warp as wp
import torch_sim as ts
from importlib.metadata import version
from cuequivariance_torch.primitives.segmented_polynomial import HAS_CUE_OPS

print("MACE:", mace.__version__)
print("TorchSim:", version("torch-sim-atomistic"))
print("Warp:", version("warp-lang"))
print("HAS_CUE_OPS:", HAS_CUE_OPS)

wp.init()
assert torch.cuda.is_available()
assert HAS_CUE_OPS, "cuEquivariance kernels not detected!"
assert wp.get_cuda_device_count() > 0

x = wp.zeros(4, dtype=wp.float32, device="cuda:0")
wp.synchronize()
print("PASS: CUDA 12.6 GPU environment ready on", torch.cuda.get_device_name(0))
PY
```

---

## 3. Training Smoke Test (Installation Check with MACE)

Run the smoke test command documented in `intro_installation.md` to verify that basic training with the MACE backbone works and generates a test checkpoint:

```bash
franken.autotune \
    --dataset-name test \
    --backbone mace --mace.path-or-id mace_mp/small \
    --rf gaussian --gaussian.num-rf 32 --gaussian.length-scale 1 \
    --force-weight 0.5 --l2-penalty 0.0001 \
    --jac-chunk-size 2 --run-dir installation-check
```

* **Expected outcome:** The process completes without errors and saves the fitted checkpoint (e.g., `installation-check/best_ckpt.pt` or in a `run_*` subfolder).

---

## 4. Running the Unit Test Suite (`pytest`)

Run the unit tests covering MACE, cuEquivariance, and the TorchSim integration:

```bash
cd franken/tests

# MACE and cuEquivariance tests
pytest test_mace_wrap.py test_mace_inf_wrap.py test_mace_cueq.py

# Specific tests for the TorchSim adapter
python -m pytest -v test_torchsim_inf_wrap.py test_mace_cueq.py::test_torchsim_cueq_batched_inference

# Full suite
pytest
```

---

## 5. Verifying Dependency Resolution with Other Extras

Test the combined installation of multiple extras to verify the absence of version conflicts among `e3nn`, `mace-torch`, `torch-sim`, and `metatrain`:

```bash
pip install -e ".[cueq-cu12,torch-sim,pet,docs]"
python -m pip check
```

You should receive `"WARNING: franken 0.8.1 does not provide the extra 'pet'"` since `pet` is not defined among the extras in `pyproject.toml`.

---

## 6. Inference and Backend Switching Tests (ASE `FrankenCalculator`)

Using the model saved in Section 3 (`installation-check/run_id/best_ckpt.pt`), verify numerical consistency between `use_cueq=True` and `use_cueq=False` via ASE:

```bash
python - <<'PY'
import glob
import numpy as np
from ase.build import bulk
from franken.calculators import FrankenCalculator

ckpt = "path/to/best_ckpt.pt"

# Create an 8-atom cubic Silicon unit cell
base_atoms = bulk("Si", "diamond", a=5.43, cubic=True)
base_atoms.positions += np.random.default_rng(17).normal(0, 0.03, base_atoms.positions.shape)
print(f"Base system: {len(base_atoms)} atoms ({base_atoms.get_chemical_formula()})")

# Test with cuEquivariance
calc_cueq = FrankenCalculator(ckpt, device="cuda", use_cueq=True)
base_atoms.calc = calc_cueq
e_cueq = base_atoms.get_potential_energy()
f_cueq = base_atoms.get_forces()

# Test with standard e3nn
calc_e3nn = FrankenCalculator(ckpt, device="cuda", use_cueq=False)
base_atoms.calc = calc_e3nn
e_e3nn = base_atoms.get_potential_energy()
f_e3nn = base_atoms.get_forces()

diff_e = abs(e_cueq - e_e3nn)
print(f"Energy Diff: {diff_e}")

max_diff_f = np.max(np.abs(f_cueq - f_e3nn))
print(f"Max Forces Diff: {max_diff_f}")

assert np.allclose(e_cueq, e_e3nn, atol=1e-3)
assert np.allclose(f_cueq, f_e3nn, atol=1e-3)
PY
```

*(Note: also verify that setting `use_cueq=True` during training raises an explicit exception, since training must remain on the `e3nn` backend.)*

---

## 7. Batched Inference and Molecular Dynamics Test with `TorchSim` (`use_cueq=True`)

Verify the `FrankenTorchSimModel` adapter with `cuEquivariance` directly enabled (`use_cueq=True`), validating both the batched numerical comparison against `e3nn` and an NVT molecular dynamics smoke test (`ts.integrate`) sized for the **8 GB of VRAM** on the Quadro RTX 4000:

```bash
python - <<'PY'
import glob
import torch
import torch_sim as ts
from ase.build import bulk
from franken.calculators.torchsim_inf_wrap import FrankenTorchSimModel

device = torch.device("cuda")
ckpt = "path/to/best_ckpt.pt"

# 1. Create a batch of 2 lightweight systems (64 atoms each, 128 total)
atoms = bulk("Si", "diamond", a=5.43, cubic=True).repeat((2, 2, 2))
distorted = atoms.copy()
distorted.positions += 0.03
systems = [atoms, distorted]

state = ts.io.atoms_to_state(systems, device=device, dtype=torch.float64)

# 2. Direct comparison between e3nn and cuEquivariance backends in TorchSim
model_e3nn = FrankenTorchSimModel(ckpt, device=device, use_cueq=False)
model_cueq = FrankenTorchSimModel(ckpt, device=device, use_cueq=True)
assert model_cueq.model.gnn.use_cueq is True

out_e3nn = model_e3nn(state)
out_cueq = model_cueq(state)

torch.testing.assert_close(out_cueq["energy"], out_e3nn["energy"], atol=1e-4, rtol=0)
torch.testing.assert_close(out_cueq["forces"], out_e3nn["forces"], atol=1e-4, rtol=1e-4)
print("Batch energies (cuEq):", out_cueq["energy"].detach().cpu().numpy())

# 3. NVT molecular dynamics smoke test (10 steps, dt = 0.1 fs, T = 100 K)
torch.manual_seed(17)
torch.cuda.manual_seed_all(17)

final_state = ts.integrate(
    system=[atoms.copy(), atoms.copy()],
    model=model_cueq,
    n_steps=10,
    timestep=0.0001,  # ps
    temperature=100,  # K
    integrator=ts.Integrator.nvt_langevin,
)

assert final_state.n_systems == 2
assert torch.isfinite(final_state.energy).all()
assert torch.isfinite(final_state.forces).all()
print("PASS: Batched inference and MD with TorchSim + cuEquivariance completed!")
PY
```

> **Note on VRAM memory:** Given the 8 GB of VRAM available on the Quadro RTX 4000, keep test batches within **1,500–2,000 total atoms** to avoid `CUDA Out of Memory` errors.

---

## 8. Running the Throughput Benchmark

Run the benchmark script included in the PR (`tests/benchmarks/mace_inference.py`) to compare inference performance between `e3nn` and `cuEquivariance` across varying atom counts (limiting the maximum atom count via the `--cells` parameter based on GPU VRAM):

```bash
python tests/benchmarks/mace_inference.py "/path/to/best_ckpt.pt" --cells 1 2 3 4 --plot throughput_f0.png
```

* **What to check:**
  * Numerical output comparison between backends (`torch.testing.assert_close`).
  * Proper generation of the `throughput_f0.png` plot and throughput table.

---

## 9. Verifying Sphinx Documentation

Verify that the updated documentation in `docs/` builds without syntax errors or broken references:

```bash
# install documentation dependencies
pip install -e ".[docs]"

# install pandoc in the conda environment to render it
conda activate test-franken
conda install -c conda-forge pandoc=3.6.4
pandoc --version   # check

# navigate to the docs folder
cd docs

# build html
make html
```

Check that the sections in `intro_installation.md` and `README.md` regarding the new installation procedure and inference with cuEquivariance render properly.
