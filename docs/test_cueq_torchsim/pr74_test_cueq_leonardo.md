# Guide to Installing and Validating `franken` with cuEquivariance on Leonardo (CINECA)

This document provides the complete procedure for installing `franken` from scratch (branch `feature/mace-default-install`, PR #74) on the **Leonardo** HPC cluster, configuring GPU acceleration via **NVIDIA cuEquivariance**, and validating numerical accuracy and inference performance against the standard `e3nn` backend.

---

## Overview of PR #74 and Expected Results

Pull Request #74 introduces major simplifications to the installation ecosystem and a significant optimization for large-scale molecular simulations:

1. **MACE as a default dependency**: The `mace-torch` (`>= 0.3.17, < 0.4`) package is now integrated into the core dependencies of `franken`. The optional `[mace]` extra has been removed.
2. **Native `cuEquivariance` support at inference**: Computation can be accelerated via optimized CUDA kernels (`cuequivariance-torch`) by enabling the `use_cueq=True` flag at inference time (`FrankenCalculator`), while maintaining full compatibility with checkpoints trained using `e3nn`.
3. **Numerical consistency and performance**:
   * **Numerical agreement**: In the Section 7 test on a Si structure, $\Delta E \sim 10^{-7}\text{ eV}$ and a maximum force difference of $\sim 10^{-6}\text{ eV/Å}$ are observed, and the assertions on energy and forces pass.
   * **Speedup**: In the exploratory benchmark in Section 8 on an A100, using the model and structures described below, a **speedup of up to 5.25x** is observed for 8,000 atoms. This result is not a guarantee for other systems or models.

---

## 1. Choosing the Working Directory

On Leonardo, the `$HOME` directory has limited disk space, while `$WORK` areas are tied to individual project expiration dates. For a persistent, project-independent installation, use the user's **Public** directory:

```bash
cd /leonardo/pub/userexternal/$USER
```

---

## 2. Clean Installation of Miniconda

To avoid conflicts with pre-existing modules or corrupted environments, follow the official CINECA procedure starting from a clean environment:

### 2.1 Cleaning up any previous installations

Remove residual configuration files in `$HOME` and clean up any `conda initialize` blocks from `~/.bashrc`:

```bash
rm -f ~/.condarc
rm -rf ~/.conda
# Open ~/.bashrc and manually remove any references to Conda/Anaconda
```

### 2.2 Download and installation

Download and run the official Miniconda installer, specifying the `public` directory as the destination path:

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

* **During setup**: answer `yes` to the license agreement, specify `/leonardo/pub/userexternal/$USER/miniconda3` as the installation directory, and answer `yes` to automatic shell initialization.
* Reload the bash session:

```bash
exec bash
```

### 2.3 Channel Configuration (Strict Conda-Forge)

Disable automatic activation of the `(base)` environment and configure only the `conda-forge` channel with strict priority, avoiding commercial Anaconda repositories (`pkgs/main`, `pkgs/r`):

```bash
conda config --set auto_activate_base false
exec bash

conda config --set channel_priority strict
conda config --add channels conda-forge
```

Edit `~/.condarc` and `/leonardo/pub/userexternal/$USER/miniconda3/.condarc` by commenting out (with `#`) any lines containing `https://repo.anaconda.com/pkgs/main` or `https://repo.anaconda.com/pkgs/r`. The final content should be:

```yaml
channels:
  - conda-forge
channel_priority: strict
auto_activate_base: false
```

Verify the configuration with:

```bash
conda config --show-sources
```

---

## 3. Environment Creation and PyTorch Installation

Ensure you are on a **login node** (e.g., `login01`, which has internet access).

### 3.1 Creating the virtual environment

```bash
conda create -n franken-cueq python=3.12 -y
conda activate franken-cueq
```

Verify that `git` is available on the system (or install it in the environment) and that PyTorch is not yet installed:

```bash
which git  # Expected output: /usr/bin/git
python -c "import torch"  # Expected output: ModuleNotFoundError
```

### 3.2 Checking CUDA modules and installing PyTorch

Check the CUDA versions available on the cluster:

```bash
module avail cuda
```

Assuming **CUDA 12.6** (`cuda/12.6`) is available, install the corresponding PyTorch build via `pip`:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

Verify the installation:

```bash
python -c "import torch; print('PyTorch Version:', torch.__version__, '| PyTorch CUDA:', torch.version.cuda)"
# Obtained output: PyTorch Version: 2.14.1+cu126 | PyTorch CUDA: 12.6
```

---

## 4. Installing `franken` and `cuEquivariance` Extras

### 4.1 Cloning the repository and checking out the PR

```bash
cd /leonardo/pub/userexternal/$USER
git clone https://github.com/CSML-IIT-UCL/franken.git
cd franken

git fetch origin
git checkout feature/mace-default-install
git branch -a  # Verify that the active branch is feature/mace-default-install
```

### 4.2 Base installation and verifying MACE dependencies

Install `franken` in editable mode with development dependencies:

```bash
pip install -e ".[develop]"
```

Verify that `mace-torch` was installed automatically and that the old `[mace]` extra no longer exists:

```bash
python -c "import mace; print('MACE version:', mace.__version__)"
# Obtained output: MACE version: 0.3.17

pip install ".[mace]"
# Obtained output: WARNING: franken 0.7.0 does not provide the extra 'mace'
```

### 4.3 Installing `cuEquivariance` kernels

Since PyTorch was built for CUDA 12.6, install the corresponding `cueq-cu12` extra:

```bash
pip install -e ".[cueq-cu12]"
```

(Note: testing the combined installation `pip install -e ".[cueq-cu12,pet]"` yields `WARNING: franken 0.7.0 does not provide the extra 'pet'`, because `pet` is not defined among the extras in `pyproject.toml`, though the installation proceeds without errors.)

**Note on kernel verification on the login node:**
Running the native kernel check on the login node will produce:

```text
Error while loading libcue_ops.so: libcuda.so.1: cannot open shared object file
HAS_CUE_OPS: False
```

This behavior is **completely normal**: Leonardo login nodes do not have physical GPUs and lack NVIDIA drivers (`libcuda.so.1`). The check must be repeated on a compute node.

---

## 5. Smoke Test and Backbone Cache Management

Leonardo compute nodes **do not have internet access**. Therefore, the initial download of pre-trained backbones (e.g., `mace_mp/small`) must be performed from the **login node** so that they are saved in the local cache (`~/.franken`).

From the login node, run the training smoke test:

```bash
cd /leonardo/pub/userexternal/$USER

franken.autotune \
    --dataset-name test \
    --backbone mace --mace.path-or-id mace_mp/small \
    --rf gaussian --gaussian.num-rf 32 --gaussian.length-scale 1 \
    --force-weight 0.5 --l2-penalty 0.0001 \
    --jac-chunk-size 2 --run-dir installation-check
```

* **Obtained result**: Training completes successfully, generating the directory `installation-check/run_<timestamp>_<jobid>/` containing the `best_ckpt.pt` checkpoint.

Tip: to pre-download additional backbones before submitting jobs on compute nodes, run `franken.backbones download <backbone_name>` from the login node. See the official documentation for supported backbone names: <https://franken.readthedocs.io/topics/training_backbones.html>

---

## 6. Running Tests on a Compute Node (A100 GPU)

Running the full `pytest` suite and GPU tests requires a compute node (on login nodes, the full tests cause *Out-Of-Memory* errors).

### 6.1 Requesting an interactive node in the debug queue

First check your active project code with `saldo -b`, then request a node with 1 A100 GPU (30-minute maximum limit on the `boost_qos_dbg` queue):

```bash
srun --nodes=1 --ntasks-per-node=1 --cpus-per-task=8 --gres=gpu:1 --time=00:30:00 -p boost_usr_prod --qos=boost_qos_dbg -A <PROJECT_ID> --pty bash
```

Once the node is allocated (e.g., `lrdn0035`), activate the environment and verify GPU and `cuEquivariance` kernel detection:

```bash
conda activate franken-cueq

# Check A100 GPU
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '| GPU:', torch.cuda.get_device_name(0), '| CUDA ver:', torch.version.cuda)"
# Expected output: CUDA: True | GPU: NVIDIA A100-SXM-64GB | CUDA ver: 12.6

# Check native cuEquivariance kernels
python -c "import cuequivariance_torch; from cuequivariance_torch.primitives.segmented_polynomial import HAS_CUE_OPS; print('HAS_CUE_OPS:', HAS_CUE_OPS)"
# Obtained output: HAS_CUE_OPS: True
```

### 6.2 Running the `pytest` suite

Navigate to the repository directory and run the unit tests:

```bash
cd /leonardo/pub/userexternal/$USER/franken

pytest tests/test_mace_wrap.py tests/test_mace_inf_wrap.py   # PASSED
pytest tests/test_mace_cueq.py                               # PASSED
pytest                                                       # Full suite
```

* **Note on connectivity**: If `requests.exceptions.ConnectionError` errors occur during the full `pytest` run, start `pytest` for a few seconds on the login node so it downloads the missing test files into the `~/.franken` cache, interrupt it (`Ctrl+C`), and re-run it on the compute node.

---

## 7. Inference and Numerical Equivalence Test (`e3nn` vs `cuEquivariance`)

On the compute node, create and run an `inference_test.py` script to compare the energies and forces of the `FrankenCalculator` ASE calculator between `use_cueq=True` and `use_cueq=False`.

> **Limitations:** This test and the next use the `best_ckpt.pt` produced in Section 5 (trained on the H/N `test` dataset) and evaluate inference performance on a cubic silicon cell. These are technical tests of agreement and performance, not a validation of physical accuracy on the silicon structure. The conclusions should not be generalized to other checkpoints or structures.

```python
import numpy as np
from ase.build import bulk
from franken.calculators import FrankenCalculator

# Create a cubic Silicon cell (8 atoms) with a random perturbation
base_atoms = bulk("Si", "diamond", a=5.43, cubic=True)
base_atoms.positions += np.random.default_rng(17).normal(0, 0.03, base_atoms.positions.shape)
print(f"Base system: {len(base_atoms)} atoms ({base_atoms.get_chemical_formula()})")

# Update with the actual path generated in Section 5
ckpt_path = "/installation-check/run_<ID>/best_ckpt.pt"

# 1. Inference with cuEquivariance
calc_cueq = FrankenCalculator(ckpt_path, device="cuda", use_cueq=True)
base_atoms.calc = calc_cueq
e_cueq = base_atoms.get_potential_energy()
f_cueq = base_atoms.get_forces()

# 2. Inference with standard e3nn
calc_e3nn = FrankenCalculator(ckpt_path, device="cuda", use_cueq=False)
base_atoms.calc = calc_e3nn
e_e3nn = base_atoms.get_potential_energy()
f_e3nn = base_atoms.get_forces()

diff_e = abs(e_cueq - e_e3nn)
print(f"Energy Diff: {diff_e}")

max_diff_f = np.max(np.abs(f_cueq - f_e3nn))
print(f"Max Forces Diff: {max_diff_f}")

assert np.allclose(e_cueq, e_e3nn, rtol=1e-05, atol=1e-3)
assert np.allclose(f_cueq, f_e3nn, rtol=1e-05, atol=1e-3)
print("Validation passed successfully!")
```

**Obtained result**:

* `Energy Diff: 2.7291644300042606e-07 eV`
* `Max Forces Diff: 8.344650268554688e-07 eV/Å`
* Both assertions on energy and forces passed

### 7.1 Test with the Water Dataset

In this test, the previous check was repeated using the water dataset. The dataset and the checkpoint are described below.

#### Dataset

The water tests use the `water` dataset of the franken dataset registry: liquid water configurations with reference energies and forces computed with DFT at the RPBE+D3 level, collected by [Montero de Hijes et al.](https://doi.org/10.1063/5.0197105). The first time the dataset is requested, franken downloads it from Zenodo and converts it to extended XYZ format in its cache directory (`~/.franken/water/`).

The source file `~/.franken/water/ML_AB_dataset_1.xyz` contains 1,495 periodic configurations: the first 473 have 192 atoms (64 H₂O molecules), and the remaining 1,022 have 189 atoms (63 H₂O molecules). It was split into three non-overlapping subsets with ASE:

```python
from pathlib import Path
from ase.io import read, write

structures = read(Path.home() / ".franken/water/ML_AB_dataset_1.xyz", index=":")

write("train_from_split.xyz", structures[:100])
write("val_from_split.xyz", structures[100:450])
write("test_from_split.xyz", structures[-350:])
```

| Split | File | Slice | Frames | Atoms per frame |
| --- | --- | --- | --- | --- |
| Training | `train_from_split.xyz` | `[:100]` | 100 | 192 |
| Validation | `val_from_split.xyz` | `[100:450]` | 350 | 192 |
| Test | `test_from_split.xyz` | `[-350:]` | 350 | 189 |

Training and validation frames come from the 64-molecule part of the dataset, while the test frames come from the 63-molecule part; for this reason the water systems used in the tests have 189 atoms.

#### Training the Checkpoint

The checkpoint was trained with `franken.autotune` on the training split, selecting the hyperparameters with a grid search evaluated on the validation split. The grid covers:

* the L2 penalty: 6 log-spaced values from 1e-11 to 1e-6;
* the force weight: 10 log-spaced values from 1e-2 to 1e4;
* the Gaussian length scale: 5, 10, 20 and 30.

The search was launched with `torchrun` on a single node of an HPC cluster (see the multi-GPU training documentation); `GPUS_PER_NODE`, `RDZV_ID`, `MASTER_ADDR`, `MASTER_PORT` and `RUN_DIR` are set by the job script:

```bash
torchrun \
    --nnodes=1 \
    --nproc-per-node="${GPUS_PER_NODE}" \
    --rdzv-id="${RDZV_ID}" \
    --rdzv-backend=c10d \
    --rdzv-endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    -m franken.autotune \
    --train-path /path/to/train_from_split.xyz \
    --val-path /path/to/val_from_split.xyz \
    --max-train-samples 100 \
    --train-targets energy forces \
    --l2-penalty="(-11, -6, 6, log)" \
    --force-weight="(-2, 4, 10, log)" \
    --metrics energy_MAE forces_MAE \
    --jac-chunk-size 8 \
    --run-dir "${RUN_DIR}" \
    --backbone=mace \
    --mace.path-or-id "mace_mh/0" \
    --mace.interaction-block 2 \
    --rf=gaussian \
    --gaussian.num-rf 4096 \
    --gaussian.length-scale="[5., 10.0, 20.0, 30.0]"
```

The best configuration selected on the validation split is:

| Parameter | Value |
| --- | --- |
| Backbone | `mace_mh/0` (MACE, interaction block 2) |
| Random features | Gaussian, 4,096 features, length scale 20.0, with offset, RNG seed 1337 |
| Feature scaling | Per species, `leading_eig` covariance normalization |
| Training targets | Energy and forces (energy weight 1.0, force weight 100.0) |
| L2 penalty | 1e-8 |

To retrain only this configuration, pass single values instead of the ranges: `--l2-penalty 1e-8 --force-weight 100 --gaussian.length-scale 20`.

Training was run on 2 × Tesla V100-SXM2-16GB GPUs; the selected configuration reaches a validation MAE of 0.60 meV/atom on energies and 12.76 meV/Å on forces. The resulting `best_ckpt.pt` and `test_from_split.xyz` were then copied to a `water_test_split/` directory on Leonardo: in the scripts below, replace `/path/to/water_test_split` with the actual location of that directory.

> **Note:** In the previous test with silicon, random noise was needed to break the crystal's perfect symmetry and avoid zero forces. Here, instead, an actual frame from the liquid water dataset is loaded: the molecules are already disordered and have real physical forces. Adding random noise risks bringing two hydrogen atoms too close together (causing huge, unphysical forces).

```python
import numpy as np
from ase.io import read
from franken.calculators import FrankenCalculator

# 1. Load the first frame of the water test set
atoms = read("/path/to/water_test_split/test_from_split.xyz", index=0)
print(f"Base system: {len(atoms)} atoms ({atoms.get_chemical_formula()})")

# 2. Path to the checkpoint trained on water (update with the actual path)
ckpt_path = "/path/to/water_test_split/best_ckpt.pt"

# 3. Inference with cuEquivariance
calc_cueq = FrankenCalculator(ckpt_path, device="cuda", use_cueq=True)
atoms.calc = calc_cueq
e_cueq = atoms.get_potential_energy()
f_cueq = atoms.get_forces()

# 4. Inference with standard e3nn
calc_e3nn = FrankenCalculator(ckpt_path, device="cuda", use_cueq=False)
atoms.calc = calc_e3nn
e_e3nn = atoms.get_potential_energy()
f_e3nn = atoms.get_forces()

# 5. Compare results
diff_e = abs(e_cueq - e_e3nn)
print(f"Energy Diff: {diff_e}")

max_diff_f = np.max(np.abs(f_cueq - f_e3nn))
print(f"Max Forces Diff: {max_diff_f}")

assert np.allclose(e_cueq, e_e3nn, rtol=1e-05, atol=1e-3)
assert np.allclose(f_cueq, f_e3nn, rtol=1e-05, atol=1e-3)
print("Validation passed successfully!")
```

**Obtained result**:

* `Energy Diff: 9.81859557214193e-07 eV`
* `Max Forces Diff: 3.0994415283203125e-06 eV/Å`
* Both assertions on energy and forces passed

---

## 8. Throughput Benchmark (`mace_inference.py`)

On the compute node, run the benchmark script included in the PR to measure inference latency (in milliseconds per step) as the number of atoms grows.

As in Section 7, the H/N checkpoint is applied to silicon structures. These timings are useful for a technical comparison on this setup, but do not necessarily represent the performance of a model trained for the system of interest. The script generates the Si structures internally and does not read an XYZ dataset.

```bash
cd /leonardo/pub/userexternal/$USER/franken

python tests/benchmarks/mace_inference.py /leonardo/pub/userexternal/$USER/installation-check/run_<ID>/best_ckpt.pt --plot throughput.png
```

### Results Obtained on NVIDIA A100-SXM-64GB

* **Hardware/software environment**: `NVIDIA A100-SXM-64GB` | `PyTorch 2.14.1+cu126` | `MACE 0.3.17` | `cuEquivariance 0.10.0`
* **Model configuration**: Backbone `mace_mp/small` | Random Features: `32`

| Engine | Number of Atoms | `e3nn` Time (ms) | `cuEq` Time (ms) | Speedup (`e3nn / cuEq`) |
| --- | --- | --- | --- | --- |
| `direct` | 8 | 13.326 | 17.058 | 0.78x |
| `direct` | 64 | 13.320 | 17.091 | 0.78x |
| `direct` | 216 | 13.666 | 17.258 | 0.79x |
| `direct` | 512 | 18.293 | 17.228 | **1.06x** |
| `direct` | 1000 | 30.349 | 17.330 | **1.75x** |
| `direct` | 1728 | 49.026 | 17.323 | **2.83x** |
| `direct` | 4096 | 108.912 | 22.716 | **4.79x** |
| `direct` | 8000 | 208.094 | 39.644 | **5.25x** |

### Performance Analysis

1. **Low atom count regime ($< 500$ atoms)**: in this test, `cuEquivariance` exhibits a latency around `~17.1–17.3 ms`, making it slower than `e3nn` (`0.78x`).
2. **Break-even point**: in this test, around **512 atoms**, `cuEquivariance` surpasses `e3nn` (`1.06x`).
3. **Larger systems ($1,000 - 8,000$ atoms)**: in this test, `cuEquivariance` maintains nearly constant runtimes up to `1,728` atoms (`17.323 ms`, **2.83x** speedup) and reaches a **5.25x speedup** at `8,000` atoms (`39.644 ms` vs `208.094 ms`). These results do not constitute a production benchmark.
4. **Graphical output**: Since `TorchSim` is not installed, the table reports exclusively the `direct` engine (*precomputed neighbors*), and the `throughput.png` plot is generated with the corresponding single panel.

(Once testing is complete, remember to release the compute node resources by typing `exit`.)
