(installation)=
# Installation

## Install Franken with MACE

Franken requires Python 3.11 or newer. The default installation includes
[MACE](https://github.com/ACEsuit/mace) and the dependencies needed for training:

```bash
pip install franken
```

In a [uv](https://docs.astral.sh/uv/) project, use:

```bash
uv add franken
```

Franken uses the supported MACE 0.3 release series, starting at 0.3.17. The
`franken[mace]` extra remains an alias for existing installation commands.
There is no additional backbone package to install for the MACE examples.
Pretrained checkpoints download on first use, which requires network access.

To check a small training run with the bundled test data:

```bash
franken.autotune \
    --dataset-name test \
    --backbone mace --mace.path-or-id mace_mp/small \
    --rf gaussian --gaussian.num-rf 32 --gaussian.length-scale 1 \
    --force-weight 0.5 --l2-penalty 0.0001 \
    --jac-chunk-size 2 --run-dir installation-check
```

Use `uv run franken.autotune ...` in a uv project. This tiny dataset checks the
installation; use your own training and validation data for a useful model.
The run writes its results below `installation-check`.

## CPU and GPU support

The same MACE backend supports CPUs and NVIDIA GPUs through PyTorch. **CuPy is
not required for GPU training.** If you need a particular CUDA or CPU-only
PyTorch build, install it first using the
[PyTorch installation guide](https://pytorch.org/get-started/locally/), then
install Franken in that environment.

CuPy is an optional acceleration dependency for the low-memory solver. Use a
binary wheel matching your CUDA installation, for example:

```bash
pip install cupy-cuda12x  # CUDA 12
# or: pip install cupy-cuda13x  # CUDA 13
```

The older `franken[cuda]` extra installs the generic `cupy` package, which may
build from source and require a CUDA toolkit. Prefer the matching binary wheel
when you need this solver; install only one CuPy distribution.

## Optional cuEquivariance acceleration

cuEquivariance provides native kernels for some MACE operations on CUDA GPUs.
It is optional so the default installation also works on CPU-only systems.
Check the CUDA version of your installed PyTorch build:

```bash
python -c "import torch; print(torch.version.cuda)"
```

For a CUDA 13 build, install:

```bash
pip install "franken[cueq-cu13]"
# uv projects: uv add "franken[cueq-cu13]"
```

For a CUDA 12 build, select `franken[cueq-cu12]` instead. These extras install
unmodified upstream MACE and a consistent cuEquivariance 0.10 release series,
within MACE's supported dependency ranges. Native wheel availability and an
appropriate NVIDIA driver are required. Choose one CUDA extra matching
PyTorch; installing a CUDA extra does not replace a CPU-only PyTorch build.

Training always uses the standard e3nn backend and its forward-mode feature
derivatives. There is no cuEquivariance training flag. To accelerate a fitted
MACE model during ASE inference:

```python
from franken.calculators import FrankenCalculator

calc = FrankenCalculator(
    "path/to/run/best_ckpt.pt", device="cuda", use_cueq=True
)
atoms.calc = calc
energy = atoms.get_potential_energy()
forces = atoms.get_forces()
```

For direct model use:

```python
from franken.rf.model import FrankenPotential

model = FrankenPotential.load(
    "path/to/run/best_ckpt.pt", map_location="cuda", use_cueq=True
)
# Or, for an existing fitted model:
model.enable_cueq()
```

Conversion replaces the frozen backbone and preserves the fitted random-feature
coefficients, input scaler and energy shifts. It uses the original registered
backbone checkpoint and unmodified upstream MACE conversion. Standard reverse
autograd computes inference forces and stress; no full training Jacobian or
double-backward fallback is needed. Convolution fusion stays disabled; native
symmetric contractions are enabled.

A model saved after conversion records the inference backend. Pass
`use_cueq=False` when loading to select e3nn again, or call
`model.enable_cueq(False)` before fitting further. The same override is available
in `FrankenCalculator`. With no override, the saved choice is preserved.
`MaceBackboneConfig(..., use_cueq=True)` also selects an inference-only backbone;
training and training feature-map requests reject it explicitly.

The first GPU evaluation may compile kernels and take several seconds. Benchmark
your inference workload before enabling acceleration. A missing or incompatible
native extension raises an error rather than silently falling back. The supported
cuEquivariance 0.10 wheels have no native kernels for Volta/V100 GPUs; use the
e3nn backend on those devices. The default e3nn backend supports both CPU and GPU
inference. Automatic backend selection is supported for eager Franken checkpoints,
not existing TorchScript archives.

## Additional backbones and interfaces

[UPET](https://github.com/lab-cosmo/upet/tree/main) is available through the
Metatomic/Metatrain ecosystem:

```bash
pip install "franken[pet]"
```

See the [Metatomic](https://docs.metatensor.org/metatomic/latest/installation.html)
and [Metatrain](https://docs.metatensor.org/metatrain/latest/installation.html)
installation guides for details. MACE remains included when extras are added.

MACE 0.3.17 requires e3nn 0.4.4. Franken respects that upstream requirement;
it does not override it or distribute a patched MACE package. Libraries
requiring a different e3nn or PyTorch version may need a separate environment.

## Conda environments

Conda can provide the Python environment while pip installs Franken:

```bash
conda create -n franken python=3.12 pip
conda activate franken
pip install franken
```

For GPU use, choose a compatible PyTorch build as described above. Conda does
not require a separate MACE or CuPy installation for the default training path.
