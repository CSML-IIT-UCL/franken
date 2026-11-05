(installation)=
# Installation
Franken requires Python 3.11 or newer, [MACE](https://github.com/acesuit/mace) support is included in the default
installation.
```{raw} html
<button type="button" class="agent-onboarding__copy" data-agent-guide="../agent-setup.md" aria-live="polite">
  <span class="agent-onboarding__icons" aria-hidden="true">
    <img src="../_static/agent-icons/claude.svg" alt="" width="20" height="20">
    <img src="../_static/agent-icons/codex.svg" alt="" width="20" height="20" class="agent-onboarding__adaptive-icon">
    <img src="../_static/agent-icons/opencode.svg" alt="" width="20" height="20" class="agent-onboarding__adaptive-icon">
  </span>
  <span data-copy-label>Copy setup prompt</span>
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" aria-hidden="true"><rect x="8" y="8" width="12" height="12" rx="2"/><path d="M16 8V5a2 2 0 0 0-2-2H5a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h3"/></svg>
</button>
```


## Default installation

With `pip`, create an environment and install Franken:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install franken
```

With [uv](https://docs.astral.sh/uv/), create a project and add Franken:

```bash
uv init --bare --python 3.13 my-franken-project
cd my-franken-project
uv add franken
```

In an existing environment or uv project, only the last command is needed.
The installer will choose a PyTorch build automatically; if you prefer choosing your own, see {ref}`Choosing a PyTorch build <choosing-a-pytorch-build>`.

## Extras

Franken supports a wide range of integrations, which can be installed as extras:

| Name | Integration |
|---|---|
| `metatomic` | [PET/UPET](https://lab-cosmo.github.io/upet/) backbones and inference with [Metatomic](https://docs.metatensor.org/) |
| `torch-sim` | Inference through [TorchSim](https://torchsim.github.io/torch-sim/) |
| `cueq-cu<12/13>` | [cuEquivariance](https://docs.nvidia.com/cuda/cuequivariance/) CUDA 12/13 inference kernels |
| `cuda` | Memory-efficient solvers, through [CuPy](https://cupy.dev/) **source build** (requires a local CUDA toolkit) |

For the cuEquivariance extras, first select the matching CUDA PyTorch build
using the recipes below. For CuPy, the {ref}`binary-wheel recipe <cupy-for-the-low-memory-solver>`
is usually easier than the legacy `cuda` extra.

You can combine the extras you want in one requirement, for example:

```bash
python -m pip install "franken[metatomic,torch-sim]"
# In a uv project:
uv add "franken[metatomic,torch-sim]"
```

If you previously ran `uv add torch`, uv may have saved a lower bound matching
the newest PyTorch release. If that version is newer than Metatomic supports,
replace the direct requirement with Franken's supported lower bound while adding
the extra:

```bash
uv add "torch>=2.5" "franken[metatomic]"
```

This lets uv choose a compatible version. If another dependency requires an
incompatible PyTorch or e3nn version, use a separate environment. See the
[Metatomic installation guide](https://docs.metatensor.org/metatomic/latest/installation.html)
for its supported PyTorch versions.

(choosing-a-pytorch-build)=
## Choosing a PyTorch build

### pip

Use the PyTorch CPU index alongside PyPI to install a CPU-only environment,
including any extras you need:

```bash
python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu \
    "franken[metatomic,torch-sim]"
```

For CUDA 12.8 with cuEquivariance and binary CuPy:

```bash
python -m pip install --extra-index-url https://download.pytorch.org/whl/cu128 \
    "torch==2.11.0+cu128" "franken[cueq-cu12]" "cupy-cuda12x[ctk]"
```

For CUDA 13.0:

```bash
python -m pip install --extra-index-url https://download.pytorch.org/whl/cu130 \
    "torch==2.14.1+cu130" "franken[cueq-cu13]" "cupy-cuda13x[ctk]"
```

These pip recipes select a PyTorch build explicitly.

### uv

In a uv project, add this configuration to `pyproject.toml`. Merge entries if
you already have a sources or index table:

```toml
[[tool.uv.index]]
name = "pytorch"
url = "https://download.pytorch.org/whl/cpu"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch" }
```

Then install the CPU-only build and your extras together:

```bash
uv add "torch>=2.5" "franken[metatomic,torch-sim]"
```

For CUDA 12.8, change the index URL to
`https://download.pytorch.org/whl/cu128` and run:

```bash
uv add "torch>=2.5" "franken[cueq-cu12]" "cupy-cuda12x[ctk]"
```

For CUDA 13.0, use `https://download.pytorch.org/whl/cu130` and run:

```bash
uv add "torch>=2.5" "franken[cueq-cu13]" "cupy-cuda13x[ctk]"
```

The explicit index binds PyTorch to the selected build while leaving other
packages on PyPI. The `torch>=2.5` requirement allows extras to select a compatible
version instead of fixing the project to the newest release. See
[uv's PyTorch guide](https://docs.astral.sh/uv/guides/integration/pytorch/).

### Check the selected build

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

In uv, prefix this command with `uv run`. `torch.version.cuda` is `None` for a
CPU-only build, `12.x` for a CUDA 12 build, and `13.x` for a CUDA 13 build.
`torch.cuda.is_available()` must be true to execute on the GPU. Installing a
cuEquivariance extra does not turn a CPU-only PyTorch into a CUDA build.

(cupy-for-the-low-memory-solver)=
## CuPy for the low-memory solver

CuPy is optional acceleration for the low-memory solver; ordinary GPU training
does not require it. Install one binary distribution matching your CUDA family:

```bash
python -m pip install "cupy-cuda12x[ctk]"
# Or, for CUDA 13:
python -m pip install "cupy-cuda13x[ctk]"
```

In uv, use `uv add` in place of `python -m pip install`. The `[ctk]` extra installs
CUDA runtime/compiler components as Python wheels, so a system-wide CUDA toolkit
is not needed. A compatible NVIDIA driver is still required. See the
[CuPy installation guide](https://docs.cupy.dev/en/stable/install.html).

Do not install `cupy`, `cupy-cuda12x` and `cupy-cuda13x` together: they provide
the same Python module. In particular, do not add `franken[cuda]` on top of a
binary CuPy installation. The legacy `cuda` extra requests the source package
`cupy`; it requires a working local CUDA toolkit and compiler. Its dependency
constraints are checked in CI, but its source compilation is not.

## Enable cuEquivariance for inference

After installing the matching CUDA build and `cueq-cu12` or `cueq-cu13` extra,
request cuEquivariance explicitly when loading a fitted MACE model:

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
```

Conversion uses unmodified upstream MACE, enables convolution fusion, and
preserves the fitted random features, coefficients and scaler. Training always
uses e3nn; cuEquivariance is inference-only. The first evaluation can compile
kernels, so benchmark after warmup. Smaller systems may be faster with e3nn.

Choose one CUDA variant matching PyTorch. Native cuEquivariance 0.10 kernels were
verified on Franklin's A100 with PyTorch 2.10/CUDA 12.6; the tested wheel
fails on Volta/V100. Use e3nn on V100. Pass `use_cueq=False` when loading to override a saved cuEquivariance choice.
Backend switching applies to eager checkpoints, not existing TorchScript archives.

## Check a training run

Pretrained checkpoints download on first use, which requires network access.
Run this small installation check with the bundled test data:

```bash
franken.autotune \
    --dataset-name test \
    --backbone mace --mace.path-or-id mace_mp/small \
    --rf gaussian --gaussian.num-rf 16 --gaussian.length-scale 1 \
    --force-weight 0.5 --l2-penalty 0.0001 \
    --jac-chunk-size 2 --run-dir installation-check
```

In uv, use `uv run franken.autotune ...`. The check writes a fitted checkpoint
under `installation-check`; the tiny dataset is not an accuracy benchmark.
Use `python -m pip check` or `uv pip check` to check installed dependencies.

## Conda environments

Conda can provide Python while pip installs Franken and its extras:

```bash
conda create -n franken python=3.13 pip
conda activate franken
python -m pip install "franken[metatomic,torch-sim]"
```

Follow the same pip recipes for selecting a PyTorch build. Conda environment
creation is not part of the GitHub installation workflow.
