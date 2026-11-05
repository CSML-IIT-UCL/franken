# Install Franken with a coding agent

You are helping the user install Franken (version 0.9 and later) and verify that it can train a small
model. Carry out the setup in their workspace. Ask the user what whether he/she wants to add support for [PET/UPET](https://lab-cosmo.github.io/upet/) backbones, or the default MACE is sufficient. Then ask them whether they want to support fast-inference with cuEquivariance, which is especially useful for systems with many atoms (more than 1000), and whether they want to support also inference throuh [Metatomic](https://docs.metatensor.org/) or [TorchSim](https://torchsim.github.io/torch-sim/). These informations will be useful to understand which extras should be included with the installation.

Do not just return installation commands: run the
appropriate commands, inspect the results, and report what works.

This guide accompanies the human installation page at
`topics/intro_installation.html`, relative to this guide's URL. For a checkout,
its `pyproject.toml` and `.github/workflows/installation.yaml` describe the
version being installed. Do not assume an unreleased checkout is already on PyPI.

## 1. Inspect the environment and choose the smallest useful setup

Check the operating system, available Python versions, project directory, and
whether the user already has a virtual environment or uv project. Given the nature of this package, it is likely that the user is trying to install this package on an HPC system. So, try to explicitly identify whether this is the case, and gather additional context from the user, if needed. Franken requires Python 3.11 or newer; Python 3.13 is a suitable choice for a new project.
Inspect an existing `pyproject.toml`, `uv.lock`, or requirements file before
changing its dependencies.

- Reuse the environment or package manager the user selected. Otherwise use an
  isolated `.venv`; use uv for an existing uv project, or pip for a plain Python
  environment. Do not install into the system Python.
- Default to Franken with MACE. Add other backbones or tools only if asked by the user.
- Keep an existing working PyTorch build when it satisfies the selected extras.
  For a new CPU-only setup, the explicit CPU recipe below avoids downloading CUDA
  libraries. For requested GPU use, inspect `nvidia-smi` and the driver before
  choosing a CUDA build. Do not change drivers or interrupt existing GPU jobs.
- On a cluster, follow its module and scheduler instructions. Run computation
  inside an allocation, not on a login node. If no allocation is available,
  finish the installation checks and report GPU execution as pending.

Examples below use POSIX shell paths. Adapt activation and executable paths for
Windows. Activate the selected environment before using `python` or
`franken.autotune`; in uv projects use `uv run python` and `uv run franken.autotune`.

## 2. Install the base package

### pip

For a new environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install franken
```

### uv

For a new project:

```bash
uv init --bare --python 3.13 my-franken-project
cd my-franken-project
uv add franken
```

In an existing uv project, run only `uv add franken`. Make sure that you install franken version 0.9.0 or later. These instructions are not suited for older versions.

If the user asked to install a particular checkout, install that checkout with
`python -m pip install /path/to/franken` or `uv add /path/to/franken` instead.
Do not silently switch a release installation to an unrelated branch or fork.

## 3. Select only the extras the user needs

Use these requirement names with `python -m pip install` or `uv add`:

| Requirement | Purpose |
|---|---|
| `franken[metatomic]` | PET/UPET backbones through Metatomic and Metatrain |
| `franken[torch-sim]` | TorchSim interface dependencies |
| `franken[cueq-cu12]` | Optional cuEquivariance inference with CUDA 12 |
| `franken[cueq-cu13]` | Optional cuEquivariance inference with CUDA 13 |
| `franken[cupy-cu12]` | CuPy binary wheels and CUDA 12 components for the low-memory solver |
| `franken[cupy-cu13]` | CuPy binary wheels and CUDA 13 components for the low-memory solver |

Combine extras in one requirement when useful:

```bash
python -m pip install "franken[metatomic,torch-sim]"
# Or in a uv project:
uv add "franken[metatomic,torch-sim]"
```

Metatomic can require an older PyTorch than the newest release. Let the resolver
choose a compatible version. If an existing uv project acquired an unnecessarily
high lower bound from `uv add torch`, change that requirement to `torch>=2.5`
when appropriate for the user's project:

```bash
uv add "torch>=2.5" "franken[metatomic]"
```

Do not loosen a deliberately required version without understanding the other
project dependencies. Use a separate environment when requirements conflict.
MACE 0.3.17 requires e3nn 0.4.4. Do not override upstream dependency constraints,
use `--no-deps` to conceal conflicts, or maintain a patched MACE/cuEquivariance.

## 4. Choose an explicit CPU or CUDA build when needed

These recipes target Linux x86-64. For other platforms, consult the
PyTorch installer at https://pytorch.org/get-started/locally/ rather than guessing
wheel names. A CUDA wheel installation alone does not prove GPU execution works.

### pip recipes

CPU-only, with optional PET/TorchSim support (use just `franken` for MACE alone):

```bash
python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu \
    "franken[metatomic,torch-sim]"
```

CUDA 12.8 with optional cuEquivariance and binary CuPy:

```bash
python -m pip install --extra-index-url https://download.pytorch.org/whl/cu128 \
    "torch==2.11.0+cu128" "franken[cueq-cu12,cupy-cu12]"
```

CUDA 13.0:

```bash
python -m pip install --extra-index-url https://download.pytorch.org/whl/cu130 \
    "torch==2.14.1+cu130" "franken[cueq-cu13,cupy-cu13]"
```

The explicit PyTorch build tags in these pip recipes prevent backtracking
from silently choosing a different CUDA family. Different indexes publish
different version sets. Verify availability and compatibility before substituting
versions. Omit the CuPy extra if the user does not need low-memory solver acceleration;
omit the cuEquivariance extra if native inference was not requested.

### uv recipes

To select a particular build, merge the following into the uv project's
`pyproject.toml`, avoiding duplicate tables:

```toml
[[tool.uv.index]]
name = "pytorch"
url = "https://download.pytorch.org/whl/cpu"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch" }
```

For CPU-only use:

```bash
uv add "torch>=2.5" "franken[metatomic,torch-sim]"
```

For CUDA 12.8, change the URL to `https://download.pytorch.org/whl/cu128` and run:

```bash
uv add "torch>=2.5" "franken[cueq-cu12,cupy-cu12]"
```

For CUDA 13.0, use `https://download.pytorch.org/whl/cu130` and run:

```bash
uv add "torch>=2.5" "franken[cueq-cu13,cupy-cu13]"
```

The explicit index applies only to PyTorch; other packages come from PyPI.
Keep this configuration when adding extras later. See
https://docs.astral.sh/uv/guides/integration/pytorch/ for the upstream explanation.

### GPU-specific checks

- Check `torch.__version__`, `torch.version.cuda`, and
  `torch.cuda.is_available()` in the selected environment. Confirm the build
  matches the requested CUDA family. If GPU use was requested, verify a small
  GPU operation inside an appropriate allocation before reporting GPU success.
- Select one CUDA family for both CuPy and cuEquivariance extras. Install only
  one CuPy distribution: `cupy-cuda12x` and `cupy-cuda13x` provide the same module.
  Franken's `cupy-cu12` and `cupy-cu13` extras install the corresponding binary
  distribution with `[ctk]`, which installs CUDA components
  as Python wheels, but still require a compatible NVIDIA driver.
- cuEquivariance is **inference-only**. Training and training feature maps must
  use e3nn. Select native inference explicitly on a compatible fitted checkpoint
  with `FrankenCalculator(..., device="cuda", use_cueq=True)` or
  `FrankenPotential.load(..., map_location="cuda", use_cueq=True)`. Check that the
  installed Franken version exposes this API before using it.
- Native cuEquivariance 0.10 was tested on A100 with PyTorch 2.10/CUDA 12.6. The
  tested wheel has no Volta/V100 kernels. Do not enable it on V100. The first
  inference call may compile kernels; small configurations may be faster with
  e3nn. Do not promise every CUDA installation is accelerated.

## 5. Verify a real installation, including a training run

First check installed dependencies and imports:

```bash
python -m pip check
python -c "import franken, mace, torch; print(torch.__version__, torch.version.cuda)"
```

For uv, use `uv pip check` and prefix Python commands with `uv run`. Also import
any requested interfaces, such as `metatomic.torch`, `metatrain`, `torch_sim`,
`cuequivariance_torch`, or `cupy`. Do not install pip into a uv environment just
for the dependency check.

Run the following Python snippet with the environment's Python (or
`uv run python`). It executes the same tiny MACE autotune check used by
installation CI, forces CPU use to avoid occupying an unrelated GPU, verifies a
checkpoint was produced, and removes its temporary output:

```python
import os
from pathlib import Path
import subprocess
import sys
import tempfile

cli = Path(sys.executable).with_name(
    "franken.autotune.exe" if os.name == "nt" else "franken.autotune"
)
with tempfile.TemporaryDirectory(prefix="franken-install-check-") as directory:
    subprocess.run(
        [
            str(cli), "--dataset-name", "test",
            "--backbone", "mace", "--mace.path-or-id", "mace_mp/small",
            "--rf", "gaussian", "--gaussian.num-rf", "16",
            "--gaussian.length-scale", "1", "--force-weight", "0.5",
            "--l2-penalty", "0.0001", "--jac-chunk-size", "2",
            "--run-dir", directory,
        ],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        check=True,
    )
    assert any(Path(directory).rglob("best_ckpt.pt")), "No fitted checkpoint produced"
print("Franken installation and MACE training check passed.")
```

The registered MACE checkpoint downloads on first use. A failed download is not
proof of a broken package: inspect the HTTP/network error and retry transient
failures. Do not disable TLS checks or substitute an unrelated backbone to hide
the problem. If network access blocks the training check, report installation
and training verification separately. The tiny dataset is an installation
check, not evidence of a physically accurate model.

## 6. Finish with a short, concrete report

Tell the user:

- Which environment/project and package manager to use, and the exact command
  for running Franken next time.
- Installed Franken, MACE, PyTorch and selected extra versions, including the
  CPU/CUDA build and whether GPU execution was actually verified.
- Whether dependency checks, interface imports and the training check passed;
  explain any remaining blocker accurately.

Keep the working environment and useful model cache. Remove only temporary or
failed setup environments and diagnostic files you created; preserve the user's
files, checkpoints, settings and running jobs.
