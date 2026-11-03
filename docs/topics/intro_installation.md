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

Enable the backend explicitly in a MACE training command:

```bash
franken.autotune \
    --train-path train.xyz --val-path val.xyz \
    --backbone mace --mace.path-or-id mace_mp/small --mace.use-cueq \
    --rf gaussian --gaussian.num-rf 4096 --gaussian.length-scale 1 \
    --force-weight 0.99 --l2-penalty 0.0001
```

The Python equivalent is `MaceBackboneConfig("mace_mp/small", use_cueq=True)`.
The saved Franken checkpoint retains this choice, so reloading it requires the
same optional dependencies and a CUDA GPU. Leave `use_cueq` at its default of
`False` for a model that can also run on CPUs.

Native cuEquivariance kernels do not supply the forward-mode derivatives used
by Franken's usual feature-map calculation. For this backend, Franken computes
those derivatives with ordinary autograd, using double backward when there
are many features. It does not modify upstream kernels or silently accept zero
forward tangents. Convolution fusion stays disabled; native symmetric
contractions are enabled. The first GPU evaluation may compile kernels and take several seconds.
This path can be slower than the default backend,
so benchmark your training workload before enabling it. `--jac-chunk-size`
controls the default forward-mode backend, rather than this autograd path.

A missing or incompatible native extension raises an error when this backend
is requested. If you do not need cuEquivariance, omit the flag and use the
standard MACE backend.

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
