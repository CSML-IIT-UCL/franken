(installation)=
# Installation

```{warning}
It is better to install **PyTorch** first. If you haven't installed it yet, please follow the [Official PyTorch Installation Guide](https://pytorch.org/get-started/locally/#linux-pip).
```

### Minimal installation

The basic installation of *franken* comes without any GNN backbone installed:
```bash
pip install franken
```

To enable CUDA support (highly-recommended):
```bash
pip install franken[cuda]
```

### Install Franken+backbones (recommended)

The currently supported backbones are [MACE](https://github.com/ACEsuit/mace) and [UPET](https://github.com/lab-cosmo/upet/tree/main). You can install them alongside `franken` using the corresponding extras.

#### MACE

To install `franken` with CUDA support and the MACE backbone:

```bash
pip install franken[cuda,mace]
```

If you already have `franken` installed, you can add MACE by installing `mace-torch` in the same environment:

```bash
pip install mace-torch
```

Further details can be found in the [MACE documentation](https://mace-docs.readthedocs.io/en/latest/guide/installation.html).
 
#### PET

The [UPET](https://github.com/lab-cosmo/upet/tree/main) models are available through the Metatomic/Metatrain ecosystem. To install `franken` with CUDA support and the PET backbone:

```bash
pip install franken[cuda,pet]
```

For further installation details, see the [Metatomic](https://docs.metatensor.org/metatomic/latest/installation.html) and [Metatrain](https://docs.metatensor.org/metatrain/latest/installation.html) documentation.

```{warning}
The backbones may have incompatible requirements, particularly for `e3nn` and PyTorch. If you wish to use multiple backbones, we recommend creating a separate Python environment for each.

In particular, `mace-torch` requires an older version of `e3nn` (0.4.4), which may conflict with other backbones. If you encounter model-loading errors, check the required `e3nn` version for your chosen backbone before changing it.
```

### Example Environment Setup (Conda)

If you are setting up a clean GPU environment from scratch, you can use Conda to install the dependencies:

```bash
# 1. Prepare environment utilities
conda install pip setuptools wheel

# 2. Install PyTorch with CUDA 12.6 (adjust according to your system)
conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia

# 3. Install CuPy (required for GPU/CUDA backbones)
conda install -c conda-forge cupy

# 4. Install franken with your preferred backbone
pip install franken[cuda,mace]
```