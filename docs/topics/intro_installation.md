(installation)=
# Installation

> ⚠️ **Important:** `franken` requires **PyTorch** to be installed in your environment *before* you proceed. Running the installation commands below without PyTorch will result in a build or installation error. If you haven't installed it yet, please follow the [Official PyTorch Installation Guide](https://pytorch.org/get-started/locally/#linux-pip) to set up the correct version for your specific hardware (CPU or GPU).

The basic installation comes without any GNN backbone installed:
```bash
pip install franken
```

To enable CUDA support (highly-recommended):
```bash
pip install franken[cuda]
```

The currently supported backbones are [MACE](https://github.com/ACEsuit/mace) and [UPET](https://github.com/lab-cosmo/upet/tree/main). If you don't have them, you can use install them along with `franken` with the required extras:

```bash
# Install with MACE backbone
pip install franken[cuda,mace]

# Install with PET backbone
pip install franken[cuda,pet]
```

**Example Environment Setup (Conda)**
If you are setting up a clean GPU environment from scratch, you can also install it with conda:

```bash
# 1. Prepare environment utilities
conda install pip setuptools wheel 

# 2. Install PyTorch with CUDA 12.6 (adjust according to your system)
conda install pytorch pytorch-cuda=12.6 -c pytorch -c nvidia 

# 3. Install CuPy (required for GPU/CUDA backbones)
conda install -c conda-forge cupy    

# 4. Install franken with your preferred backbone
pip install franken[cuda,mace]
```

### Supported pre-trained models

#### MACE
To use a MACE model as a backbone for `franken` just `pip`-install `mace-torch` in `franken`'s environment
```bash
pip install mace-torch
```
further details can be found in the [MACE documentation](https://mace-docs.readthedocs.io/en/latest/guide/installation.html).

#### PET

The [UPET](https://github.com/lab-cosmo/upet/tree/main) models are available through the Metatomic/Metatrain ecosystem.
To use PET models as a backbone for `franken`, you can follow the instructions on [metatomic](https://docs.metatensor.org/metatomic/latest/installation.html) and [metatrain](https://docs.metatensor.org/metatrain/latest/installation.html) documentation, or just install them with franken with `pip install franken[cuda,pet]`.


```{warning}
Each backbone seems to have mutually incompatible requirements, particularly with regards to `e3nn` - but also pytorch versions might be a problem.
To minimize incompatibilities, we suggest that the users who wishes to use multiple backbones create independent python environments for each.
In particular, the `mace-torch` package requires an old version of `e3nn` (0.4.4) which may conflict with other backbones. If you encounter errors with model loading, simply upgrade `e3nn` by running `pip install -U e3nn`.
```