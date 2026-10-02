(installation)=
# Installation

> ⚠️ **Important:** `franken` requires **PyTorch** to be installed in your environment *before* you proceed. Running the installation commands below without PyTorch will result in a build or installation error. If you haven't installed it yet, please follow the [Official PyTorch Installation Guide](https://pytorch.org/get-started/locally/#linux-pip) to set up the correct version for your specific hardware (CPU or GPU).

### 1. Standard Installation (Bare-bones)
The basic installation comes without any GNN backbone installed. You can install it by running:
```bash
pip install franken
```

### 2. Installation with GNN Backbones (Recommended for GPUs)
If you wish to use GPUs and need specific backbones, you can install `franken` directly with the required extras:
```bash
# Install with MACE backbone
pip install franken[cuda,mace]

# Install with PET backbone
pip install franken[cuda,pet]
```

### 3. Example Environment Setup (Conda)
If you are setting up a clean GPU environment from scratch, here is a recommended configuration pipeline using Conda and CuPy:
```bash
# 1. Prepare environment utilities
conda install pip setuptools wheel 

# 2. Install PyTorch with CUDA 12.1 (adjust according to your system)
conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia 

# 3. Install CuPy (required for GPU/CUDA backbones)
conda install -c conda-forge cupy    

# 4. Install franken with your preferred backbone
pip install franken[cuda,mace]
```

In more detail:
 - the `cuda` qualifier installs dependencies which are only relevant on GPU-enabled environments and can be omitted.
 - the supported backbones are [MACE](https://github.com/ACEsuit/mace) and [UPET](https://github.com/lab-cosmo/upet/tree/main). They are explained in more detail below.


```{warning}
Each backbone seems to have mutually incompatible requirements, particularly with regards to `e3nn` - but also pytorch versions might be a problem.
To minimize incompatibilities, we suggest that the users who wishes to use multiple backbones create independent python environments for each.
In particular, the `mace-torch` package requires an old version of `e3nn` (0.4.4) which may conflict with other backbones. If you encounter errors with model loading, simply upgrade `e3nn` by running `pip install -U e3nn`.
```

## Supported pre-trained models
### MACE
We support several models which use the [MACE architecture](https://github.com/ACEsuit/mace):
 - The [`MACE-MP0`](https://arxiv.org/abs/2401.00096) models trained on the materials project data by Batatia et al. Additional informations on the pre-training of `MACE-MP0` are available on its [HuggingFace model card](https://huggingface.co/cyrusyc/mace-universal).
 - The MACE-OFF ([paper](https://github.com/ACEsuit/mace-off) and [github](https://github.com/ACEsuit/mace-off)) models which are pretrained on organic molecules.
 - The Egret ([github](https://github.com/rowansci/egret-public)) family of models (`Egret-1`, `Egret-1e`, `Egret-1t`), also tuned for organic molecules.

To use any MACE model as a backbone for `franken` just `pip`-install `mace-torch` in `franken`'s environment
```bash
pip install mace-torch
```
or directly install franken with mace support (`pip install franken[cuda,mace]`).

In addition to MACE-MP0 trained on the materials project dataset, Franken also supports the [`MACE-OFF` models](https://arxiv.org/abs/2312.15211) for organic chemistry.


### PET

Franken supports [UPET](https://github.com/lab-cosmo/upet/tree/main) models through the Metatomic/Metatrain ecosystem.
To use PET models as a backbone for `franken`, install the required dependencies with `pip install franken[cuda,pet]` or follow the instructions on [metatomic](https://docs.metatensor.org/metatomic/latest/installation.html) and [metatrain](https://docs.metatensor.org/metatrain/latest/installation.html) documentation.
