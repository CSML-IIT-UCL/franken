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