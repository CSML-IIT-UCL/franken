(installation)=
# Installation

To install `franken`, start by setting up your environment with the correct **version of [PyTorch](https://pytorch.org/)**. This is especially necessary if you wish to use GPUs. Then install `franken` by running
```bash
pip install franken
```
The basic installation comes bare-bones without any GNN backbone installed. You can install franken with a specific backbone directly, by running one of the following commands
```bash
pip install franken[cuda,mace]
pip install franken[cuda,fairchem]
pip install franken[cuda,sevenn]
```
In more detail:
 - the `cuda` qualifier installs dependencies which are only relevant on GPU-enabled environments and can be omitted.
 - the three supported backbones are [MACE](https://github.com/ACEsuit/mace), [SchNet from fairchem](https://github.com/FAIR-Chem/fairchem), and [SevenNet](https://github.com/MDIL-SNU/SevenNet). They are explained in more detail below.


```{warning}
Each backbone seems to have mutually incompatible requirements, particularly with regards to `e3nn` - but also pytorch versions might be a problem.
To minimize incompatibilities, we suggest that the users who wishes to use multiple backbones create independent python environments for each.
In particular, the `mace-torch` package requires an old version of `e3nn` (0.4.4) which conflicts with `fairchem-core`, see [this relevant issue](https://github.com/ACEsuit/mace/issues/555) and with `SevenNet`. If you encounter errors with model loading, simply upgrade `e3nn` by running `pip install -U e3nn`.
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


### SevenNet

Franken also supports the [SevenNet model](https://arxiv.org/abs/2402.03789) by Park et al. as implemented in the [`sevennet`](https://github.com/MDIL-SNU/SevenNet) library.
We have only tested the SevenNet-0 model trained on the materials project dataset, but support for other models should be possible (open an issue if you encounter any problem).

### SchNet OC20 (fairchem, formerly OCP)
We support the [SchNet model](https://arxiv.org/abs/1706.08566) by Schütt et al. as implemented in the [`fairchem`](https://fair-chem.github.io/) library by Meta's FAIR. The pre-training was done on the [Open Catalyst dataset](https://fair-chem.github.io/core/datasets/oc20.html). To use it as a backbone for `franken`, install the `fairchem` library
```bash
pip install fairchem-core
```
and the `torch_geometric` dependencies as explained in the [FairChem docs](https://fair-chem.github.io/core/install.html).
```{note}
Not all of fairchem's dependencies can be installed by `pip` alone, check the [FairChem docs](https://fair-chem.github.io/core/install.html).
```
Note that `SchNet` is not competitive with more recent GNN models and is only meant as a baseline, and to showcase support for diverse backends.
For now we do not support fairchem v2 models, if you wish to see this implemented please file an issue!