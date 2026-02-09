# Franken

[![Test status](https://github.com/CSML-IIT-UCL/franken/actions/workflows/CI.yaml/badge.svg)](https://github.com/CSML-IIT-UCL/franken/actions/workflows/CI.yaml)
[![Docs status](https://github.com/CSML-IIT-UCL/franken/actions/workflows/rtd.yaml/badge.svg)](https://franken.readthedocs.io/)


## Introduction

Franken is an open-source library that can be used to enhance the accuracy of atomistic foundation models. It can be used for molecular dynamics simulations, and has a focus on computational efficiency.

`franken` features include:
 - Supports fine-tuning for a variety of foundation models ([MACE](https://github.com/ACEsuit/mace), [SevenNet](https://github.com/MDIL-SNU/SevenNet), [SchNet](https://github.com/facebookresearch/fairchem))
 - Automatic [hyperparameter tuning](https://franken.readthedocs.io/notebooks/autotune.html) simplifies the adaptation procedure, for an out-of-the-box user experience.
 - Several random-feature approximations to common kernels (e.g. Gaussian, polynomial) are available to flexibly fine-tune any foundation model.
 - Support for running within [LAMMPS](https://www.lammps.org/) molecular dynamics, as well as with [ASE](https://wiki.fysik.dtu.dk/ase/).

<img src="/docs/_static/diagram_part1.png" alt="Franken diagram" width="1000px">

For detailed information and benchmarks please check our paper [*Fast and Fourier Features for Transfer Learning of Interatomic Potentials*](https://arxiv.org/abs/2505.05652).

## Documentation

A full documentation including several examples is available: [https://franken.readthedocs.io/index.html](https://franken.readthedocs.io/index.html). [The paper](https://arxiv.org/abs/2505.05652) also contains a comprehensive description of the methods behind franken.

## Install

To install the latest release of `franken`, you can simply do:

```bash
pip install franken
```

Several optional dependencies can be specified, to install packages required for certain operations:
 - `cuda` includes packages which speed up training on GPUs (note that `franken` will work on GPUs even without these dependencies thanks to pytorch).
 - `fairchem`, `mace`, `sevenn` install the necessary dependencies to use a specific backbone. Note that Fairchem v2 introduced breaking changes, so use v1 for SchNet support.
 - `docs` and `develop` are only needed if you wish to build the documentation, or work on extending the library.

They can be installed for example by running

```bash
pip install franken[mace,cuda]
```

For more details read the [relevant documentation page](https://franken.readthedocs.io/topics/installation.html)

## Quickstart

### Train
You can directly run `franken.autotune` to get started with the `franken` library. 

```bash
franken.autotune \
    --train-path train.xyz \
    --val-path val.xyz \
    --backbone=mace --mace.path-or-id "mace_mp/small" --mace.interaction-block 2 \
    --rf=ms-gaussian --ms-gaussian.num-rf 4096 --ms-gaussian.length-scale-num 5\
    --ms-gaussian.length-scale-low 1  --ms-gaussian.length-scale-high 32 \
    --force-weight=0.99 \
    --l2-penalty="(-10, -6, 5, log)" \
    --jac-chunk-size "auto"
```

For more details you can check out the [autotune tutorial](https://franken.readthedocs.io/notebooks/autotune.html) or the [getting started notebook](https://franken.readthedocs.io/notebooks/getting_started.html).

### Inference/MD
The trained model can be used as a ASE (Atomistic Simulations Environment) calculator for easy inference. 

```python
from franken.calculators import FrankenCalculator
calc = FrankenCalculator('best_model.ckpt', device='cuda:0')
atoms.calc = calc
```

See the [MD tutorial](.https://franken.readthedocs.io/notebooks/molecular_dynamics.html) for a complete example about running molecular dynamics, while for deploying it to LAMMPS see the dedicated [page](https://franken.readthedocs.io/topics/lammps.html).

## Citing

If you find this library useful, please cite our work using the folowing bibtex entry:
```
@article{novelli2025fast,
  title={Fast and Fourier features for transfer learning of interatomic potentials},
  author={Novelli, Pietro and Meanti, Giacomo and Buigues, Pedro J and Rosasco, Lorenzo and Parrinello, Michele and Pontil, Massimiliano and Bonati, Luigi},
  journal={npj Computational Materials},
  volume={11},
  number={1},
  pages={293},
  year={2025},
  publisher={Nature Publishing Group UK London}
}
```
