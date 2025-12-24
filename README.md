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

You can directly run `franken.autotune` to get started with the `franken` library. A quick example is to fine-tune MACE-MP0 on a high-level-of-theory water dataset:

```bash
franken.autotune \
    --dataset-name="water" --max-train-samples=8 \
    --l2-penalty="(-10, -5, 5, log)" \
    --force-weight="(0.01, 0.99, 5, linear)" \
    --seed=42 \
    --jac-chunk-size=64 \
    --run-dir="./results" \
    --backbone=mace --mace.path-or-id="mace_mp/small" --mace.interaction-block=2 \
    --rf=gaussian --gaussian.num-rf=512 --gaussian.length-scale="[10.0, 15.0]"
```

For more details you can check out the [autotune tutorial](https://franken.readthedocs.io/notebooks/autotune.html) or the [getting started notebook](https://franken.readthedocs.io/notebooks/getting_started.html).


## Citing

If you find this library useful, please cite our work using the folowing bibtex entry:
```
@misc{novelli25franken,
    title={Fast and Fourier Features for Transfer Learning of Interatomic Potentials},
    author={Pietro Novelli and Giacomo Meanti and Pedro J. Buigues and Lorenzo Rosasco and Michele Parrinello and Massimiliano Pontil and Luigi Bonati},
    year={2025},
    eprint={2505.05652},
    archivePrefix={arXiv},
    url={https://arxiv.org/abs/2505.05652}
}
```
