import argparse
from copy import copy
import datetime
import json
import os
import pathlib
import pickle
import time
from typing import Any, Literal, Mapping, Sequence
import warnings
from git import Repo

import ase.build
import ase.md.velocitydistribution
import ase.units
import ase.io
import numpy as np
import torch

from franken.autotune import autotune
from franken.calculators.ase_calc import FrankenCalculator
from franken.config import (
    BackboneConfig, DatasetConfig, 
    MultiscaleGaussianRFConfig, SolverConfig, HPSearchConfig, 
    AutotuneConfig
)

from franken.datasets.registry import DATASET_REGISTRY
from franken.backbones.utils import CacheDir
from franken.rf.model import FrankenPotential


def training(
    gnn_config: BackboneConfig,
    dset: str,
    n_train_samples: int,  # 128
    n_rf: int,  # 8192
    seed: int,
):
    train_path = DATASET_REGISTRY.get_path(dset, "train", base_path=CacheDir.get())
    val_path = DATASET_REGISTRY.get_path(dset, "val", base_path=CacheDir.get())

    dataset_cfg = DatasetConfig(train_path=str(train_path),
                                max_train_samples=n_train_samples,
                                val_path=str(val_path)
    )
    print("Dataset configuration")
    print(dataset_cfg)

    rf_config = MultiscaleGaussianRFConfig(
        num_random_features=n_rf,
        length_scale_low=8,
        length_scale_high=32,
        length_scale_num=4,
        rng_seed=seed,
    )
    solver_cfg = SolverConfig(
        l2_penalty=HPSearchConfig(start=-12, stop=-5, num=8, scale='log'),  # equivalent of numpy.logspace
        force_weight=HPSearchConfig(start=0.01, stop=0.99, num=5, scale='linear'),  # equivalent of numpy.linspace
    )
    autotune_cfg = AutotuneConfig(
        dataset=dataset_cfg,
        solver=solver_cfg,
        backbone=gnn_config,
        rfs=rf_config,
        metrics=["energy_MAE", "forces_MAE"],
        seed=seed,
        jac_chunk_size='auto',
        run_dir="./results",
    )

    run_path = autotune(autotune_cfg)
    assert run_path is not None

    with open(run_path / "best.json", "r") as fh:
        best_log = json.load(fh)
    best_model_path = run_path / "best_ckpt.pt"
    best_model = FrankenPotential.load(best_model_path, map_location="cpu")
    best_info = {
        "l2_penalty": best_log["hyperparameters"]["solver"]["l2_penalty"],
        "force_weight": best_log["hyperparameters"]["solver"]["force_weight"],
        "forces_MAE": best_log['metrics']['validation']['forces_MAE'],
        "energy_MAE": best_log['metrics']['validation']['energy_MAE'],
        "train_time": best_log['timings']['cov_coeffs'],
    }
    return best_info, best_model


def get_md_data(data_type: Literal["water", "diamond"], **kwargs) -> ase.Atoms:
    if data_type == "diamond":
        num_primitive_reps = kwargs.pop("cell_reps", 3)
        primitive = ase.build.bulk(name="C", crystalstructure="diamond", a=3.567)
        atoms = ase.build.make_supercell(primitive, num_primitive_reps * np.eye(3))
    elif data_type == "water":
        val_path = DATASET_REGISTRY.get_path("water", "val", base_path=CacheDir.get())
        read_ase_atoms = ase.io.read(val_path, index=":")
        if isinstance(read_ase_atoms, ase.Atoms):
            # workaround edge case of a single configuration
            atoms = read_ase_atoms
        else:
            atoms = read_ase_atoms[0]
    else:
        raise ValueError(data_type)
    for key in kwargs.keys():
        warnings.warn(f"Data kwarg {key} was not used.")
    return atoms


def molecular_dynamics_ase(
    model: FrankenPotential, 
    atoms: ase.Atoms, 
    device,
    num_steps: int,
    seed: int,
):
    num_atoms = len(atoms)
    # 1. Ase calculator
    ase_calc = FrankenCalculator(
        franken_ckpt=model,
        device=device,
    )
    atoms.calc = ase_calc
    # 2. Setup MD
    rng = np.random.default_rng(seed=seed)
    ase.md.velocitydistribution.MaxwellBoltzmannDistribution(atoms, temperature_K=300)
    integrator = ase.md.Langevin(
        atoms,
        timestep=1.0 * ase.units.fs,
        temperature_K=300,
        friction=0.1 / ase.units.fs,
        rng=rng
    )
    # 3. Run MD (collect timings)
    times = []
    eval_every = 10
    for i in range(num_steps // eval_every):
        t_s = time.time()
        integrator.run(eval_every)  # run for 10 steps at a time
        t_e = time.time()
        if i > 3:  # warmup for time collection
            times.append(t_e - t_s)
        energy = atoms.get_total_energy()
        assert not np.any(np.isnan(energy))
    return {
        "md_time_per_atom": np.mean(times) / eval_every / num_atoms
    }


def add_prefix(d: Mapping, prefix: str | None) -> Mapping:
    if prefix is None:
        return d
    return {
        f"{prefix}_{k}": v for k, v in d.items()
    }


def check_db_has_config(db: Sequence[dict[str, Any]], config: dict[str, Any]) -> bool:
    for db_info in db:
        is_equal = True
        db_info_c = copy(db_info)
        db_info_c.pop("results")  # don't care about results!
        for new_k, new_v in config.items():
            try:
                db_info_v = db_info_c.pop(new_k)
                if new_v != db_info_v:
                    is_equal = False
                    break  # db element is not equal to new config
            except KeyError:
                is_equal = False
                break  # db element does not have new config key
        if len(db_info_c) != 0:
            is_equal = False  # db element has more keys than new config
        if is_equal:
            return True
    return False


def logtime():
    return datetime.datetime.now().isoformat()

def run(db_path):
    # 0. Options
    md_data_info = {
        "data_type": "diamond",
        "cell_reps": 5,  # 5^3 * 2 = 250 atoms
    }
    md_options = {
        "num_steps": 1000,
        "seed": 1,
    }
    train_options = {
        "n_train_samples": 128,
        "n_rf": 8192,
        "seed": 1,
        "dset": "water",
    }
    md_data = get_md_data(**md_data_info)  
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # 1. Define all gnn configs we're interested in
    gnn_ckpts = [
        {
            "family": "pet",
            "path_or_id": "PET_MAD/xs_1.5",
        },
        {
            "family": "mace",
            "path_or_id": "mace_mp/small",
            "interaction_block": 2,
        },
        {
            "family": "mace",
            "path_or_id": "mace_mp/medium",
            "interaction_block": 2,
        },
    ]
    repo = Repo(pathlib.Path(__file__).parent.resolve(), search_parent_directories=True)
    commit_hash = repo.git.rev_parse("HEAD")

    db = []
    if os.path.isfile(db_path):
        with open(db_path, "rb") as fh:
            db = pickle.load(fh)

    for gnn_ckpt in gnn_ckpts:
        train_info, franken_model = None, None
        gnn_config = BackboneConfig.from_ckpt(gnn_ckpt)
        for compile in ["jit", "compile", "none"]:
            # Check if an entry with the same options exists in db.
            key_info = {
                "commit": commit_hash,
                "device": device.type,
                "compile": compile,
                **add_prefix(gnn_ckpt, "gnn"),
                **add_prefix(md_data_info, "md_data"),
                **add_prefix(md_options, "md"),
                **add_prefix(train_options, "train"),
            }
            if check_db_has_config(db, key_info):
                print(f"Configuration for {gnn_ckpt['path_or_id']} already exists in database; skipping.")
                continue

            # Train (only once for all compile options)
            if train_info is None or franken_model is None:
                print(f"[{logtime()}] starting training of {gnn_config.path_or_id}")
                train_info, franken_model = training(
                    gnn_config=gnn_config,
                    **train_options
                )
            # 2. Define preprocessing steps for the franken-potential e.g. compile
            if compile == "jit":
                print(f"[{logtime()}] jit compiling {gnn_config.path_or_id}")
                franken_model = torch.jit.script(franken_model)
            elif compile == "compile":
                print(f"[{logtime()}] torch compiling {gnn_config.path_or_id}")
                franken_model = torch.compile(franken_model)
            print(f"[{logtime()}] starting MD for {gnn_config.path_or_id}")
            md_info = molecular_dynamics_ase(
                franken_model, md_data, device=device, **md_options
            )
            results_info = add_prefix(md_info | train_info, "results")
            all_info = key_info | results_info # type: ignore
            db.append(all_info)
            with open(db_path, "wb") as fh:
                pickle.dump(db, fh)
                fh.flush()
            print(f"[{logtime()}] Finished experiment MD...")
            print(all_info)
            print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path", type=str, required=True)

    args = parser.parse_args()
    print(f"Starting run with DB path: {args.db_path}")
    run(args.db_path)
