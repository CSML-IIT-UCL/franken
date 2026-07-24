# Minimal Example

## Dataset format and Measurement Units

When parsing `.xyz` or `.extxyz` files (such as `train.xyz` and `val.xyz` below), Franken inherits the default [ASE (Atomic Simulation Environment) standard for units](https://docs.ase-lib.org/ase/units.html). Ensure your data adheres to the following:
* **Energy**: `eV`
* **Forces**: `eV/Å`
* **Stress**: `eV/Å³` (Note: if your simulation engine outputs stress in `kbar` -- such as VASP -- you must manually convert it before passing the dataset to Franken. The conversion factor is `1 kbar` $\approx 1/1602.176634$ `eV/Å³`).

### Train 

Franken models can be easily trained using the autotune CLI tool:
```bash
franken.autotune \
    --train-path train.xyz --val-path val.xyz \
    --backbone=mace --mace.path-or-id "mace_mp/small" \
    --rf=ms-gaussian --ms-gaussian.num-rf 8192 --ms-gaussian.length-scale-num 5\
    --ms-gaussian.length-scale-low 1  --ms-gaussian.length-scale-high 32 \
```

This will create a folder `run_DATE_TIME_...` containing:
* `best_ckpt.pt`  -->  model checkpoint
* `best.json`  -->  train/val/test metrics for the best model
* `config.json`  -->  training configuration
* `log.json`  -->  metrics for all tested models (hyperparameter optimization)

Below is an example `best.json` file, which contains info about the **metrics**, **timings**, and **hyperparameters**.

```
{
    "checkpoint": {
        "hash": "83d606c2b5bb7f878c353a7b0f43dd38",
        "rf_weight_id": 0
    },
    "timings": {
        "cov_coeffs": 300.4266714300029,
        "solve": 0.0742890159599483
    },
    "metrics": {
        "train": {
            "energy_MAE": 0.2496591668855935,
            "forces_MAE": 15.919811367988586,
            "energy_RMSE": 0.31621758133552114,
            "forces_RMSE": 20.368873955871486,
        },
        "validation": {
            "energy_MAE": 0.23749234564490368,
            "forces_MAE": 16.388884401321413,
            "energy_RMSE": 0.31755570717758935,
            "forces_RMSE": 20.917586551063245,
        }
    },
    "hyperparameters": {
        "franken": {
            "path_or_id": "mace_mp/small",
            "interaction_block": 2,
            "family": "mace"
        },
        "random_features": {
            "rf_type": "multiscale-gaussian",
            "num_random_features": 4096,
            "length_scale_low": 1.0,
            "length_scale_high": 32.0,
            "length_scale_num": 5,
            "use_offset": "True",
            "rng_seed": 1337
        },
        "input_scaler": {
            "scale_by_Z": true
        },
        "solver": {
            "force_weight": 0.99,
            "l2_penalty": 1e-10,
            "dtype": "torch.float64"
        }
    }
}
```

See the *Franken CLI Reference > [Autotune](https://franken.readthedocs.io/reference/franken-cli/franken.autotune.html)* page to view the description of the options and the dedicated [tutorial](https://franken.readthedocs.io/notebooks/autotune.html).

### Predict 

The trained model can be used as a ASE (Atomistic Simulations Environment) calculator for easy inference. 

```python
from franken.calculators import FrankenCalculator
calc = FrankenCalculator('best_model.ckpt', device='cuda:0')

# calculate
calc.calculate(atoms)
# or attach it:
atoms.calc = calc
```

See the [MD tutorial](https://franken.readthedocs.io/notebooks/molecular_dynamics.html) for a complete example about running molecular dynamics, while for deploying it to LAMMPS see the dedicated [page](https://franken.readthedocs.io/topics/lammps.html).
