# Stress

Franken can fit periodic structures to energy, forces, and stress simultaneously. 
Each training and validation frame must contain a periodic cell and an ASE-readable
stress label. Franken uses the ASE stress convention and units of eV/Å³.

Enable stress as a training target and assign its loss weight with `stress_weight`:

```python
from franken.config import AutotuneConfig, SolverConfig

cfg = AutotuneConfig(
    ...,
    train_targets=["energy", "forces", "stress"],
    solver=SolverConfig(
        energy_weight=1.0,
        force_weight=10.0,
        stress_weight=1.0,
    ),
    metrics=["energy_MAE", "forces_MAE", "stress_MAE"],
)
```

The equivalent CLI option is `--train-targets energy forces stress`. 

Notes:

* Stress fitting requires cell derivatives and consequently has a
higher memory and computational cost than energy-and-force fitting.
