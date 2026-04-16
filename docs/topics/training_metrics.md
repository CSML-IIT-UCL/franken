# Metrics

This page summarizes:
- the objective optimized by **franken** during fitting
- which evaluation metrics are available
- how autotune selects the best model.

## Training Objective (Loss)

During fitting, franken solves a weighted least-squares problem combining mean square error (MSE) on energy and forces:

$$
\mathcal{L}(\mathbf{w}) =
(1-\alpha)\,\mathrm{MSE}\!\left(E(\mathbf{w}), E^{\star}\right)
\;+\;
\alpha\,\mathrm{MSE}\!\left(F(\mathbf{w}), F^{\star}\right)
$$

where the hyperparameter  `force_weight` (between 0 and 1) controls the relative contribution. In addition, there is also a ridge regularization with weight `l2_penalty`.

## Evaluation Metrics

The following metrics can be calculated: 
* `energy_MAE`, `energy_RMSE`, `forces_MAE`, `forces_RMSE`, `forces_cosim`, `forces_MAE_species`, `forces_RMSE_species`.

They can be customized using the autotune configuration: 

```bash
franken.autotune \
  ... \
  --metrics energy_MAE forces_MAE forces_MAE_species \
  --best-model-selection energy_MAE forces_MAE
```

```python
from franken.config import AutotuneConfig

cfg = AutotuneConfig(
    ...,
    metrics=["energy_MAE", "forces_MAE", "forces_MAE_species"],
    best_model_selection=["energy_MAE", "forces_MAE"]
)
```

When using species-resolved metrics (`forces_MAE_species`, `forces_RMSE_species`), the logs do not store a single scalar, but rather a value per each atomic number `Z` (`..._<Z>`) and the average of the metric per species (`..._average`)

## Best-Model Selection in Autotune

Autotune chooses the best model from the metrics in `best_model_selection` by (i) building the Pareto frontier using a list of metrics and (ii) among Pareto-efficient models, minimizing the p-norm (`p=1`).

The metric(s) which are used to perfom the selection can be customized in the CLI using `--best-model-selection` (`best_model_selection` for the APIs).


Notes:
- This affects model ranking only (`best.json` / `best_ckpt.pt`), not the training loss.
- To use species-resolved metrics, use `*_average` (or an explicit `*_Z`) in `best_model_selection`.
