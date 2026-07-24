# Metrics

This page summarizes:
- the objective optimized by **franken** during fitting
- which evaluation metrics are available
- how autotune selects the best model.

## Training Objective (Loss)

During fitting, franken solves a weighted least-squares problem combining mean square error (MSE) on energy, forces and, optionally, stress:

$$
\mathcal{L}(\mathbf{w}) =
w_e\,\mathrm{MSE}\!\left(E(\mathbf{w}), E^{\star}\right)
\;+\;
w_f\,\mathrm{MSE}\!\left(F(\mathbf{w}), F^{\star}\right)
\;+\;
w_s\,\mathrm{MSE}\!\left(S(\mathbf{w}), S^{\star}\right)
$$

where $w_e$, $w_f$, and $w_s$ are the normalized weights corresponding to the hyperparameters `energy_weight`, `force_weight`, and `stress_weight` respectively, such that $w_e + w_f + w_s = 1$. By default, `energy_weight` is set to `1.0`, while `force_weight` and `stress_weight` (if stress target is enabled) are automatically tuned via a grid search on a logarithmic scale. These parameters are controlled via the `SolverConfig`/CLI. In addition, there is also a ridge regularization with weight `l2_penalty`.

## Evaluation Metrics

The following metrics can be calculated: 
* `energy_MAE`, `energy_RMSE`, `forces_MAE`, `forces_RMSE`, `forces_cosim`, `forces_MAE_species`, `forces_RMSE_species`, `stress_MAE`, `stress_RMSE`.

They can be customized using the autotune configuration: 

```bash
franken.autotune \
  ... \
  --metrics energy_MAE forces_MAE forces_MAE_species stress_MAE \
  --best-model-selection energy_MAE forces_MAE stress_MAE
```

```python
from franken.config import AutotuneConfig

cfg = AutotuneConfig(
    ...,
    metrics=["energy_MAE", "forces_MAE", "forces_MAE_species", "stress_MAE"],
    best_model_selection=["energy_MAE", "forces_MAE", "stress_MAE"]
)
```

When using species-resolved metrics (`forces_MAE_species`, `forces_RMSE_species`), the logs do not store a single scalar, but rather a value per each atomic number `Z` (`..._<Z>`) and the average of the metric per species (`..._average`)

By default, if `metrics` is omitted (i.e. `None`), the program computes all the available metrics associated with the specified training targets (e.g. `energy`, `forces`, `stress`).

## Best-Model Selection in Autotune

Autotune chooses the best model from the metrics in `best_model_selection` by (i) building the Pareto frontier using a list of metrics and (ii) among Pareto-efficient models, minimizing the p-norm (`p=1`).

The metric(s) which are used to perfom the selection can be customized in the CLI using `--best-model-selection` (`best_model_selection` for the APIs).

By default, if `best_model_selection` is left empty, the algorithm uses the `MAE` of all provided `train_targets` to perform the selection.
Therefore, a metric can be computed and logged in the evaluation process via `metrics` (e.g. `stress_MAE`), but ignored during the autotune model selection if it is not explicitly listed in `best_model_selection`.

Notes:
- This affects model ranking only (`best.json` / `best_ckpt.pt`), not the training loss.
- To use species-resolved metrics, use `*_average` (or an explicit `*_Z`) in `best_model_selection`.
