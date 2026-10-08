# Metrics

The following metrics can be calculated on the provided datasets:

* Energy: `energy_MAE`, `energy_RMSE`, 
* Forces: `forces_MAE`, `forces_RMSE`, `forces_cosim`, `forces_MAE_species`, `forces_RMSE_species`
* Stress: `stress_MAE`, `stress_RMSE`.

They can be requested using the autotune keyword `metrics`: 

``` 
  --metrics energy_MAE forces_MAE forces_MAE_species  
``` 

Notes:
* By default, if `metrics` is omitted, the program computes the available metrics associated with the specified training targets (e.g. `energy`, `forces`, `stress`).

* When using species-resolved metrics, the logs do not store a single scalar called `forces_MAE_species` or `forces_RMSE_species`, but rather a value per each atomic number `Z` (`..._<Z>`) and the average of the metric per species (`..._average`).


### Best-model selection

Autotune chooses the best model by (i) building the Pareto frontier using a list of metrics and (ii) among Pareto-efficient models, minimizing the p-norm (`p=1`). The list of metrics used for selection can be customized with 
```
  --best-model-selection energy_MAE forces_MAE_average
``` 

By default, if `best_model_selection` is left empty, the algorithm uses the `MAE` of all provided `train_targets` to perform the selection.

