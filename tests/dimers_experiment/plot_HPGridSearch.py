import json
import matplotlib.pyplot as plt
import seaborn as sns
import json
import numpy as np

def plot_HPGridSearch(run,hyperparameter_1,hyperparameter_2,kind):
    with open(run / "log.json") as f:
        data = json.load(f)

    heatmap_data_energy = {}
    heatmap_data_forces = {}

    for run in data:
        hp1 = run["hyperparameters"]["solver"][hyperparameter_1]
        hp2= run["hyperparameters"]["solver"][hyperparameter_2]
        
        rmse_forces = run["metrics"]["validation"]["forces_RMSE"]
        rmse_energy = run["metrics"]["validation"]["energy_RMSE"]
        
        if np.isnan(rmse_energy) or np.isnan(rmse_forces):
            continue

        heatmap_data_energy[(hp1, hp2)] = rmse_energy
        heatmap_data_forces[(hp1, hp2)] = rmse_forces



    fig, ax = plt.subplots(1, 2, figsize=(12,5))

    # energy
    hp1s = sorted({hp1 for hp1, hp2 in heatmap_data_energy})
    hp2s = sorted({hp2 for hp1, hp2 in heatmap_data_energy})
    grid = np.full((len(hp1s), len(hp2s)), np.nan)
    for (hp1, hp2), value in heatmap_data_energy.items():
        i = hp1s.index(hp1)
        j = hp2s.index(hp2)
        grid[i, j] = value
        
    sns.heatmap(
        grid,
        annot=True,
        fmt=".1f",
        cmap="viridis",
        xticklabels=[f"{v:.0e}" for v in hp2s],
        yticklabels=[f"{v:.0e}" for v in hp1s],
        ax=ax[0],
    )
    ax[0].set_title("Energy RMSE [meV/atom]")
    ax[0].set_xlabel(hyperparameter_2)
    ax[0].set_ylabel(hyperparameter_1)

    # forces
    hp1s = sorted({hp1 for hp1, hp2 in heatmap_data_forces})
    hp2s = sorted({hp2 for hp1, hp2 in heatmap_data_forces})
    grid = np.full((len(hp1s), len(hp2s)), np.nan)
    for (hp1, hp2), value in heatmap_data_forces.items():
        i = hp1s.index(hp1)
        j = hp2s.index(hp2)
        grid[i, j] = value
        
    sns.heatmap(
        grid,
        annot=True,
        fmt=".1f",
        cmap="viridis",
        xticklabels=[f"{v:.0e}" for v in hp2s],
        yticklabels=[f"{v:.0e}" for v in hp1s],
        ax=ax[1],
    )
    ax[1].set_title("Force RMSE [meV/A]")
    ax[1].set_xlabel(hyperparameter_2)
    ax[1].set_ylabel(hyperparameter_1)

    plt.tight_layout()
    plt.savefig(f"{kind}_hyperparameter_analysis.png")
    #plt.show()
    return
