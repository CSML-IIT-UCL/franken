import json
import matplotlib.pyplot as plt
import numpy as np


def plot_HPGridSearch(run, hyperparameter_1, hyperparameter_2, kind):
    with open(run / "log.json") as f:
        data = json.load(f)

    heatmap_data_energy = {}
    heatmap_data_forces = {}

    for run in data:
        hp1 = run["hyperparameters"]["solver"][hyperparameter_1]
        hp2 = run["hyperparameters"]["solver"][hyperparameter_2]

        rmse_forces = run["metrics"]["validation"]["forces_RMSE"]
        rmse_energy = run["metrics"]["validation"]["energy_RMSE"]

        if np.isnan(rmse_energy) or np.isnan(rmse_forces):
            continue

        heatmap_data_energy[(hp1, hp2)] = rmse_energy
        heatmap_data_forces[(hp1, hp2)] = rmse_forces

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, values, title in (
        (axes[0], heatmap_data_energy, "Energy RMSE [meV/atom]"),
        (axes[1], heatmap_data_forces, "Force RMSE [meV/A]"),
    ):
        hp1s = sorted({hp1 for hp1, hp2 in values})
        hp2s = sorted({hp2 for hp1, hp2 in values})
        if not values:
            ax.set_title(title)
            ax.text(
                0.5,
                0.5,
                "No valid data",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_axis_off()
            continue

        grid = np.full((len(hp1s), len(hp2s)), np.nan)
        for (hp1, hp2), value in values.items():
            grid[hp1s.index(hp1), hp2s.index(hp2)] = value

        image = ax.imshow(np.ma.masked_invalid(grid), cmap="viridis", aspect="auto")
        fig.colorbar(image, ax=ax)
        ax.set_xticks(range(len(hp2s)), labels=[f"{v:.0e}" for v in hp2s])
        ax.set_yticks(range(len(hp1s)), labels=[f"{v:.0e}" for v in hp1s])
        for (i, j), value in np.ndenumerate(grid):
            if np.isfinite(value):
                red, green, blue, _ = image.cmap(image.norm(value))
                brightness = 0.299 * red + 0.587 * green + 0.114 * blue
                ax.text(
                    j,
                    i,
                    f"{value:.1f}",
                    ha="center",
                    va="center",
                    color="black" if brightness > 0.5 else "white",
                )
        ax.set_title(title)
        ax.set_xlabel(hyperparameter_2)
        ax.set_ylabel(hyperparameter_1)

    fig.tight_layout()
    fig.savefig(f"{kind}_hyperparameter_analysis.png")
    plt.close(fig)
