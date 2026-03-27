import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd
import pickle


def plots(file_name):
    with open(file_name, "rb") as fh:
        data = pickle.load(fh)
    df = pd.DataFrame(data)

    # Make sure no duplicates exist
    counts = df.groupby(['gnn_path_or_id', 'compile']).size()
    if (counts > 1).any():
        raise ValueError(
            "Multiple rows found for some (gnn_path_or_id, compile) pairs:\n"
            f"{counts[counts > 1]}"
        )
    # Note: aggregation is fake, we don't allow duplicates
    agg_df = (
        df.groupby(['gnn_path_or_id', 'compile'], as_index=False)
        .agg({
            'gnn_family': 'first',
            'results_train_time': 'mean',
            'results_md_time_per_atom': 'mean',
            'results_forces_MAE': 'mean',
            'results_md_stable': 'all',
        })
        .sort_values(['gnn_family', 'gnn_path_or_id'])
    )

    fig, ax = plt.subplots(ncols=1, nrows=3, figsize=(6, 8))
    plot_grouped_bar(
        agg_df,
        value_col='results_train_time',
        title='Training Time per GNN',
        ylabel='Time',
        x_ticks=False,
        affects_compile=False,
        figax=(fig, ax[0]),
    )
    plot_grouped_bar(
        agg_df,
        value_col='results_md_time_per_atom',
        title='MD Time per Atom per GNN',
        ylabel='Time per atom',
        x_ticks=False,
        affects_compile=True,
        stability_col="results_md_stable",
        figax=(fig, ax[1]),
    )
    plot_grouped_bar(
        agg_df,
        value_col='results_forces_MAE',
        title='Forces MAE per GNN',
        ylabel='MAE',
        x_ticks=True,
        affects_compile=False,
        figax=(fig, ax[2]),
    )
    fig.tight_layout()
    plt.show()


def plot_grouped_bar(
    df, 
    value_col, 
    title, 
    ylabel, 
    x_ticks,
    family_col="gnn_family",
    affects_compile=True, 
    stability_col=None, 
    figax=None
):
    gnn_ids = df['gnn_path_or_id'].unique()
    compile_modes = df['compile'].unique()
    families = df[family_col].unique()
    cmap = plt.get_cmap('tab10')
    family_to_color = {
        fam: cmap(i % 10) for i, fam in enumerate(families)
    }
    hatch_map = {
        comp: hatch
        for comp, hatch in zip(
            compile_modes, ['/', '\\\\', 'x']
        )
    }
    x = np.arange(len(gnn_ids))
    width = 0.25
    
    if figax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        fig, ax = figax
    
    if affects_compile:
        for i, comp in enumerate(compile_modes):
            subset = df[df['compile'] == comp]
            # Align values with all GNNs
            values = []
            colors = []
            stability = []
            for g in gnn_ids:
                row = subset[subset['gnn_path_or_id'] == g]
                if not row.empty:
                    values.append(row[value_col].values[0])
                    fam = row[family_col].values[0]
                    colors.append(family_to_color[fam])
                    if stability_col:
                        stability.append(row[stability_col].values[0])
                    else:
                        stability.append(None)
                else:
                    values.append(np.nan)
                    colors.append('gray')
                    stability.append(None)
            bars = ax.bar(
                x + i * width,
                values,
                width,
                label=f'compile={comp}',
                hatch=hatch_map[comp],
                color=colors
            )
            # Overlay stability markers
            if stability_col:
                for xi, yi, stab in zip(x + i * width, values, stability):
                    if stab is None or np.isnan(yi):
                        continue
                    marker = 'o'
                    color = 'green' if stab else 'red'
                    ax.scatter(
                        xi,
                        yi * 1.02,  # slightly above bar
                        color=color,
                        marker=marker,
                        zorder=3
                    )
        ax.set_xticks(x + width)
    else:
        # No compile distinction (single bar per GNN)
        values = []
        colors = []
        stability = []
        for g in gnn_ids:
            subset = df[df['gnn_path_or_id'] == g]
            values.append(subset[value_col].mean())
            fam = subset[family_col].iloc[0]
            colors.append(family_to_color[fam])
            if stability_col:
                # if any run unstable → mark unstable
                stability.append(subset[stability_col].all())
            else:
                stability.append(None)
        
        ax.bar(x, values, width=0.5, color=colors)
        if stability_col:
            for xi, yi, stab in zip(x, values, stability):
                marker = 'o'
                color = 'green' if stab else 'red'
                
                ax.scatter(
                    xi,
                    yi * 1.02,
                    color=color,
                    marker=marker,
                    zorder=3
                )
        ax.set_xticks(x)
    if x_ticks:
        ax.set_xticklabels(gnn_ids, rotation=45, ha='right')
    else:
        ax.set_xticks([], [])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    # Legend for compile
    compile_legend = []
    if affects_compile:
        compile_legend = [
            Patch(facecolor='white',
                edgecolor='black',
                hatch=hatch_map[c],
                label=f'compile={c}')
            for c in compile_modes
        ]
    # Legend for family (color)
    family_legend = [
        Patch(color=family_to_color[f], label=f)
        for f in family_to_color
    ]
    
    ax.legend(handles=compile_legend + family_legend, loc="best")
    
    return fig, ax

if __name__ == "__main__":
    plots("results_2703.pkl")