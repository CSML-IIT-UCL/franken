from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd
import pickle


def plots(file_name, save: bool):
    with open(file_name, "rb") as fh:
        data = pickle.load(fh)
    for d in data:
        for j in [1, 2, 4, 16, 64, 128]:
            key = f"results_ts_time_per_atom_{j}"
            if key not in d:
                continue
            if isinstance(d[key], tuple):
                d[key] = d[key][0]
    df = pd.DataFrame(data)

    # Make sure no duplicates exist
    counts = df.groupby(['gnn_path_or_id', 'compile']).size()
    if (counts > 1).any():
        raise ValueError(
            "Multiple rows found for some (gnn_path_or_id, compile) pairs:\n"
            f"{counts[counts > 1]}"
        )
    # Check whether to plot the torch-sim experiment
    plot_torchsim = True
    if "results_ts_time_per_atom_1" not in df.columns:
        plot_torchsim = False
    # Note: aggregation is fake, we don't allow duplicates
    agg_lbls = {
        'gnn_family': 'first',
        'results_train_time': 'mean',
        'results_md_time_per_atom': 'mean',
        'results_forces_MAE': 'mean',
        'results_md_stable': 'all',
    }
    if plot_torchsim:
        agg_lbls |= {
            'results_ts_time_per_atom_1': 'mean',
            'results_ts_time_per_atom_2': 'mean',
            'results_ts_time_per_atom_4': 'mean',
            'results_ts_time_per_atom_16': 'mean',
            'results_ts_time_per_atom_64': 'mean',
            'results_ts_time_per_atom_128': 'mean',
        }
    agg_df = (
        df.groupby(['gnn_path_or_id', 'compile'], as_index=False)
        .agg(agg_lbls)
        .sort_values(['gnn_family', 'gnn_path_or_id'])
    )

    fig, ax = plt.subplots(ncols=2, nrows=3, figsize=(12, 8))
    plot_grouped_bar(
        agg_df,
        value_col='results_train_time',
        title='Training Time per GNN',
        ylabel='Time (s)',
        x_ticks=False,
        affects_compile=False,
        figax=(fig, ax[0, 0]),
    )
    plot_grouped_bar(
        agg_df,
        value_col='results_md_time_per_atom',
        title='MD Time per Atom per GNN',
        ylabel='Time per atom (s)',
        x_ticks=False,
        affects_compile=True,
        stability_col="results_md_stable",
        figax=(fig, ax[1, 0]),
    )
    ax[1, 0].set_yscale('log', base=2)
    plot_grouped_bar(
        agg_df,
        value_col='results_forces_MAE',
        title='Forces MAE per GNN',
        ylabel='MAE',
        x_ticks=True,
        affects_compile=False,
        figax=(fig, ax[2, 0]),
    )
    if plot_torchsim:
        plot_throughput(
            agg_df,
            col_prefix="results_ts_time_per_atom", 
            lw=2,
            figax=(fig, ax[0, 1]),
        )
        ax[0, 1].set_title("Torch-Sim throughput")
    else:
        ax[0, 1].axis('off')
    ax[1, 1].axis('off')
    ax[2, 1].axis('off')
    fig.tight_layout()
    if save:
        save_path = Path(file_name).with_suffix(".png")
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
    plt.show()


def plot_throughput(df, col_prefix="", family_col="gnn_family", lw=1, figax=None):
    if figax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        fig, ax = figax
    families = df[family_col].unique()
    cmap = plt.get_cmap('tab10')
    linestyles = ["solid", "dotted", "dashed", "dashdot"]
    family_to_ls = {
        fam: linestyles[i] for i, fam in enumerate(families)
    }
    x = sorted([int(c.split("_")[-1]) for c in df.columns if c.startswith(col_prefix)])

    for fam, subdf in df.groupby(family_col):
        gnn_ids = sorted(subdf['gnn_path_or_id'].unique())
        for i, g in enumerate(gnn_ids):
            subset = subdf[subdf['gnn_path_or_id'] == g]
            y = [subset[f"{col_prefix}_{x_val}"].mean() for x_val in x]
            ax.plot(x, y, ls=family_to_ls[fam], c=cmap(i % 10), label=g, marker='o', lw=lw)
    ax.legend(bbox_to_anchor=(0.5, -0.3))
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Time per atom (s)")
    ax.set_xscale('log', base=2)
    ax.set_yscale('log', base=2)
    return fig, ax


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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str, help="Path to database")
    parser.add_argument('-s', '--save', action="store_true", help="Save output plot")
    args = parser.parse_args()
    plots(args.path, args.save)

