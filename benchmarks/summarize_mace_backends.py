#!/usr/bin/env python3
"""Print benchmark results and optionally export a CSV and speedup figure."""
import argparse
import csv
import json
from pathlib import Path


def records(inputs):
    rows, failures = [], []
    for path in inputs:
        result = json.loads(path.read_text())
        gpu = result.get("environment", {}).get(
            "gpu", result.get("arguments", {}).get("label", "?")
        )
        for case in result.get("cases", []):
            if case["status"] != "passed":
                failures.append(
                    {
                        "file": str(path),
                        "gpu": gpu,
                        "model": case["model"],
                        "error": case.get("error", "unfinished"),
                    }
                )
                continue
            for phase, data in case["phases"].items():
                for backend, timing in data["backends"].items():
                    row = {
                        "source": path.name,
                        "gpu": gpu,
                        "model": case["model"],
                        "atoms": case["atoms"],
                        "features": case["features"],
                        "phase": phase,
                        "backend": backend,
                        "median_seconds": timing["median_seconds"],
                        "min_seconds": timing["min_seconds"],
                        "max_seconds": timing["max_seconds"],
                        "first_observed_seconds": timing["first_observed_seconds"],
                        "e3nn_over_cueq": data["e3nn_over_cueq"],
                        "peak_allocated_bytes": timing["memory"][
                            "peak_allocated_bytes"
                        ],
                        "transient_peak_bytes": timing["memory"][
                            "transient_peak_bytes"
                        ],
                    }
                    rows.append(row)
        if result.get("status") != "complete":
            failures.append(
                {
                    "file": str(path),
                    "gpu": gpu,
                    "error": result.get("error", result.get("status")),
                }
            )
    return rows, failures


def plot(rows, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = [
        row
        for row in rows
        if row["backend"] == "cueq"
        and row["phase"] in ["energy_forces", "ase_energy_forces"]
    ]
    if not data:
        raise ValueError("No successful cuEquivariance inference results to plot")
    facets = sorted({(row["gpu"], row["model"]) for row in data})
    fig, axes = plt.subplots(
        1,
        len(facets),
        figsize=(6 * len(facets), 4),
        squeeze=False,
        constrained_layout=True,
    )
    for ax, (gpu, model) in zip(axes[0], facets):
        subset = [row for row in data if row["gpu"] == gpu and row["model"] == model]
        for features in sorted({row["features"] for row in subset}):
            for phase, label, color in [
                ("energy_forces", "Prepared graph", "#2463a6"),
                ("ase_energy_forces", "ASE", "#bc661b"),
            ]:
                series = sorted(
                    [
                        row
                        for row in subset
                        if row["features"] == features and row["phase"] == phase
                    ],
                    key=lambda row: row["atoms"],
                )
                ax.plot(
                    [row["atoms"] for row in series],
                    [row["e3nn_over_cueq"] for row in series],
                    marker="o",
                    linestyle=(
                        "-"
                        if features == min(row["features"] for row in subset)
                        else "--"
                    ),
                    color=color,
                    label=f"{label}, {features} RFs",
                )
        ax.axhline(1, color="gray", linestyle=":", linewidth=1)
        ax.set_title(f"{model}\n{gpu}")
        ax.set_xlabel("Atoms")
        ax.set_ylabel("e3nn time / cuEquivariance time")
        ax.set_xticks(sorted({row["atoms"] for row in subset}))
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.suptitle("Inference after e3nn training — warmed median speedup")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("inputs", nargs="+", type=Path)
    cli.add_argument("--csv", type=Path)
    cli.add_argument("--plot", type=Path)
    args = cli.parse_args()
    rows, failures = records(args.inputs)
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as stream:
            columns = list(rows[0]) if rows else ["source", "gpu", "model", "error"]
            writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    print("| GPU | Model | Atoms | RFs | Phase | e3nn ms | cueq ms | Speedup |")
    print("| --- | --- | ---: | ---: | --- | ---: | ---: | ---: |")
    keys = sorted(
        {
            tuple(row[key] for key in ["gpu", "model", "atoms", "features", "phase"])
            for row in rows
        }
    )
    for key in keys:
        group = {
            row["backend"]: row
            for row in rows
            if tuple(row[k] for k in ["gpu", "model", "atoms", "features", "phase"])
            == key
        }
        legacy = group.get("e3nn", {}).get("median_seconds")
        native = group.get("cueq", {}).get("median_seconds")
        speedup = legacy / native if legacy is not None and native is not None else None

        def format_ms(value):
            return f"{value*1000:.3f}" if value is not None else "—"

        print(
            "| "
            + " | ".join(map(str, key))
            + f" | {format_ms(legacy)} | {format_ms(native)} | "
            + (f"{speedup:.2f}x" if speedup is not None else "—")
            + " |"
        )
    if failures:
        print("\nFailures or incomplete runs:")
        for failure in failures:
            print(
                f'{failure["file"]}: {failure.get("model", "")}: {str(failure["error"])[:250]}'
            )
    if args.plot:
        plot(rows, args.plot)


if __name__ == "__main__":
    main()
