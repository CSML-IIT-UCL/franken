"""Compare e3nn/cuEquivariance energy-and-force inference as atom count grows.

Run from the repository root with a fitted MACE checkpoint and a cueq extra:
    python tests/benchmarks/mace_inference.py fitted.pt --plot throughput.png

Batch size is one. Neighbor lists and transfers are outside the timings, as is
training. Both backends use the same fitted head and ordinary reverse autograd.
"""

import argparse
import csv
from importlib.metadata import version
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
import torch
from ase.build import bulk

from franken.data import Configuration
from franken.rf.model import FrankenPotential


def graph(atoms, model):
    data = Configuration(
        atom_pos=torch.tensor(atoms.positions, dtype=torch.float32),
        atomic_numbers=torch.tensor(atoms.numbers),
        natoms=torch.tensor([len(atoms)]),
        cell=torch.tensor(atoms.cell.array, dtype=torch.float32),
        pbc=torch.tensor(atoms.pbc),
    )
    return model.gnn.get_neighbors(data).to(device=torch.device("cuda"))


def predict(model, data):
    # Force inference mutates atom_pos; give every call a fresh leaf tensor.
    data.atom_pos = data.atom_pos.detach().clone()
    return model.predict(["energy", "forces"], data)


def compare(models, data, warmup, repeats):
    outputs = {}
    for name, model in models.items():
        for _ in range(max(1, warmup)):
            outputs[name] = predict(model, data)
    for key in ["energy", "forces"]:
        torch.testing.assert_close(
            outputs["cueq"][key], outputs["e3nn"][key], atol=1e-4, rtol=3e-4
        )
    del outputs
    samples = {name: [] for name in models}
    for iteration in range(repeats):
        names = list(models)
        if iteration % 2:
            names.reverse()
        for name in names:
            torch.cuda.synchronize()
            start = perf_counter()
            output = predict(models[name], data)
            torch.cuda.synchronize()
            samples[name].append(perf_counter() - start)
            del output
    return {name: median(times) for name, times in samples.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "--cells", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 8, 10]
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--output", type=Path, default=Path("mace-inference.csv"))
    parser.add_argument("--plot", type=Path)
    args = parser.parse_args()
    if min(args.cells) < 1 or min(args.repeats, args.warmup, args.threads) < 1:
        parser.error("cells, repeats, warmup and threads must be positive")
    if not torch.cuda.is_available():
        parser.error("a CUDA GPU and the matching cueq extra are required")
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    models = {
        name: FrankenPotential.load(
            args.checkpoint, map_location="cuda", use_cueq=cueq
        ).eval()
        for name, cueq in [("e3nn", False), ("cueq", True)]
    }
    assert any(
        type(module).__name__ == "CueqConvFusionWrapper"
        for module in models["cueq"].gnn.modules()
    ), "upstream cuEquivariance convolution fusion must be active"
    print(f"GPU: {torch.cuda.get_device_name()}; torch {torch.__version__}", flush=True)
    print(
        f"MACE {version('mace-torch')}; cuEq {version('cuequivariance-torch')}",
        flush=True,
    )
    print(
        f"Backbone: {models['e3nn'].gnn_config.path_or_id}; RFs: {models['e3nn'].rf.total_random_features}",
        flush=True,
    )
    print(f"{'Atoms':>8} {'e3nn ms':>12} {'cuEq ms':>12} {'Speedup':>10}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "atoms",
                "backend",
                "seconds",
                "atoms_per_second",
                "configurations_per_second",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        for cells in args.cells:
            atoms = bulk("Si", "diamond", a=5.43, cubic=True).repeat((cells,) * 3)
            atoms.positions += np.random.default_rng(17).normal(
                0, 0.03, atoms.positions.shape
            )
            data = graph(atoms, models["e3nn"])
            times = compare(models, data, args.warmup, args.repeats)
            for name, seconds in times.items():
                row = dict(
                    atoms=len(atoms),
                    backend=name,
                    seconds=seconds,
                    atoms_per_second=len(atoms) / seconds,
                    configurations_per_second=1 / seconds,
                )
                writer.writerow(row)
                rows.append(row)
            stream.flush()
            print(
                f"{len(atoms):8d} {times['e3nn'] * 1000:12.3f} {times['cueq'] * 1000:12.3f} {times['e3nn'] / times['cueq']:9.2f}x",
                flush=True,
            )
            del data
    if args.plot:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")
        for name in models:
            series = [row for row in rows if row["backend"] == name]
            ax.plot(
                [row["atoms"] for row in series],
                [row["atoms_per_second"] for row in series],
                "o-",
                label=name,
            )
        ax.set(
            xlabel="Atoms per configuration",
            ylabel="Throughput (atoms/s)",
            title="Franken energy + forces, batch size 1",
            xscale="log",
        )
        ax.grid(alpha=0.25)
        ax.legend()
        args.plot.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.plot, dpi=180)
        plt.close(fig)


if __name__ == "__main__":
    main()
