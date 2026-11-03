#!/usr/bin/env python3
"""Compare Franken's stock e3nn and native cuEquivariance MACE paths.

Run in an installed Franken environment with the appropriate cueq extra.
Results are incrementally written as JSON; temporary training data is removed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import tempfile
import time
import traceback


def positive_int(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return result


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument(
        "--backends", nargs="+", choices=["e3nn", "cueq"], default=["e3nn", "cueq"]
    )
    cli.add_argument("--source-revision", default=None)
    cli.add_argument(
        "--inference-only-cueq",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    cli.add_argument("--models", nargs="+", default=["mace_mp/small", "mace_mp/medium"])
    cli.add_argument(
        "--repeats",
        nargs="+",
        type=positive_int,
        default=[1, 2],
        help="Silicon conventional-cell repetitions: 1=8 atoms, 2=64 atoms",
    )
    cli.add_argument("--features", nargs="+", type=positive_int, default=[256, 4096])
    cli.add_argument("--train-frames", type=positive_int, default=4)
    cli.add_argument(
        "--trials", type=positive_int, default=3, help="paired timed trials per phase"
    )
    cli.add_argument(
        "--inference-calls",
        type=positive_int,
        default=10,
        help="fresh evaluations per inference timing sample",
    )
    cli.add_argument("--warmup", type=int, default=2)
    cli.add_argument("--threads", type=positive_int, default=5)
    cli.add_argument("--jac-chunk-size", type=positive_int, default=4)
    cli.add_argument("--seed", type=int, default=1729)
    cli.add_argument(
        "--stress", action="store_true", help="include stress training and inference"
    )
    cli.add_argument("--label", default="local")
    cli.add_argument("--output", type=Path, required=True)
    return cli


def command(args):
    try:
        return subprocess.check_output(
            args, text=True, stderr=subprocess.STDOUT, timeout=15
        ).strip()
    except (OSError, subprocess.SubprocessError) as error:
        return str(error)


def gpu_snapshot(uuid=None):
    if uuid and not uuid.startswith("GPU-"):
        uuid = "GPU-" + uuid
    fields = "uuid,name,driver_version,utilization.gpu,memory.used,clocks.sm,power.draw"
    args = ["nvidia-smi", "--query-gpu=" + fields, "--format=csv,noheader,nounits"]
    if uuid:
        args += ["--id=" + uuid]
    processes = command(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    lines = [
        line for line in processes.splitlines() if not uuid or line.startswith(uuid)
    ]
    return {"gpu_fields": fields, "gpu": command(args), "processes": lines}


def environment(torch):
    versions = {}
    for name in [
        "franken",
        "torch",
        "mace-torch",
        "e3nn",
        "cuequivariance",
        "cuequivariance-torch",
        "cuequivariance-ops-torch-cu12",
        "cuequivariance-ops-torch-cu13",
        "numpy",
        "ase",
    ]:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    props = torch.cuda.get_device_properties(0)
    uuid = str(getattr(props, "uuid", "")) or None
    return {
        "versions": versions,
        "python": platform.python_version(),
        "host": platform.node(),
        "platform": platform.platform(),
        "cpu": command(["lscpu"]),
        "gpu": props.name,
        "gpu_uuid": uuid,
        "vram_bytes": props.total_memory,
        "compute_capability": list(torch.cuda.get_device_capability()),
        "torch_cuda": torch.version.cuda,
        "torch_arch_list": torch.cuda.get_arch_list(),
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "slurm": {
            key: os.environ[key]
            for key in [
                "SLURM_JOB_ID",
                "SLURM_JOB_PARTITION",
                "SLURM_CPUS_PER_TASK",
                "SLURM_JOB_GPUS",
                "CUDA_VISIBLE_DEVICES",
            ]
            if key in os.environ
        },
        "gpu_snapshot": gpu_snapshot(uuid),
    }


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def sample_stats(samples):
    ordered = sorted(samples)
    return {
        "samples_seconds": samples,
        "median_seconds": statistics.median(samples),
        "min_seconds": ordered[0],
        "max_seconds": ordered[-1],
        "mean_seconds": statistics.mean(samples),
        "stdev_seconds": statistics.stdev(samples) if len(samples) > 1 else 0.0,
    }


def measure(torch, fn, calls=1):
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    result = None
    for _ in range(calls):
        result = fn()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return (
        result,
        elapsed / calls,
        {
            "baseline_allocated_bytes": baseline,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "transient_peak_bytes": max(
                0, torch.cuda.max_memory_allocated() - baseline
            ),
        },
    )


def paired_phase(torch, calls, args, rng, repetitions=1):
    names = list(calls)
    first, memories, values = {}, {}, {}
    rng.shuffle(names)
    for name in names:
        values[name], first[name], memories[name] = measure(torch, calls[name])
    for _ in range(args.warmup):
        rng.shuffle(names)
        for name in names:
            values[name], _, _ = measure(torch, calls[name])
    samples = {name: [] for name in names}
    order = []
    for _ in range(args.trials):
        rng.shuffle(names)
        order.append(names.copy())
        for name in names:
            values[name], seconds, memory = measure(torch, calls[name], repetitions)
            samples[name].append(seconds)
            memories[name]["peak_allocated_bytes"] = max(
                memories[name]["peak_allocated_bytes"], memory["peak_allocated_bytes"]
            )
            memories[name]["transient_peak_bytes"] = max(
                memories[name]["transient_peak_bytes"], memory["transient_peak_bytes"]
            )
    results = {
        name: sample_stats(samples[name])
        | {"first_observed_seconds": first[name], "memory": memories[name]}
        for name in samples
    }
    if "e3nn" in samples and "cueq" in samples:
        speedup = results["e3nn"]["median_seconds"] / results["cueq"]["median_seconds"]
    else:
        speedup = None
    return {
        "backends": results,
        "e3nn_over_cueq": speedup,
        "paired_order": order,
        "calls_per_timed_sample": repetitions,
    }, values


def fresh_graph(data, torch):
    out = data.to(device=torch.device("cuda"))
    out.atom_pos = out.atom_pos.detach().clone()
    return out


def parity(torch, actual, reference, atol, rtol):
    output = {}
    for key in reference:
        torch.testing.assert_close(actual[key], reference[key], atol=atol, rtol=rtol)
        diff = (actual[key] - reference[key]).detach().double()
        ref = reference[key].detach().double()
        output[key] = {
            "max_abs": diff.abs().max().item(),
            "relative_norm": (diff.norm() / ref.norm().clamp_min(1e-12)).item(),
        }
    return output


def make_data(repeat, count, seed, stress):
    import numpy as np
    from ase.build import bulk
    from ase.calculators.lj import LennardJones
    from ase.calculators.singlepoint import SinglePointCalculator

    rng = np.random.default_rng(seed)
    base = bulk("Si", "diamond", a=5.43, cubic=True).repeat((repeat,) * 3)
    frames = []
    for _ in range(count):
        atoms = base.copy()
        atoms.positions += rng.normal(0.0, 0.03, atoms.positions.shape)
        atoms.calc = LennardJones(epsilon=0.1, sigma=2.0, rc=5.0)
        labels = {"energy": atoms.get_potential_energy(), "forces": atoms.get_forces()}
        if stress:
            labels["stress"] = atoms.get_stress(voigt=False)
        atoms.calc = SinglePointCalculator(atoms, **labels)
        frames.append(atoms)
    return frames


def native_events(torch, model, data):
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as prof:
        model.predict(["energy", "forces"], fresh_graph(data, torch))
        torch.cuda.synchronize()
    events = [
        event.key
        for event in prof.key_averages()
        if any(
            word in event.key.lower()
            for word in ["cuequivariance", "uniform_1d", "segmented_polynomial"]
        )
    ]
    if not any("cuequivariance::" in event for event in events):
        raise RuntimeError(
            "cuEquivariance requested but no native operator appeared in the profiler"
        )
    return events


def benchmark_case(
    torch, args, model_id, repeat, features, directory, checkpoints, result
):
    from ase.io import write
    from franken.backbones.utils import get_checkpoint_path
    from franken.backbones.wrappers.common_patches import patch_e3nn
    from franken.calculators.ase_calc import FrankenCalculator
    from franken.config import GaussianRFConfig, MaceBackboneConfig
    from franken.data import Configuration, FrankenAtomsDataset
    from franken.rf.model import FrankenPotential
    from franken.trainers import RandomFeaturesTrainer

    patch_e3nn()
    targets = ["energy", "forces"] + (["stress"] if args.stress else [])
    frames = make_data(repeat, args.train_frames + 1, args.seed, args.stress)
    train_path = directory / "train.xyz"
    write(train_path, frames[:-1])
    train = FrankenAtomsDataset(
        train_path, split="train", gnn_config=MaceBackboneConfig(model_id)
    )
    loader = train.get_dataloader(distributed=False)
    list(loader)
    torch.cuda.synchronize()
    start = time.perf_counter()
    classic = (
        FrankenPotential(
            MaceBackboneConfig(model_id),
            GaussianRFConfig(
                num_random_features=features, length_scale=2.0, rng_seed=args.seed
            ),
            scale_by_Z=False,
            num_species=1,
            jac_chunk_size=args.jac_chunk_size,
        )
        .cuda()
        .eval()
    )
    torch.cuda.synchronize()
    setup = {"e3nn": time.perf_counter() - start}
    data, _ = train[0]
    data = data.to(device=torch.device("cuda"))
    atoms = frames[-1]
    partial = Configuration(
        atom_pos=torch.tensor(atoms.positions, dtype=torch.float32),
        atomic_numbers=torch.tensor(atoms.numbers),
        natoms=torch.tensor([len(atoms)]),
        cell=torch.tensor(atoms.cell.array, dtype=torch.float32),
        pbc=torch.tensor(atoms.pbc),
    )
    holdout_graph = classic.gnn.get_neighbors(partial).to(device=torch.device("cuda"))
    uuid = str(getattr(torch.cuda.get_device_properties(0), "uuid", "")) or None
    result.update(
        train_frames=args.train_frames,
        targets=targets,
        setup_seconds=setup,
        training_backend="e3nn",
        phases={},
        parity={},
        gpu_before=gpu_snapshot(uuid),
    )
    path = get_checkpoint_path(model_id)
    if model_id not in checkpoints:
        checkpoints[model_id] = {
            "filename": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    rng = random.Random(args.seed + repeat + features)
    calls = {
        "e3nn": lambda: classic.grad_feature_map(fresh_graph(data, torch), targets)
    }
    result["phases"]["training_feature_maps"], values = paired_phase(
        torch, calls, args, rng
    )
    if values["e3nn"]["forces"].norm() < 1e-7:
        raise RuntimeError("Force feature maps vanished in the benchmark fixture")
    del values

    def fit():
        trainer = RandomFeaturesTrainer(
            loader,
            training_targets=targets,
            l2_penalty=[0.01],
            target_weight={"forces": [0.1, 0.9]},
            random_features_normalization="leading_eig",
            log_dir=None,
            save_every_model=False,
            device="cuda",
            dtype=torch.float64,
            save_fmaps=False,
            metrics=["energy_MAE", "forces_MAE"],
        )
        logs, weights = trainer.fit(classic)
        if len(logs) != 2 or not torch.isfinite(weights).all() or weights.norm() == 0:
            raise RuntimeError(
                "The fit did not return two finite, nonzero candidate models"
            )
        return weights

    result["phases"]["fit"], fitted = paired_phase(torch, {"e3nn": fit}, args, rng)
    classic.rf.weights = fitted["e3nn"][0:1].detach().clone()
    result["coefficient_dtype"] = str(classic.rf.weights.dtype)
    del fitted
    models = {"e3nn": classic}
    if "cueq" in args.backends:
        # Exercise the real public workflow: fit once with e3nn, save, then load
        # the exact same RF coefficients and scaler with a converted backbone.
        checkpoint = directory / "fitted.pt"
        classic.save(checkpoint)
        torch.cuda.synchronize()
        start = time.perf_counter()
        models["cueq"] = FrankenPotential.load(
            checkpoint, map_location="cuda", use_cueq=True
        ).eval()
        torch.cuda.synchronize()
        setup["cueq_checkpoint_load_and_conversion"] = time.perf_counter() - start
        result["cueq_conv_fusion"] = any(
            type(module).__name__ == "CueqConvFusionWrapper"
            for module in models["cueq"].gnn.modules()
        )
    both = "cueq" in models
    for phase, prediction_targets in [
        ("energy", ["energy"]),
        ("energy_forces", targets),
    ]:

        def inference(model, selected=prediction_targets):
            with torch.set_grad_enabled("forces" in selected):
                return model.predict(selected, fresh_graph(holdout_graph, torch))

        calls = {
            name: (lambda model=model: inference(model))
            for name, model in models.items()
        }
        result["phases"][phase], values = paired_phase(
            torch, calls, args, rng, repetitions=args.inference_calls
        )
        if both:
            result["parity"][phase] = parity(
                torch, values["cueq"], values["e3nn"], 1e-4, 3e-4
            )
        del values

    calculators = {}
    for name, model in models.items():
        start = time.perf_counter()
        calculators[name] = FrankenCalculator(model, device="cuda")
        setup[name + "_ase_calculator"] = time.perf_counter() - start
    properties = ["energy", "forces"] + (["stress"] if args.stress else [])

    def ase_inference(calc):
        calc.dataset.ase_atoms.clear()
        calc.calculate(frames[-1], properties=properties)
        return {
            key: torch.as_tensor(value).clone() for key, value in calc.results.items()
        }

    calls = {
        name: (lambda calc=calc: ase_inference(calc))
        for name, calc in calculators.items()
    }
    result["phases"]["ase_energy_forces"], values = paired_phase(
        torch, calls, args, rng, repetitions=args.inference_calls
    )
    if both:
        result["parity"]["ase_energy_forces"] = parity(
            torch, values["cueq"], values["e3nn"], 1e-4, 3e-4
        )
        # Profile after timing, so it cannot warm kernels before first-observed calls.
        result["native_profile_events"] = native_events(
            torch, models["cueq"], holdout_graph
        )
    result["gpu_after"] = gpu_snapshot(uuid)
    for phase_name, phase in result["phases"].items():
        for row in phase["backends"].values():
            count = args.train_frames if phase_name == "fit" else 1
            row["configurations_per_second"] = count / row["median_seconds"]
            row["atoms_per_second"] = len(frames[0]) * row["configurations_per_second"]
    return result


def main():
    args = parser().parse_args()
    if "e3nn" not in args.backends:
        parser().error("The benchmark always trains with e3nn; include --backends e3nn")
    if args.warmup < 0:
        parser().error("--warmup cannot be negative")
    import torch

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    output = {
        "schema": 1,
        "date_utc": datetime.now(timezone.utc).isoformat(),
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_revision": args.source_revision
        or command(["git", "rev-parse", "HEAD"]),
        "checkpoints": {},
        "cases": [],
        "status": "running",
    }
    write_json(args.output, output)
    if not torch.cuda.is_available():
        output.update(
            status="unavailable",
            error="CUDA is unavailable; refusing to benchmark CPU fallback",
        )
        write_json(args.output, output)
        return 2
    try:
        output["environment"] = environment(torch)
        # Fail early if the PyTorch build has no working kernels for this GPU.
        torch.ones((16, 16), device="cuda").square().sum().item()
    except Exception as error:
        output.update(
            status="unavailable", error=str(error), traceback=traceback.format_exc()
        )
        write_json(args.output, output)
        return 2
    write_json(args.output, output)
    failures = 0
    for model_id in args.models:
        for repeat in args.repeats:
            for features in args.features:
                case = {
                    "model": model_id,
                    "repeat": repeat,
                    "atoms": 8 * repeat**3,
                    "features": features,
                }
                start = time.perf_counter()
                try:
                    with tempfile.TemporaryDirectory(
                        prefix="franken-backend-case-"
                    ) as temp:
                        case = benchmark_case(
                            torch,
                            args,
                            model_id,
                            repeat,
                            features,
                            Path(temp),
                            output["checkpoints"],
                            case,
                        )
                    case["status"] = "passed"
                except Exception as error:
                    failures += 1
                    case.update(
                        status="failed",
                        exception=type(error).__name__,
                        error=str(error),
                        traceback=traceback.format_exc(),
                    )
                    traceback.print_exc()
                case["elapsed_seconds"] = time.perf_counter() - start
                output["cases"].append(case)
                write_json(args.output, output)
                print(
                    json.dumps(
                        {
                            "case": {
                                key: case[key]
                                for key in ["model", "atoms", "features", "status"]
                            },
                            "seconds": case["elapsed_seconds"],
                        }
                    ),
                    flush=True,
                )
                gc.collect()
                torch.cuda.empty_cache()
    output["status"] = "complete" if not failures else "completed_with_failures"
    output["completed_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(args.output, output)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
