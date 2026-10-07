"""Compare CUDA solver times; matrix construction and input copies are not timed.

Run from the repository root:
    python benchmarks/benchmark_psd_solve.py
"""

import argparse
from statistics import median
from time import perf_counter

import torch

from franken.trainers.rf_lowmem import LowMemRandomFeaturesTrainer, cupy
from franken.trainers.rf_trainer import RandomFeaturesTrainer


def time_solver(solver, cov, rhs, warmup, repeats):
    times = []
    for iteration in range(warmup + repeats):
        # Both solvers mutate their inputs; give each run fresh, identical data.
        solve_cov, solve_rhs = cov.clone(), rhs.clone()
        torch.cuda.synchronize(cov.device)
        start = perf_counter()
        solution = solver(solve_cov, solve_rhs, 1e-3)
        torch.cuda.synchronize(cov.device)
        elapsed = perf_counter() - start
        if iteration >= warmup:
            times.append(elapsed)
        del solve_cov, solve_rhs, solution
    return median(times)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gpu", type=int, default=0, help="Index of the single visible GPU to use"
    )
    parser.add_argument("--min-power", type=int, default=8)
    parser.add_argument("--max-power", type=int, default=14)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    args = parser.parse_args()
    if not 0 <= args.min_power <= args.max_power <= 14:
        parser.error("powers must satisfy 0 <= min-power <= max-power <= 14")
    if args.repeats < 1 or args.warmup < 0:
        parser.error("repeats must be positive and warmup nonnegative")
    if not torch.cuda.is_available() or cupy is None:
        parser.error("CUDA and CuPy are required to compare both solvers")

    if not 0 <= args.gpu < torch.cuda.device_count():
        parser.error("gpu must be the index of a visible CUDA device")
    device = torch.device(f"cuda:{args.gpu}")
    # PyTorch and CuPy must select the same device, including cuSOLVER allocations.
    torch.cuda.set_device(device)
    cupy.cuda.Device(args.gpu).use()
    generator = torch.Generator(device=device).manual_seed(0)
    dtype = getattr(torch, args.dtype)
    trainer_args = dict(
        train_dataloader=None,
        training_targets=["energy", "forces"],
        l2_penalty=1e-3,
        target_weight={},
        device=device,
        dtype=dtype,
    )
    naive = RandomFeaturesTrainer(**trainer_args).psd_solve
    lowmem = LowMemRandomFeaturesTrainer(**trainer_args).psd_solve
    print(f"GPU: {torch.cuda.get_device_name(device)} ({device})", flush=True)
    print(flush=True)
    print(
        f"{'Size':>8} {'Naive (ms)':>12} {'Lowmem (ms)':>13} {'Naive/Lowmem':>14}",
        flush=True,
    )
    print(f"{'-' * 8} {'-' * 12} {'-' * 13} {'-' * 14}", flush=True)
    with torch.no_grad():
        for power in range(args.min_power, args.max_power + 1):
            n = 2**power
            features = torch.randn(
                n, n, device=device, dtype=dtype, generator=generator
            )
            cov = (features @ features.T).div_(n)
            del features
            rhs = torch.randn(n, device=device, dtype=dtype, generator=generator)
            naive_seconds = time_solver(naive, cov, rhs, args.warmup, args.repeats)
            lowmem_seconds = time_solver(lowmem, cov, rhs, args.warmup, args.repeats)
            print(
                f"{n:>8,} {naive_seconds * 1000:>12.3f} "
                f"{lowmem_seconds * 1000:>13.3f} "
                f"{naive_seconds / lowmem_seconds:>12.2f}x",
                flush=True,
            )
            del cov, rhs


if __name__ == "__main__":
    main()
