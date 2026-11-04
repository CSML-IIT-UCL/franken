# Minimal inference scaling benchmark, 4 November 2026

Default installation checks passed with pip on Python 3.11.14 and a uv project
on Python 3.13.11, using PyTorch 2.10 CPU wheels and MACE 0.3.17. Both produced
a trained checkpoint without cuEquivariance installed. The MACE extra is absent
from the built wheel. Combined extras resolved with pip for each CUDA choice;
uv compiled all extras for Python 3.11 and 3.13 and locked combined extras in a
project. See `installation.json`. These are dependency-resolution checks, not
runtime tests of every optional backend. The existing generic CuPy extra is
unchanged; resolution alone does not establish whether its installation will
need a local toolkit.

The active benchmark is `tests/benchmarks/mace_inference.py` (170 lines),
replacing the previous multi-script benchmark. It accepts a fitted checkpoint
and measures energy/force inference only. Batch size is one; neighbor building,
GPU transfers, setup and warmup are excluded. Both backends use ordinary reverse
autograd, float32 backbones and the same float64 fitted head. There is no custom
training Jacobian fallback or automatic cuEquivariance selection.

For this run, `prepare_checkpoint.py` fits one MACE-MP medium model with 4096
random features on two perturbed 8-atom silicon cells. It runs once before the
benchmark; larger configurations are never used for fitting. The synthetic
Lennard-Jones labels exercise the inference pipeline, not physical accuracy.
The preparation helper was checked through fitting and checkpoint reloading on
CPU before GPU submission. It is a fixture for this reference run, not part of
the timed benchmark.

Franklin environment preparation completed as CPU job 483907. Dedicated A100
job **483913** is queued; Slurm revised its estimated start from 21:43 to
13:20 local time on November 4 (an estimate, not a reservation). It requests one A100 80 GB, two CPUs, 16 GiB
host memory and five minutes. It sweeps 8, 64, 216, 512, 1000, 1728, 4096 and 8000
atoms, with two warmup calls and five alternating paired timed trials per size.
No workstation GPU benchmark was run because its existing simulation is busy.

The job removes its isolated environment, models and caches when it exits.
A temporary collector copies the CSV and plot here once the job stops, removes
the remote run directory after successful transfer, and then deletes itself.
`run.json` records current status and provenance. The pending submission 483908
was canceled and replaced before running so that the exit cleanup was included
in the actual script spooled by Slurm.
