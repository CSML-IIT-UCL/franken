# MACE backend benchmarks

`mace_backends.py` compares Franken's standard e3nn 0.4.4 MACE backend with
its optional native cuEquivariance backend. Training always uses e3nn. The fitted checkpoint is then loaded with both
inference backends, preserving its random features, coefficients and scaler.
Both use the same unmodified MACE release and pretrained backbone checkpoint.
This measures Franken training and inference, rather than MACE foundation-model
training.

Install the current checkout and the cuEquivariance extra matching PyTorch's
CUDA build, then run on an allocated GPU:

```bash
python benchmarks/mace_backends.py \
  --models mace_mp/small mace_mp/medium \
  --repeats 1 2 --features 256 4096 \
  --train-frames 4 --trials 3 --warmup 2 --inference-calls 10 \
  --threads 5 --jac-chunk-size 4 --output results/mace-backends.json
```

`--repeats 1 2` constructs periodic silicon systems with 8 and 64 atoms.
`--repeats 3` adds a 216-atom system.
`--stress` adds stress to the
training and energy/force inference targets. `--backends e3nn` runs a baseline
without requiring native kernels, useful when a GPU does not support them.

## Measurements

For each model, system size and feature count, the output records:

- **Training feature maps:** energy and force feature maps for one prepared
  structure using e3nn. This isolates the derivative computation used during fitting.
- **Fit:** `RandomFeaturesTrainer.fit` on the specified training structures,
  including descriptor statistics, feature derivatives, covariance construction,
  normalization, and fitting two force weights (0.1 and 0.9) at one length scale
  and ridge penalty. Every repetition uses a fresh trainer, so statistics and
  feature maps cannot be reused from a previous fit.
- **Energy inference:** one fitted model on a prepared GPU graph, with gradients
  disabled.
- **Energy and force inference:** one fitted model on a prepared GPU graph using
  ordinary reverse autograd. This is distinct from training feature derivatives.
- **ASE energy and force inference:** includes neighbor construction, host/device
  transfers and the normal `FrankenCalculator` call. The benchmark invokes
  `calculate()` explicitly so ASE cannot return a cached prediction.

File reading, reference-label generation, neighbor precomputation for training,
checkpoint downloads and model initialization are excluded from fitting timings.
Model initialization has a separate field. Training labels come from a fixed
Lennard-Jones potential; they test the numerical path, not physical accuracy.
A held-out perturbed structure is used for inference. The final fitted e3nn
model is saved, then loaded with `use_cueq=True` through the public API.
Checkpoint loading/conversion has a separate timing field. Inference checks
numerical agreement before accepting the timing comparison.

Reported timings synchronize CUDA before and after each sample. Inference backend order is
randomized and interleaved for every trial. Training has one e3nn baseline,
so there is no cuEquivariance training speedup field. The output includes individual
samples, median, range and standard deviation. `e3nn_over_cueq` is the ratio of
median e3nn time to median cuEquivariance time; values above one favor
cuEquivariance. Inference samples can contain multiple **fresh** evaluations to
reduce timer noise. Their recorded seconds are per evaluation.

The first observed call is reported separately from warmup and timed trials.
It can include kernel compilation, but is not a guaranteed cold start: other
phases or earlier cases in the same process may already have compiled related
kernels. Model initialization and first evaluation should therefore be inspected
alongside warmed throughput when estimating a short run.

Backbone computations and random-feature maps use float32. The ridge solver
and the fitted head coefficients use float64, matching saved training models.
PyTorch TF32 matmul and cuDNN flags are disabled. No `torch.compile` or AMP is enabled. cuEquivariance inference uses upstream
convolution fusion and native symmetric contractions. Archived unfused runs are
labelled separately and measure symmetric-contraction acceleration only. Native symmetric-contraction operators must appear
in the profiler before a cuEquivariance result is accepted. Numerical comparisons
cover energy/force (and optionally stress) inference from the same fitted model.
Zero force maps or failed/nonfinite solves fail the case rather than producing
performance claims.

The e3nn training backend uses the requested Jacobian chunk size. cuEquivariance
is inference-only, so it never needs a full random-feature Jacobian. Inference
forces use ordinary reverse autograd for the fitted energy.

Both models are resident during paired inference measurements. Memory fields report current
PyTorch allocated bytes, peak allocated bytes, and additional transient peak bytes.
They exclude CUDA allocator reservations, driver allocations and other processes.
GPU snapshots, device UUID, versions, checkpoint hashes, script hash and Slurm
metadata accompany the results. Results from a busy shared GPU should be identified
as contended rather than compared with dedicated allocations.

## Franklin

Use an isolated project run directory containing `source/`, an installed
`environment/`, downloaded checkpoints in `models/`, a `source-revision` file,
a `READY` marker, and a `results/` directory. `franklin_mace_backends.sh` runs
inside a Slurm allocation and puts temporary datasets and kernel caches in
`SLURM_SCRATCHDIR`; the Slurm epilog and Python temporary-directory context clean
them up.

For the tested V100/A100 comparison, the shared environment uses Python 3.12,
PyTorch 2.10.0 from the official CUDA 12.6 index, stock MACE 0.3.17, e3nn 0.4.4,
and cuEquivariance core/frontend/native CUDA 12 wheels at 0.10.0. CUDA 13 builds
cannot target the V100 architecture. The tested native cuEquivariance 0.10
CUDA 12 wheel also has no V100 kernel image; report that incompatibility and
use `--backends e3nn` for the V100 baseline. Install and check the environment in a
preparation job, download both checkpoints, and record `uv pip freeze`.

Example submission from an existing prepared run directory:

```bash
module load intel/slurm
run=/projects/dynrepr/your-isolated-benchmark-run
sbatch --partition=gpua --qos=gpua \
  --gres=gpu:nv-a100-sxm4-80gb:1 \
  --chdir="$run/source" --output="$run/results/a100-%j.log" \
  "$run/source/benchmarks/franklin_mace_backends.sh" "$run" a100
```

For a 16 GB V100, use `--partition=gpuv --qos=gpuv` and
`--gres=gpu:nv-tesla-v100-sxm2-16gb:1`. Always inspect current availability and
existing jobs first. The launcher requests one GPU, five CPUs, 32 GiB host memory
and two hours; adapt these requests to your case matrix.

The script writes JSON incrementally. A failed case includes its exception and
traceback and has no accepted speedup. A successful environment installation alone
does not establish that native kernels support a particular GPU. Preserve results
and environment metadata, then remove the isolated environment, downloads and
installation caches after collecting them.

## Summarizing results

```bash
python benchmarks/summarize_mace_backends.py a100-small.json a100-medium.json v100.json \
  --csv summary.csv --plot inference-speedup.png
```

The CSV contains each measurement's median, range, first-observed time and memory
fields. The optional figure requires Matplotlib and shows inference speedups only;
e3nn is the sole training backend. Failed or incomplete cases are listed separately
and excluded from accepted speedups.
