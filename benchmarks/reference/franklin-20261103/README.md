# Franklin MACE backend measurements, 3 November 2026

These measurements evaluate Franken inference after fitting a random-feature head.
They do not measure MACE foundation-model training. Training remains e3nn-only;
there is no accepted cuEquivariance training speedup in this report.

The corrected inference configuration enables stock MACE convolution fusion and
native symmetric contractions. The earlier `unfused/` measurements disabled
convolution fusion and must not be presented as the performance of the corrected
configuration. All inference comparisons use the same fitted coefficients,
scaler, random features and backbone checkpoint, with numerical parity checks.

## Environment and provenance

Python 3.12.12, PyTorch 2.10.0+cu126, MACE 0.3.17, e3nn 0.4.4,
cuEquivariance core/frontend/CUDA-12 native operators 0.10.0. These are stock
upstream packages. See `environment.txt` for the full freeze, and each JSON for
checkpoint hashes, timings, memory, profiler events and allocation metadata.

Runs used dedicated Slurm GPU allocations on Franklin, with two CPU threads.
The corrected A100 jobs requested two CPUs, 12 GiB host memory and five minutes
on afnode01. No local GPU benchmark was run because the workstation GPUs had
active GROMACS workloads.

The JSON source_revision field records base commit `94b71f0`. Corrected fused
runs additionally used the wrapper change subsequently committed as `97c4e7b`.
The script hash in each JSON identifies the exact benchmark script used.
Earlier unfused results lack GPU clock/process snapshots because their UUID
query omitted the NVIDIA `GPU-` prefix; the corrected fused runs include them.
Raw measurements are preserved without rewriting their provenance fields.

## Validation and compatibility

The fused A100 regression job 483762 passed 31 tests and skipped eight PET tests.
The CPU regression job 483748 passed 20 tests and skipped 18 CUDA/PET tests;
job 483760 separately passed CPU loading with a saved cuEquivariance choice
overridden to e3nn. Numerical energy/force parity is checked in the benchmark.

The tested cuEquivariance 0.10 CUDA-12 wheel fails on V100 with
`cudaErrorNoKernelImageForDevice`; `compatibility/v100-smoke.json` records this
failure. V100 measurements therefore use e3nn only. This is a finding about the
tested wheel, not a claim that every cuEquivariance version is incompatible.

The local workstation's active simulations were untouched. The isolated remote
environment, downloaded models and installation caches are removed after result
collection; only the reproducible scripts and deliberate reference records remain.

## Corrected fused inference results

A100, MACE-MP medium, 64 atoms, 4096 random features; three paired timed
samples, ten fresh evaluations per inference sample, two warmup calls.

| Phase | e3nn | cuEquivariance | e3nn / cuEq |
|---|---:|---:|---:|
| Energy | 7.16 ms | 12.15 ms | 0.59 |
| Energy and forces, prepared graph | 22.62 ms | 30.89 ms | 0.73 |
| Energy and forces, ASE | 26.65 ms | 35.02 ms | 0.76 |

For prepared-graph energy/force inference, transient PyTorch peak allocation
fell from about 250 MB to 66 MB. These are allocator measurements with both
models resident; they are not total GPU memory usage. Numerical differences
were below 2.4e-7 absolute for the tested energies and force components.
Native operators appeared in the profiler and convolution fusion was active.
Both backends used ordinary reverse autograd for forces.

Fusion does not make this 64-atom Franken workload faster. Fixed launch and
framework overhead is a plausible explanation, but this benchmark does not
attribute elapsed time to individual kernels or prove that explanation.
The finding cannot establish cuEquivariance performance for larger structures,
other neighbor densities, species, batching, models, compilation settings or
precision. Treat the earlier unfused slowdown as superseded for the default
inference configuration, and avoid advertising a general acceleration claim.

The complete four-frame e3nn fit, including two candidate force weights, took
9.12 seconds in this case. Training speed is unchanged by the inference backend.
cuEquivariance checkpoint loading/conversion took about 5.0 seconds, and the
first observed native energy/force call took 44 seconds, including compilation.
These setup costs matter for short calculations and are excluded from the
warmed timings above.

## V100 e3nn training baseline

MACE-MP medium, 4096 random features, four training frames, two force-weight candidates.

| Atoms | Feature derivatives, one frame | Complete fit |
|---|---:|---:|
| 8 | 0.36 s | 1.76 s |
| 64 | 2.77 s | 11.84 s |

## Incomplete larger-system attempt

Job 483765 attempted a 512-atom medium/4096-feature case with two training
frames and one timed trial. It had not produced a complete case when its
five-minute allocation limit was reached; it was canceled while still listed
as running beyond that limit. Its raw JSON therefore retains `status=running`
and an empty cases list, while Slurm accounting records termination. The log
shows the e3nn fitting stages were still in progress. The `incomplete/` files
are diagnostic records, not accepted inference measurements. No speedup or
scaling claim can be made from that attempt. A larger-system inference study
should fit once on a small dataset and vary only the inference structure size,
so training Jacobian costs do not consume the inference measurement budget.
