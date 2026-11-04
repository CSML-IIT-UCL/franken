# MACE inference benchmark

The active benchmark is [`tests/benchmarks/mace_inference.py`](../tests/benchmarks/mace_inference.py),
following the small solver benchmark in the same directory. It takes an existing
fitted MACE checkpoint; training is never part of the timing or atom-count sweep.

```bash
python tests/benchmarks/mace_inference.py fitted.pt \
    --cells 1 2 3 4 5 6 8 10 --repeats 5 --warmup 2 \
    --output inference.csv --plot throughput.png
```

Install the matching `cueq-cu12` or `cueq-cu13` extra first. The default Franken
installation includes MACE and uses e3nn; cuEquivariance remains opt-in and
inference-only. Matplotlib is needed only for `--plot`.

The benchmark compares energy **and forces** on periodic, slightly perturbed
silicon crystals, from 8 to 8000 atoms. Batch size is one. Both backends load the
same fitted checkpoint and use ordinary reverse autograd. It checks numerical
parity at every size and verifies convolution fusion is enabled. Neighbor lists,
CPU/GPU transfers, model loading, compilation warmup and training are excluded.
CUDA synchronization surrounds each timed call; backend order alternates between
trials. The CSV reports median seconds, atoms/second and configurations/second.
The plot shows atoms/second against atoms per configuration. This measures the
Franken inference computation, not complete ASE/MD throughput or physical accuracy.
Use a dedicated GPU; an unrelated workload can distort the comparison.

`reference/` contains recorded runs. The November 3 records are historical;
the previous larger benchmark scripts are available in commit `babf0d6`.
