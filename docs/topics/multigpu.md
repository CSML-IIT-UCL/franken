# Distributed training

Franken supports data-parallel training across multiple GPUs via `torchrun`. Each rank processes a shard of the
dataset, accumulates local covariance/coefficients, and the results are summed across ranks to obtain the global
covariance/coefficients before solving.

```bash
torchrun --standalone --nnodes=1 --nproc-per-node=4 franken.autotune \
    --train-path="train_dataset.xyz" \
    --backbone=mace --mace.path-or-id "mace_mp/small" \
    --jac-chunk-size "auto" \
    --rf=gaussian --gaussian.num-rf 8192 --gaussian.length-scale="[5.,10.,20.]"
```

If you see a `FileNotFoundError`, call the script via its absolute path, for example:
`ENV_PATH/bin/franken.autotune`.

Below is an example Slurm script (for Leonardo HPC):

```bash 
#!/bin/bash
#SBATCH --account=XXX
#SBATCH --nodes=1                   # node
#SBATCH --ntasks-per-node=4         # tasks out of 32
#SBATCH --gres=gpu:4                # gpus per node out of 4
#SBATCH --cpus-per-task=8
#SBATCH -p boost_usr_prod

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# Activate the Franken environment
ENV_PATH=...
conda activate "$ENV_PATH"

torchrun --standalone --nnodes=1 --nproc-per-node="${SLURM_NTASKS_PER_NODE}" "$ENV_PATH/bin/franken.autotune" \
    --train-path="train_dataset.xyz" \
    --val-path="valid_dataset.xyz" \
    --backbone=mace --mace.path-or-id "mace_mp/small" --mace.interaction-block 2 \
    --force-weight=0.99 \
    --jac-chunk-size "auto" \
    --rf=gaussian --gaussian.num-rf 4096 --gaussian.length-scale="[5.,10.,20.]"
```
