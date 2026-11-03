#!/bin/bash
# Supply the partition, QOS, typed GPU GRES, --chdir and --output to sbatch.
#SBATCH --job-name=franken-mace-benchmark
#SBATCH --account=computationalstatisticsandmachinelearning
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=32G
#SBATCH --time=02:00:00
set -euo pipefail
root=${1:?isolated run directory}
label=${2:?a100 or v100}
shift 2
module load intel/slurm
if [ ! -f "$root/READY" ]; then
    echo 'Preparation did not finish successfully' >&2
    exit 2
fi
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONDONTWRITEBYTECODE=1
export FRANKEN_CACHE_DIR="$root/models"
export PYTHONPATH="$root/source"
srun --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" bash -c '
    root=$1
    label=$2
    shift 2
    scratch=${SLURM_SCRATCHDIR:?Slurm task scratch was not initialized}
    export TMPDIR="$scratch/tmp"
    export CUDA_CACHE_PATH="$scratch/cuda"
    export TRITON_CACHE_DIR="$scratch/triton"
    export TORCHINDUCTOR_CACHE_DIR="$scratch/inductor"
    export XDG_CACHE_HOME="$scratch/xdg"
    export MPLCONFIGDIR="$scratch/matplotlib"
    mkdir -p "$TMPDIR" "$CUDA_CACHE_PATH" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$XDG_CACHE_HOME" "$MPLCONFIGDIR"
    exec "$root/environment/bin/python" "$root/source/benchmarks/mace_backends.py" \
        --label "$label" --threads "$SLURM_CPUS_PER_TASK" \
        --source-revision "$(cat "$root/source-revision")" \
        --output "$root/results/$label.json" "$@"
' bash "$root" "$label" "$@"
