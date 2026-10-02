#!/bin/bash
#SBATCH --job-name=astrouda_aggregate
# ---- Cluster settings: edit these for your site, aggregation needs no GPU ----
#SBATCH --partition=gpu
#SBATCH --account=CHANGE_ME
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/%x_%j.out
# ---- End of cluster settings ----
# Usage: sbatch --dependency=afterok:ARRAY_JOB_ID slurm/aggregate.sbatch [CONFIG] [--set key=value ...]
set -euo pipefail

CONFIG_PATH="${1:-configs/lsst_full.json}"
if [ "$#" -gt 0 ]; then shift; fi

# ---- Environment: edit these for your site ----
CONDA_ENVIRONMENT_NAME="${CONDA_ENVIRONMENT_NAME:-astrouda}"
# ---- End of environment ----

cd "${SLURM_SUBMIT_DIR}"

# conda activation scripts reference unset variables
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENVIRONMENT_NAME}"
set -u

# Deterministic cuBLAS kernels, needed when torch deterministic algorithms are on
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
# Same hash seed in every job so set/dict ordering cannot differ between runs
export PYTHONHASHSEED=0
# Do not oversubscribe the CPUs allocated to this job (image resizing and augmentation run on CPU)
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
# Flush log lines immediately so slurm/logs can be followed with tail -f
export PYTHONUNBUFFERED=1

python -m astrouda.cli aggregate --config "${CONFIG_PATH}" "$@"
