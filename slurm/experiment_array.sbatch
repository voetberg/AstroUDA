#!/bin/bash
#SBATCH --job-name=astrouda_experiment
# The array range is a fallback for number_of_seeds=5, submit_experiment.sh overrides it from the config
#SBATCH --array=0-9
# ---- Cluster settings: edit these for your site ----
#SBATCH --partition=gpu
#SBATCH --account=CHANGE_ME
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=slurm/logs/%x_%A_%a.out
# ---- End of cluster settings ----
# Usage: sbatch slurm/experiment_array.sbatch [CONFIG] [--set key=value ...]
# Array task I runs seed index I/2, adapted when I is even and source only when I is odd
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

echo "Job ${SLURM_JOB_ID} array task ${SLURM_ARRAY_TASK_ID:-none} on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python -m astrouda.cli experiment --config "${CONFIG_PATH}" --run-index "${SLURM_ARRAY_TASK_ID}" "$@"
