#!/bin/bash
#SBATCH --job-name=astrouda_optimize
# The array range is a fallback for 20 trials, override it with: sbatch --array=0-N-1
#SBATCH --array=0-19
# ---- Cluster settings: edit these for your site ----
#SBATCH --partition=gpu
#SBATCH --account=CHANGE_ME
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=slurm/logs/%x_%A_%a.out
# ---- End of cluster settings ----
# Usage: sbatch slurm/optimize_array.sbatch CONFIG SEARCH_SPACE [--set key=value ...]
# NUMBER_OF_TRIALS must match the array size, it defines the random search the trial indices belong to
set -euo pipefail

CONFIG_PATH="${1:-configs/lsst_full.json}"
SEARCH_SPACE_PATH="${2:?search space JSON path is required as the second argument}"
if [ "$#" -gt 1 ]; then shift 2; elif [ "$#" -gt 0 ]; then shift; fi
NUMBER_OF_TRIALS="${NUMBER_OF_TRIALS:-20}"

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

python -m astrouda.cli optimize --config "${CONFIG_PATH}" --search-space "${SEARCH_SPACE_PATH}" \
    --trials "${NUMBER_OF_TRIALS}" --trial-index "${SLURM_ARRAY_TASK_ID}" "$@"
