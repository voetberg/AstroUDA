#!/bin/bash
#SBATCH --job-name=astrouda_experiment
# The array range is a fallback for number_of_seeds=5, submit_experiment.sh overrides it from the config
#SBATCH --array=0-9
# ---- Cluster settings, NERSC Perlmutter values taken from calo-om2rta/submit_batch.slurm: edit for your site ----
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=m2612
#SBATCH --qos=regular
#SBATCH --constraint=gpu
#SBATCH -G 1
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --output=slurm/logs/%x_%A_%a.out
#SBATCH --mail-user=maggiev@fnal.gov
#SBATCH --mail-type=END,FAIL
# ---- End of cluster settings ----
# Usage: sbatch slurm/experiment_array.sbatch [CONFIG] [--set key=value ...]
# Array task I runs seed index I/2, adapted when I is even and source only when I is odd
set -euo pipefail

# Saved before the shift below so a resubmission gets exactly the arguments this job got
ORIGINAL_ARGUMENTS=("$@")
CONFIG_PATH="${1:-configs/lsst_full.json}"
if [ "$#" -gt 0 ]; then shift; fi

# ---- Environment: edit these for your site. A path (as in calo-om2rta) or an environment name ----
CONDA_ENVIRONMENT="${CONDA_ENVIRONMENT:-/pscratch/sd/v/voetberg/astrouda}"
# ---- End of environment ----

cd "${SLURM_SUBMIT_DIR}"
source slurm/resubmit_helpers.sh

# conda activation scripts reference unset variables
set +u
command -v module > /dev/null && module load python
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENVIRONMENT}"
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

WALL_TIME_BUDGET_SECONDS="$(wall_time_budget_seconds)"
BUDGET_ARGUMENTS=()
if [ -n "${WALL_TIME_BUDGET_SECONDS}" ]; then
    BUDGET_ARGUMENTS=(--set "maximum_wall_time_seconds=${WALL_TIME_BUDGET_SECONDS}")
    echo "Wall time budget for training: ${WALL_TIME_BUDGET_SECONDS} s"
else
    echo "Could not read the time limit from scontrol, training without a wall time budget"
fi

set +e
python -m astrouda.cli experiment --config "${CONFIG_PATH}" --run-index "${SLURM_ARRAY_TASK_ID}" "$@" \
    ${BUDGET_ARGUMENTS[@]+"${BUDGET_ARGUMENTS[@]}"} --set auto_resume=true
RUN_EXIT_STATUS=$?
set -e

if [ "${RUN_EXIT_STATUS}" -eq 0 ]; then
    echo "Run ${SLURM_ARRAY_TASK_ID} finished"
    exit 0
elif [ "${RUN_EXIT_STATUS}" -eq 75 ]; then
    echo "Run ${SLURM_ARRAY_TASK_ID} stopped for the wall time limit, resubmitting this task"
    resubmit_experiment_task slurm/experiment_array.sbatch ${ORIGINAL_ARGUMENTS[@]+"${ORIGINAL_ARGUMENTS[@]}"}
    exit 0
fi

echo "Run ${SLURM_ARRAY_TASK_ID} failed with exit status ${RUN_EXIT_STATUS}"
exit "${RUN_EXIT_STATUS}"
