#!/bin/bash
#SBATCH --job-name=astrouda-test
# ---- Cluster settings, NERSC Perlmutter values taken from calo-om2rta/submit_batch.slurm: edit for your site ----
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=m2612
#SBATCH --qos=debug
#SBATCH --constraint=gpu
#SBATCH -G 1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --mail-user=maggiev@fnal.gov
#SBATCH --mail-type=ALL
# ---- End of cluster settings ----
# Usage: sbatch slurm/test_gpu.sbatch
# Needs no data. Checks the CUDA mixed-precision path, then trains a full-size model (ResNet-50, crop 256, batch 64)
# on synthetic images for two epochs, logging GPU memory and utilisation, so the epoch times and peak memory in the
# log are real numbers for issue #16.
set -euo pipefail

# ---- Environment: edit these for your site. A path (as in calo-om2rta) or an environment name ----
CONDA_ENVIRONMENT="${CONDA_ENVIRONMENT:-/pscratch/sd/v/voetberg/astrouda}"
# ---- End of environment ----

cd "${SLURM_SUBMIT_DIR}"

# conda activation scripts reference unset variables
set +u
command -v module > /dev/null && module load python
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENVIRONMENT}"
set -u

# Flush log lines immediately so the log can be followed with tail -f
export PYTHONUNBUFFERED=1
# Do not oversubscribe the CPUs allocated to this job (image resizing and augmentation run on CPU)
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

OUTPUT_DIRECTORY="${SLURM_SUBMIT_DIR}/outputs/gpu_test_${SLURM_JOB_ID}"
GPU_USAGE_LOG="${OUTPUT_DIRECTORY}/gpu_usage.csv"
mkdir -p "${OUTPUT_DIRECTORY}"

echo "=== Job ${SLURM_JOB_ID} on $(hostname) at $(date) ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python -c "import torch; print('torch', torch.__version__, 'cuda available:', torch.cuda.is_available())"

nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader,nounits -l 2 > "${GPU_USAGE_LOG}" &
GPU_LOGGER_PID=$!
trap 'kill ${GPU_LOGGER_PID} 2> /dev/null || true' EXIT

echo "=== 1/2 CUDA tests ==="
python -m pytest -m gpu -q tests/test_slurm.py

echo "=== 2/2 Full-size model on synthetic data, mixed precision ==="
python -m astrouda.cli train --config configs/lsst_full.json \
    --set dataset_name=synthetic \
    --set synthetic_samples_per_domain=2048 \
    --set device=cuda \
    --set use_mixed_precision=true \
    --set maximum_epochs=2 \
    --set output_directory="${OUTPUT_DIRECTORY}"

kill "${GPU_LOGGER_PID}" 2> /dev/null || true
echo "=== Done at $(date) ==="
awk -F', ' '{ if ($2 + 0 > peak_memory) peak_memory = $2 + 0; if ($3 + 0 > peak_utilisation) peak_utilisation = $3 + 0 }
    END { print "Peak GPU memory used (MiB):", peak_memory; print "Peak GPU utilisation (%):", peak_utilisation }' "${GPU_USAGE_LOG}"
echo "Outputs: ${OUTPUT_DIRECTORY}"
