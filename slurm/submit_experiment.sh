#!/bin/bash
# Usage: slurm/submit_experiment.sh [CONFIG] [--set key=value ...]
# Run from the repository root. Submits the experiment array, then aggregation once every array task succeeded.
set -euo pipefail

CONFIG_PATH="${1:-configs/lsst_full.json}"
if [ "$#" -gt 0 ]; then shift; fi

mkdir -p slurm/logs

# Read through the Config class so --set overrides of number_of_seeds are respected
NUMBER_OF_SEEDS="$(python -c '
import sys
from astrouda.cli import parse_override
from astrouda.config import Config
overrides = dict(parse_override(text) for text in sys.argv[2:] if text != "--set")
print(Config(sys.argv[1], **overrides).number_of_seeds)
' "${CONFIG_PATH}" "$@")"
LAST_RUN_INDEX=$((2 * NUMBER_OF_SEEDS - 1))
echo "number_of_seeds=${NUMBER_OF_SEEDS}, array indices 0-${LAST_RUN_INDEX}"

ARRAY_JOB_ID="$(sbatch --parsable --array="0-${LAST_RUN_INDEX}" slurm/experiment_array.sbatch "${CONFIG_PATH}" "$@")"
echo "Submitted experiment array ${ARRAY_JOB_ID}"

AGGREGATE_JOB_ID="$(sbatch --parsable --dependency="afterok:${ARRAY_JOB_ID}" slurm/aggregate.sbatch "${CONFIG_PATH}" "$@")"
echo "Submitted aggregation ${AGGREGATE_JOB_ID}, runs after ${ARRAY_JOB_ID} succeeds"
