# Functions only, sourced by the sbatch scripts after the conda environment is active.
#
# prepare_data: when DOWNLOAD_DATASET is lsst or gz2, fetch that dataset into scratch at the start of the job and set
# DATA_ARGUMENTS to the --set that points the run at it. Files that are already complete are skipped, so resubmitted
# jobs and parallel array tasks (the downloader takes a lock) are safe. Otherwise DATA_ARGUMENTS is left empty and
# the config's own data_directory is used.
#   DOWNLOAD_DATASET   lsst or gz2, unset to skip the download
#   DATA_DIRECTORY     target directory, default $SCRATCH/astrouda_data/<dataset>

prepare_data() {
    DATA_ARGUMENTS=()
    if [ -z "${DOWNLOAD_DATASET:-}" ]; then
        return 0
    fi

    local data_directory="${DATA_DIRECTORY:-${SCRATCH:?SCRATCH must be set, or set DATA_DIRECTORY, to download data}/astrouda_data/${DOWNLOAD_DATASET}}"
    echo "Fetching ${DOWNLOAD_DATASET} into ${data_directory}"
    python -m astrouda.cli download --dataset "${DOWNLOAD_DATASET}" --directory "${data_directory}"

    DATA_ARGUMENTS=(--set "data_directory=${data_directory}")
}
