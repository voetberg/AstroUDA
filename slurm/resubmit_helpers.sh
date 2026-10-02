#!/bin/bash
# Functions only, sourced by train.sbatch, experiment_array.sbatch and aggregate.sbatch. Nothing runs on source.
#   TIME_MARGIN_SECONDS     seconds kept free for startup, the final checkpoint and shell overhead (default 1200)
#   MAXIMUM_RESUBMISSIONS   cap on the length of a resubmission chain (default 20)
#   RESUBMISSION_COUNT      how many times this chain was already resubmitted, set by the previous link (default 0)

parse_slurm_duration_seconds() {
    # D-HH:MM:SS, HH:MM:SS or MM:SS to seconds. Prints nothing and returns 1 for anything else
    local duration_text="$1"
    local days=0
    local clock_text="${duration_text}"
    local -a clock_parts

    if [[ "${duration_text}" == *-* ]]; then
        days="${duration_text%%-*}"
        clock_text="${duration_text#*-}"
    fi
    if ! [[ "${days}" =~ ^[0-9]+$ && "${clock_text}" =~ ^[0-9]+(:[0-9]+){1,2}$ ]]; then
        return 1
    fi

    IFS=: read -r -a clock_parts <<< "${clock_text}"
    if [ "${#clock_parts[@]}" -eq 3 ]; then
        echo $(( 10#${days} * 86400 + 10#${clock_parts[0]} * 3600 + 10#${clock_parts[1]} * 60 + 10#${clock_parts[2]} ))
    else
        echo $(( 10#${days} * 86400 + 10#${clock_parts[0]} * 60 + 10#${clock_parts[1]} ))
    fi
}

slurm_time_limit_seconds() {
    # TimeLimit of this job from scontrol. Prints nothing and returns 1 when it cannot be determined
    local job_description
    local time_limit_text

    if [ -z "${SLURM_JOB_ID:-}" ] || ! command -v scontrol > /dev/null; then
        return 1
    fi

    job_description="$(scontrol show job "${SLURM_JOB_ID}" 2> /dev/null)" || return 1
    time_limit_text="$(grep -o 'TimeLimit=[^ ]*' <<< "${job_description}" | head -n 1 | cut -d= -f2)" || true

    parse_slurm_duration_seconds "${time_limit_text}"
}

wall_time_budget_seconds() {
    # Training budget: time limit - seconds this script already used ($SECONDS) - TIME_MARGIN_SECONDS, never negative
    local time_limit_seconds
    local budget_seconds

    time_limit_seconds="$(slurm_time_limit_seconds)" || return 0
    if [ -z "${time_limit_seconds}" ]; then
        return 0
    fi

    budget_seconds=$(( time_limit_seconds - SECONDS - ${TIME_MARGIN_SECONDS:-1200} ))
    if [ "${budget_seconds}" -lt 0 ]; then
        budget_seconds=0
    fi

    echo "${budget_seconds}"
}

check_resubmission_allowed() {
    local resubmission_count="${RESUBMISSION_COUNT:-0}"
    local maximum_resubmissions="${MAXIMUM_RESUBMISSIONS:-20}"

    if [ "${resubmission_count}" -ge "${maximum_resubmissions}" ]; then
        echo "ERROR: resubmission cap reached (${resubmission_count} of ${maximum_resubmissions}), not resubmitting. Raise MAXIMUM_RESUBMISSIONS or inspect the run." >&2
        return 1
    fi
}

resubmit_training_job() {
    # Usage: resubmit_training_job SCRIPT [ARGUMENT ...]. Submits the same script with the same arguments
    local script_path="$1"
    shift
    local next_resubmission_count=$(( ${RESUBMISSION_COUNT:-0} + 1 ))
    local new_job_id

    check_resubmission_allowed || return 1
    new_job_id="$(sbatch --parsable --export="ALL,RESUBMISSION_COUNT=${next_resubmission_count}" "${script_path}" "$@")"
    echo "Resubmitted ${script_path} as job ${new_job_id} (resubmission ${next_resubmission_count})"
}

resubmit_experiment_task() {
    # Usage: resubmit_experiment_task SCRIPT [ARGUMENT ...]. Resubmits only this array task, plus a later aggregation job
    local script_path="$1"
    shift
    local aggregate_script_path
    aggregate_script_path="$(dirname "${script_path}")/aggregate.sbatch"
    local next_resubmission_count=$(( ${RESUBMISSION_COUNT:-0} + 1 ))
    local new_job_id
    local aggregation_job_id

    check_resubmission_allowed || return 1
    new_job_id="$(sbatch --parsable --export="ALL,RESUBMISSION_COUNT=${next_resubmission_count}" \
        --array="${SLURM_ARRAY_TASK_ID}" "${script_path}" "$@")"
    echo "Resubmitted ${script_path} task ${SLURM_ARRAY_TASK_ID} as job ${new_job_id} (resubmission ${next_resubmission_count})"

    aggregation_job_id="$(sbatch --parsable --dependency="afterok:${new_job_id}" "${aggregate_script_path}" "$@")"
    echo "Submitted aggregation job ${aggregation_job_id}, runs after ${new_job_id} succeeds"
}

aggregation_should_be_deferred() {
    # Usage: aggregation_should_be_deferred OUTPUT_FILE. True when the aggregate failed on missing runs and
    # experiment jobs of this user are still queued or running, so a later aggregation job will do it
    local aggregate_output_file="$1"

    if ! grep -q 'have no report.json' "${aggregate_output_file}"; then
        return 1
    fi

    [ -n "$(squeue -h -u "${USER}" -n astrouda_experiment 2> /dev/null)" ]
}
