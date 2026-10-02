Run everything from the repository root. The cluster settings block at the top of each .sbatch holds the
NERSC Perlmutter values from calo-om2rta (account, qos, constraint, -G, cpus, time); CONDA_ENVIRONMENT
(a path or a name, default /pscratch/sd/v/voetberg/astrouda) is set just below it.
Every script except test_gpu.sbatch takes: CONFIG [--set key=value ...]. CONFIG defaults to configs/lsst_full.json.

First GPU check (debug qos, 20 minutes, needs no data): CUDA AMP test, then ResNet-50, crop 256, batch 64 on
synthetic images for two epochs, with peak GPU memory and utilisation printed at the end:
  sbatch slurm/test_gpu.sbatch
  tail -f slurm/logs/astrouda-test_<jobid>.out

Single training run:
  sbatch slurm/train.sbatch configs/lsst_full.json --set random_seed=1

Experiment (all seeds, adapted and source only, then aggregation):
  slurm/submit_experiment.sh configs/lsst_full.json
  Array size is 2 * number_of_seeds, aggregate.sbatch runs after every task succeeds (afterok).

Hyperparameter sweep (NUMBER_OF_TRIALS must match the array size):
  NUMBER_OF_TRIALS=30 sbatch --array=0-29 slurm/optimize_array.sbatch CONFIG SEARCH_SPACE.json

Wall time and automatic resubmission (train.sbatch, experiment_array.sbatch, optimize_array.sbatch request 8 hours):
  The job reads its TimeLimit from scontrol and trains with a budget of
  limit - time already used - TIME_MARGIN_SECONDS (default 1200, for startup, final checkpoint, shell overhead).
  When the next epoch would not fit, training checkpoints and exits 75; the script then resubmits itself with the
  same arguments and continues from <output_directory>/checkpoint.pt (auto_resume=true is passed by the scripts).
  Experiment array: only the stopped task is resubmitted (--array=<task>), plus a new aggregate.sbatch job that
  depends on it (afterok). An aggregation that finds runs missing while astrouda_experiment jobs are still queued
  or running defers to the later aggregation job and exits 0.
  MAXIMUM_RESUBMISSIONS (default 20) caps a chain, RESUBMISSION_COUNT is set by each link, the job fails clearly at the cap:
    MAXIMUM_RESUBMISSIONS=40 TIME_MARGIN_SECONDS=1800 sbatch slurm/train.sbatch CONFIG
  See the chain: squeue -u $USER -o "%i %j %T %E", and slurm/logs/*.out ("Resubmitted ... as job N").
  optimize_array.sbatch has the limit only: trials are short and finished trials are skipped on rerun.

Resume a training run from its checkpoint:
  sbatch slurm/train.sbatch CONFIG --set resume_from_checkpoint=outputs/NAME/checkpoint.pt

Rerun failed experiment tasks only, then aggregate by hand:
  sbatch --array=3,7 slurm/experiment_array.sbatch CONFIG
  sbatch slurm/aggregate.sbatch CONFIG

Outputs (output_directory in the config, relative to the submit directory):
  train:      <output_directory>/{checkpoint.pt,best_model.pt,history.json,config.json}
  experiment: <output_directory>/seed_<seed>/{adapted,source_only}/, then aggregate.json and comparison_table.md
  sweep:      written by the optimize command under <output_directory>
  logs:       slurm/logs/<jobname>_<jobid>_<arraytask>.out
