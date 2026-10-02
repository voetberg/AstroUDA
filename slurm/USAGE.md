Run everything from the repository root. Edit the cluster settings block at the top of each .sbatch
(partition, account, time, mem, cpus, --gres=gpu:a100:1) and CONDA_ENVIRONMENT_NAME first.
Every script takes: CONFIG [--set key=value ...]. CONFIG defaults to configs/lsst_full.json.

Single training run:
  sbatch slurm/train.sbatch configs/lsst_full.json --set random_seed=1

Experiment (all seeds, adapted and source only, then aggregation):
  slurm/submit_experiment.sh configs/lsst_full.json
  Array size is 2 * number_of_seeds, aggregate.sbatch runs after every task succeeds (afterok).

Hyperparameter sweep (NUMBER_OF_TRIALS must match the array size):
  NUMBER_OF_TRIALS=30 sbatch --array=0-29 slurm/optimize_array.sbatch CONFIG SEARCH_SPACE.json

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
