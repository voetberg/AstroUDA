# AstroUDA

From-scratch implementation of DeepAstroUDA, universal domain adaptation for galaxy morphology: https://arxiv.org/abs/2302.02005

## Install

```
conda env create -f environment.yml
conda activate astrouda
```

`conda-forge` provides the CUDA build of PyTorch on Linux and the CPU/MPS build on macOS.

## Quick start

Synthetic-data smoke runs on a laptop (`train` takes about 15 seconds, `experiment` a minute or two):

```
python -m astrouda.cli train --config configs/smoke.json
python -m astrouda.cli experiment --config configs/smoke.json
```

Every setting lives in `astrouda/config.py` as a typed default and is overwritten from a JSON file, then from `--set key=value` flags (values are parsed as JSON). Unknown keys and wrongly typed values raise.

## Commands

| Command | What it does |
| --- | --- |
| `python -m astrouda.cli train --config C.json` | One training run. Writes `config.json`, `history.json`, `checkpoint.pt` (resumable) and `best_model.pt` to `output_directory`. |
| `python -m astrouda.cli experiment --config C.json [--run-index I]` | `number_of_seeds` seeds, each trained with and without adaptation, then aggregated. `--run-index` runs a single (seed, mode) pair for Slurm job arrays. |
| `python -m astrouda.cli aggregate --config C.json` | Mean and population standard deviation over seeds, written to `aggregate.json` and `comparison_table.md`. |
| `python -m astrouda.cli optimize --config C.json --search-space S.json --trials N [--trial-index I]` | Random search over Config fields. Trial `I` depends only on the seed, `I` and the search space. |

Cluster usage (Slurm, one A100 per job) is in `slurm/USAGE.md`.

## Experiments

| Experiment | Config | Data |
| --- | --- | --- |
| Smoke test, 3 classes | `configs/smoke.json` | synthetic |
| Smoke test, 10 classes | `configs/smoke_10class.json` | synthetic |
| LSST Y1 to Y10, 3 classes | `configs/lsst_full.json` | https://zenodo.org/records/5514180 |
| Galaxy Zoo 2, SDSS to DECaLS | `configs/gz2_sdss_decals.json` | https://zenodo.org/records/7473597 |
| Galaxy Zoo 2, SDSS Wide to Deep | `configs/gz2_sdss_wide_deep.json` | https://zenodo.org/records/7473597 |

Data is not downloaded by the code. Place the files in `data_directory`; a missing file raises an error that lists the expected paths.

## Method

One randomly initialised ResNet with domain-specific BatchNorm, trained with

L = L_CE + lambda (L_AC + L_ES), lambda = 0.005

- **L_CE**: class-weighted cross entropy on the source (Eq. 4).
- **L_AC**: adaptive clustering on the target. Pairs are labelled similar when the top-k classes agree in the same order, and compared against a bank of earlier samples (Eq. 1).
- **L_ES**: entropy separation on the target (Eqs. 2-3). Its parameters rho and m are tuned during training by Algorithm 1.
- **Optimiser**: SGD with Nesterov momentum, lr 0.001, StepLR.

## Choices where the paper is silent

| Item | Choice |
| --- | --- |
| Memory bank | FIFO of detached probabilities, 2048 samples, holds source and target. The AC loss is computed before the current batch is added. |
| Domain-specific BatchNorm | One `BatchNorm2d` per domain with its own statistics and affine parameters; convolution weights are shared. |
| L_min in Algorithm 1 | Minimum training total loss. The step index wraps after the fourth step. The doubling phase reuses the margin patience. |
| Augmented views | Two views per image (exact 90 degree rotation; zoom to 300 then centre crop). Target views get gradients and are scored separately for AC and ES, averaged with the plain term. Source views run without gradients and only feed the bank. CE uses the plain source batch. |
| Early stopping | Target validation accuracy, patience 12. The best-epoch weights are used for test evaluation. |
| Source-only baseline | Target is evaluated with the source BatchNorm statistics. |
| Unspecified hyperparameters | Batch size 64, momentum 0.9, weight decay 1e-4, 100 epochs maximum. |
| Image normalisation | uint8 divided by 255; other dtypes min-max scaled per domain. |

## Tests

```
python -m pytest tests -q
```

The suite runs on CPU in a couple of minutes. Tests marked `gpu` skip unless CUDA is available.

## Status

- The pipeline runs end to end on synthetic data. The paper's results (for example about 74% target accuracy on LSST with adaptation) have not been reproduced; that needs full-size runs on the real data.
- The Galaxy Zoo 2 loaders were only tested on synthetic HDF5 files; the real file names, keys and shapes are unverified.
- The GPU and mixed-precision paths and the Slurm scripts have not been run on a real GPU or cluster.
- Resizing and augmentation run on the CPU and may limit A100 throughput (issue #16).
