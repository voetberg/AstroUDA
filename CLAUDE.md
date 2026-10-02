# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

From-scratch implementation of DeepAstroUDA (https://arxiv.org/abs/2302.02005), built from the paper alone. Do not read or copy the authors' original code (github.com/deepskies/DeepAstroUDA); the repo owner ruled it out. Where the paper is silent, the choices made are listed in `README.md` ("Choices where the paper is silent"); keep that table current when changing one.

## Commands

```
conda env create -f environment.yml      # env name: astrouda
python -m pytest tests -q                # full suite, about 2 minutes on CPU
python -m pytest tests/losses -q         # one package
python -m pytest tests/test_smoke.py -q  # end-to-end smoke run on synthetic data
python -m astrouda.cli train --config configs/smoke.json
```

CLI subcommands are `train`, `experiment`, `aggregate` and `optimize`; see `README.md`. Run pytest as `python -m pytest` from the repo root so the local package is imported rather than another worktree's editable install.

## Architecture

- **`astrouda/config.py`**: the single source of settings. Typed class attributes with defaults, overwritten from JSON and `--set key=value`. Unknown keys and wrong types raise. Add new settings here rather than hard-coding them.
- **`astrouda/data/`**: loaders yield `DomainAdaptationBatch` (see `batch.py`). Source and target are sampled independently, never index-paired. Target labels appear only in validation and test batches. Datasets: synthetic, LSST (`.npy`), Galaxy Zoo 2 (HDF5, layout unverified against the real files). Splits are stratified and seeded.
- **`astrouda/models/`**: one ResNet (own implementation, random init) split into feature extractor and head. Domain-specific BatchNorm takes an explicit integer `domain_index` (0 source, 1 target) through every layer; there is no hidden state.
- **`astrouda/losses/`**: pure functions plus `ProbabilityBank`. Callers must compute the AC loss before adding the current batch to the bank, otherwise samples pair with themselves.
- **`astrouda/tuning/`**: `EntropySeparationTuner`, a torch-free state machine for Algorithm 1 (owns rho and m).
- **`astrouda/training/`**: `Trainer` ties everything together, with checkpoint and resume, early stopping on `target_validation_accuracy`, and `load_best_model()`. Source-only mode (`enable_domain_adaptation` false) evaluates the target with domain index 0.
- **`astrouda/evaluation/`**: trainer-independent metrics, plots and seed aggregation (population std, ddof=0).
- **`astrouda/experiment/`**: multi-seed driver, adapted versus source-only, one output directory per (seed, mode).
- **`astrouda/optimization/`**: random-search loop with a resumable `trials.jsonl`.
- **`slurm/`**: A100 job scripts that call the CLI. Cluster-specific values are in the header block of each script.

## Conventions

- Strict type annotations everywhere, verbose variable names, lots of `logger.debug` logging.
- Test with pytest; keep tests CPU-only and fast. Do not assert on wall-clock time. Use the `gpu` marker for CUDA tests.
- Work happens in git worktrees under `.worktrees/` (ignored by git). Open issues are on voetberg/AstroUDA.
