# Continual Learning

This repository contains the experimental code we use to study optimisation and plasticity in continual learning scenarios. The core library is written in JAX/Flax and provides reusable building blocks for continual supervised and reinforcement learning research.

## Requirements
- Python 3.12
- JAX build for your accelerator (`pip install .[cpu]`, `[cuda12]`, `[cuda13]`, or `[tpu]`)
- Optional: Weights & Biases for experiment logging

## Installation
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .[cpu]
```
Replace `[cpu]` with another extra if you plan to run on GPU/TPU hardware.

## Running Experiments
All runnable entry points live in `experiments/`. Each script exposes a Tyro CLI, so you can inspect options with `--help`.

```bash
# Disable logging to quickly smoke-test a run
python -m experiments.split_mnist --wandb_mode disabled

# Run the full slippery ant continual RL benchmark
python -m experiments.slippery_ant --wandb_project <project> --wandb_entity <entity>
```
You can include/exclude optimisers and adjust other hyper-parameters through the corresponding CLI flags. Results, checkpoints, and logging are handled inside the scripts.

## Repository Layout
- `continual_learning/` – reusable library modules (configs, models, trainers, optimisers, utils)
- `experiments/` – experiment drivers and batch/sweep helpers
- `tests/` – lightweight regression tests for core utilities

## Development Notes
- Run `pytest` to execute the test suite.
- We use Ruff and Black (configured via `pyproject.toml`) for style checks.
- Plotting and analysis utilities live in `continual_learning/plots/`.

Feel free to open issues or PRs if you discover bugs or have ideas for new continual learning benchmarks.
