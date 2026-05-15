# Continual Learning
In this project we introduce a new reset method, CPR (Calibrated Partial Resets), where calibrated transformations are used to partially reset low gradient neurons. All baseline reset methods and CPR can be found `continual_learning/optim/` as one file Optax implementations. Note that although implemented in Optax our reset methods operate over parameters, with some baselines also taking in additional unconventional inputs such as activation features and base optimizer state. Because of this, we recommend using our custom optax chain to attach reset methods (`continual_learning/utils/optim.py`). Furthermore we also have a custom trainstate to handle passing the extra parameters needed in the baselines (`continual_learning/utils/training.py`).

## Installation
 * Clone repository: `git clone XXX`
 * Install [uv](https://docs.astral.sh/uv/getting-started/installation/)
 * Install with: `uv pip install -e .`

## Example Usage
```bash
uv run python -m experiments.slippery_ant --include cpr
```
With W&B logging
```bash
uv run python -m experiments.slippery_ant --include cpr --wandb-mode online --wandb-entity XXX --wandb-project XXX
```
