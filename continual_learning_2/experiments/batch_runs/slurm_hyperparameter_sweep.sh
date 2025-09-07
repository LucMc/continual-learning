#!/bin/bash
#SBATCH --job-name=hyperparam_sweep
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=24G
#SBATCH --partition=3090
#SBATCH --gpus=1
#SBATCH --time=02-23:00:00
#SBATCH -o slurm.%N.%j.out
#SBATCH -e slurm.%N.%j.err

# Usage: sbatch --array=0-N slurm_hyperparameter_sweep.sh <algo> [wandb_entity] [wandb_project]
# To get N, run: python ../sweep_slippery_ant.py <algo> --list

VENV_DIR="../../../.venv"
algo="${1:-ccbp}"
wandb_entity="${2:-lucmc}"
wandb_project="${3:-crl_experiments}"
seed=42

# Setup environment
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

[ ! -d "$VENV_DIR" ] && uv venv "$VENV_DIR" --python 3.12
source "$VENV_DIR/bin/activate"

# Run hyperparameter configuration
echo "Running $algo config $SLURM_ARRAY_TASK_ID seed $seed"
python "sweep_slippery_ant.py" "$algo" "$SLURM_ARRAY_TASK_ID" "$seed" "$wandb_entity" "$wandb_project"
