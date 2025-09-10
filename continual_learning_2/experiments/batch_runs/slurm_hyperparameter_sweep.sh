#!/bin/bash
#SBATCH --job-name=hyperparam_sweep
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=24G
#SBATCH --partition=3090
#SBATCH --gpus=1
#SBATCH --time=00-23:00:00
#SBATCH -o slurm.%N.%j.out
#SBATCH -e slurm.%N.%j.err

# Usage: sbatch --array=0-N slurm_hyperparameter_sweep.sh <algo> [wandb_entity] [wandb_project] [seed]
# To get N, run: ./get_sweep_size.sh <algo>
# Example: sbatch --array=0-159 slurm_hyperparameter_sweep.sh ccbp lucmc crl_experiments 42

VENV_DIR="../../../.venv"
algo="${1:-ccbp}"
script="${2:-perm_mnist}"
wandb_entity="${3:-lucmc}"
wandb_project="${4:-crl_experiments}"
seed="${5:-42}"

# Setup environment
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

[ ! -d "$VENV_DIR" ] && uv venv "$VENV_DIR" --python 3.12
source "$VENV_DIR/bin/activate"

# Run hyperparameter configuration
echo "Running $algo config $SLURM_ARRAY_TASK_ID seed $seed"
python sweep_$script.py --algo "$algo" --config-id "$SLURM_ARRAY_TASK_ID" --seed "$seed" --wandb-entity "$wandb_entity" --wandb-project "$wandb_project"
