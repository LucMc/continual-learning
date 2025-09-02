#!/bin/bash
#SBATCH --job-name=multi_algo_train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=24G
#SBATCH --partition=3090
#SBATCH --gpus=1
#SBATCH --time=02-23:00:00
#SBATCH -o slurm.%N.%j.out
#SBATCH -e slurm.%N.%j.err
#SBATCH --array=0-3   # adjust based on num_algos * num_seeds - 1

# --- Configuration ---
VENV_DIR="../../../.venv"
script_name="${1:-slippery_ant.py}"   # defaults to slippery_ant.py if not provided

# Algorithms and seeds
algos=("redo" "regrama" "adam" "cbp" "ccbp")
seeds=(0)

num_algos=${#algos[@]}
num_seeds=${#seeds[@]}

# Compute indices from SLURM_ARRAY_TASK_ID
algo_idx=$(( SLURM_ARRAY_TASK_ID % num_algos ))
seed_idx=$(( SLURM_ARRAY_TASK_ID / num_algos ))

algo="${algos[$algo_idx]}"
seed="${seeds[$seed_idx]}"

output_filename="task_${SLURM_ARRAY_TASK_ID}_${algo}_seed_${seed}.out"

# WandB settings (fill in your values)
wandb_entity="lucmc"
wandb_project="crl_experiments"

# --- Setup environment (uv or venv) ---
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

if [ ! -d "$VENV_DIR" ]; then
    uv venv "$VENV_DIR" --python 3.12
fi
source "$VENV_DIR/bin/activate"

# Optionally install deps if not synced
# uv sync
# uv pip install -e ".[cuda12]"

# --- Run experiment ---
echo "Running $script_name for algo=$algo seed=$seed"
python "../$script_name" --include "$algo" --seed "$seed" \
       --wandb-entity "$wandb_entity" --wandb-project "$wandb_project" \
       > "$output_filename" 2>&1

