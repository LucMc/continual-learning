#!/bin/bash
#SBATCH --job-name=metaworld_bro
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --partition=3090
#SBATCH --gpus=1
#SBATCH --time=03-00:00:00
#SBATCH -o slurm.%N.%j.out
#SBATCH -e slurm.%N.%j.err
#SBATCH --array=0-14   # 3 optimizers x 5 seeds = 15 jobs

# Usage: sbatch slurm_metaworld_bro.sh [wandb_entity] [wandb_project]
# Example: sbatch --array=0-14 slurm_metaworld_bro.sh lucmc metaworld_bro
#
# Optimizers: adamw, redo, regrama
# Note: CBP, CCBP, and ShrinkAndPerturb are not yet compatible with BRO's architecture
# Seeds: 0, 1, 2, 3, 4

VENV_DIR="../../.venv"
wandb_entity="${1:-lucmc}"
wandb_project="${2:-metaworld_bro}"

# Optimizers and seeds arrays
optimizers=(adamw redo regrama)
seeds=(0 1 2 3 4)

num_optimizers=${#optimizers[@]}
num_seeds=${#seeds[@]}

# Map flat array index to (optimizer_idx, seed_idx)
optimizer_idx=$((SLURM_ARRAY_TASK_ID / num_seeds))
seed_idx=$((SLURM_ARRAY_TASK_ID % num_seeds))

optimizer="${optimizers[$optimizer_idx]}"
seed="${seeds[$seed_idx]}"

output_filename="bro_${optimizer}_seed_${seed}_job_${SLURM_JOB_ID}.out"

# --- Setup environment ---
# Change to project root (2 levels up from batch_runs/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."

if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

if [ ! -d "$VENV_DIR" ]; then
    uv venv "$VENV_DIR" --python 3.12
fi
source "$VENV_DIR/bin/activate"

# --- Install package ---
uv pip install -e ".[cuda12]"

# --- Run BRO experiment ---
echo "Running BRO on MetaWorld MT10 with optimizer=$optimizer, seed=$seed"
python -m experiments.metaworld_mt10 \
    --seed "$seed" \
    --optimizer "$optimizer" \
    --wandb-entity "$wandb_entity" \
    --wandb-project "$wandb_project" \
    > "$output_filename" 2>&1
