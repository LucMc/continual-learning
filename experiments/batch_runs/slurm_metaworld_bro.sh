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
#SBATCH --array=0-4   # 5 seeds by default

# Usage: sbatch slurm_metaworld_bro.sh [wandb_entity] [wandb_project]
# Example: sbatch --array=0-4 slurm_metaworld_bro.sh lucmc metaworld_bro

VENV_DIR="../../../.venv"
wandb_entity="${1:-lucmc}"
wandb_project="${2:-metaworld_bro}"

# Seeds array
seeds=(0 1 2 3 4)
seed="${seeds[$SLURM_ARRAY_TASK_ID]}"

output_filename="bro_seed_${seed}_job_${SLURM_JOB_ID}.out"

# --- Setup environment ---
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

if [ ! -d "$VENV_DIR" ]; then
    uv venv "$VENV_DIR" --python 3.12
fi
source "$VENV_DIR/bin/activate"

# --- Run BRO experiment ---
echo "Running BRO on MetaWorld MT10 with seed=$seed"
python -m experiments.metaworld_mt10 \
    --seed "$seed" \
    --wandb-entity "$wandb_entity" \
    --wandb-project "$wandb_project" \
    > "$output_filename" 2>&1
