#!/bin/bash
#SBATCH --job-name=metaworld_mt1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --partition=2080ti
#SBATCH --gpus=1
#SBATCH --time=03-00:00:00
#SBATCH -o slurm.%N.%j.out
#SBATCH -e slurm.%N.%j.err
#SBATCH --array=0-249   # 5 methods x 10 tasks x 5 seeds = 250 jobs

# Usage: sbatch slurm_metaworld_mt1.sh [wandb_entity]
# Example: sbatch slurm_metaworld_mt1.sh lucmc
#
# This script runs SAC on each of the 10 MetaWorld MT10 tasks individually (as MT1)
# with 5 different reset methods across 5 seeds.
#
# Total runs: 5 methods x 10 tasks x 5 seeds = 250 runs
#
# Methods: adam, cbp, ccbp, regrama, shrink_and_perturb
# Tasks: reach-v3, push-v3, pick-place-v3, door-open-v3, drawer-open-v3,
#        drawer-close-v3, button-press-topdown-v3, peg-insert-side-v3,
#        window-open-v3, window-close-v3
# Seeds: 0, 1, 2, 3, 4
#
# Results are logged to W&B project "MT1 results" with:
#   - Group: task name (e.g., "reach-v3")
#   - Run name: sac_{task}_{method}_{seed}

VENV_DIR="../../.venv"
wandb_entity="${1:-lucmc}"
wandb_project="MT1 results"

# Methods (5)
methods=("adam" "cbp" "ccbp" "regrama" "shrink_and_perturb")

# MT10 tasks (10)
tasks=(
    "reach-v3"
    "push-v3"
    "pick-place-v3"
    "door-open-v3"
    "drawer-open-v3"
    "drawer-close-v3"
    "button-press-topdown-v3"
    "peg-insert-side-v3"
    "window-open-v3"
    "window-close-v3"
)

# Seeds (5)
seeds=(0 1 2 3 4)

num_methods=${#methods[@]}
num_tasks=${#tasks[@]}
num_seeds=${#seeds[@]}

# Map SLURM_ARRAY_TASK_ID to (method_idx, task_idx, seed_idx)
# Layout: iterate over seeds -> tasks -> methods
# Total = num_methods * num_tasks * num_seeds = 5 * 10 * 5 = 250
#
# method_idx = SLURM_ARRAY_TASK_ID % num_methods
# task_idx = (SLURM_ARRAY_TASK_ID / num_methods) % num_tasks
# seed_idx = SLURM_ARRAY_TASK_ID / (num_methods * num_tasks)

method_idx=$((SLURM_ARRAY_TASK_ID % num_methods))
task_idx=$(((SLURM_ARRAY_TASK_ID / num_methods) % num_tasks))
seed_idx=$((SLURM_ARRAY_TASK_ID / (num_methods * num_tasks)))

method="${methods[$method_idx]}"
task="${tasks[$task_idx]}"
seed="${seeds[$seed_idx]}"

output_filename="mt1_${task}_${method}_seed${seed}_job${SLURM_JOB_ID}.out"

# --- Setup environment ---
cd "$SLURM_SUBMIT_DIR"

# Derive project root from venv path (VENV_DIR is ../../.venv relative to batch_runs/)
PROJECT_ROOT="$(cd "$VENV_DIR/.." && pwd)"
source "$VENV_DIR/bin/activate"

# --- Run MT1 experiment ---
echo "Running SAC MT1: task=$task, method=$method, seed=$seed"
python "$PROJECT_ROOT/experiments/metaworld_mt1.py" \
    --task-name "$task" \
    --optimizer "$method" \
    --seed "$seed" \
    --wandb-entity "$wandb_entity" \
    --wandb-project "$wandb_project" \
    > "$output_filename" 2>&1
