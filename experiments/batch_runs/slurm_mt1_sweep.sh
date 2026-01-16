#!/bin/bash
#SBATCH --job-name=mt1_sweep
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --partition=2080ti
#SBATCH --gpus=1
#SBATCH --time=01-00:00:00
#SBATCH -o slurm.%N.%j.out
#SBATCH -e slurm.%N.%j.err

# MT1 Hyperparameter Sweep for Single-Task RL
#
# This script runs hyperparameter sweeps for reset methods on MetaWorld MT1.
# Each SLURM array task runs one (config, seed) combination for a given algorithm and task.
#
# Usage:
#   # First, get the number of configs for your algorithm:
#   python sweep_metaworld_mt1.py --algo cbp --get-count
#
#   # Then submit the sweep (configs × seeds):
#   # Example: cbp has 24 configs × 3 seeds = 72 jobs per task
#   sbatch --array=0-71 slurm_mt1_sweep.sh cbp reach-v3 lucmc MT1_sweep
#
#   # Example: redo has 20 configs × 3 seeds = 60 jobs per task
#   sbatch --array=0-59 slurm_mt1_sweep.sh redo reach-v3 lucmc MT1_sweep
#
#   # To run all 10 MT10 tasks for one algorithm:
#   for task in reach-v3 push-v3 pick-place-v3 door-open-v3 drawer-open-v3 \
#               drawer-close-v3 button-press-topdown-v3 peg-insert-side-v3 \
#               window-open-v3 window-close-v3; do
#       sbatch --array=0-71 slurm_mt1_sweep.sh cbp $task lucmc MT1_sweep
#   done
#
# Config counts per algorithm (× 3 seeds):
#   adam:               3 configs →   9 jobs per task  (array 0-8)
#   redo:              20 configs →  60 jobs per task  (array 0-59)
#   regrama:           24 configs →  72 jobs per task  (array 0-71)
#   cbp:               24 configs →  72 jobs per task  (array 0-71)
#   ccbp:              12 configs →  36 jobs per task  (array 0-35)
#   shrink_and_perturb: 27 configs →  81 jobs per task  (array 0-80)
#
# Total: 110 configs × 3 seeds × 10 tasks = 3,300 jobs
#
# Paper-recommended optimal values:
#   ReDo:    τ=0.025-0.05 (continuous), interval=1000-2000
#   ReGraMa: α=1e-4, reset_rate=0.01
#   CBP:     ρ=1e-4, maturity=1000-2000
#
# CRITICAL: All methods must reset optimizer moments for recycled neurons!

VENV_DIR="../../.venv"
algo="${1:-cbp}"
task="${2:-reach-v3}"
wandb_entity="${3:-lucmc}"
wandb_project="${4:-MT1_sweep}"

# Number of seeds to sweep
NUM_SEEDS=3

# --- Calculate config_id and seed from array task ID ---
# Layout: SLURM_ARRAY_TASK_ID = config_id * NUM_SEEDS + seed_idx
config_id=$((SLURM_ARRAY_TASK_ID / NUM_SEEDS))
seed_idx=$((SLURM_ARRAY_TASK_ID % NUM_SEEDS))

# Map seed_idx to actual seed values
seeds=(0 1 2)
seed="${seeds[$seed_idx]}"

output_filename="mt1_${task}_${algo}_cfg${config_id}_seed${seed}_job${SLURM_JOB_ID}.out"

# --- Setup environment ---
cd "$SLURM_SUBMIT_DIR"

if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

if [ ! -d "$VENV_DIR" ]; then
    uv venv "$VENV_DIR" --python 3.12
fi
source "$VENV_DIR/bin/activate"

# --- Run sweep config ---
echo "=============================================="
echo "MT1 Hyperparameter Sweep"
echo "=============================================="
echo "Algorithm: $algo"
echo "Task: $task"
echo "Config ID: $config_id"
echo "Seed: $seed"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "SLURM Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "=============================================="

python sweep_metaworld_mt1.py \
    --algo "$algo" \
    --task "$task" \
    --config-id "$config_id" \
    --seed "$seed" \
    --wandb-entity "$wandb_entity" \
    --wandb-project "$wandb_project" \
    > "$output_filename" 2>&1
