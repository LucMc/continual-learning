#!/bin/bash
#SBATCH --job-name=td_mt1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=24G
#SBATCH --partition=3090
#SBATCH --gpus=1
#SBATCH --time=02-23:00:00
#SBATCH -o slurm.%N.%j.out
#SBATCH -e slurm.%N.%j.err
#SBATCH --array=0-9   # num_tasks * num_optimizers * num_seeds - 1

# Usage: sbatch slurm_td_mt1.sh [script_name] [max_obs_delay] [max_act_delay] [delay_mode]
# Example (validation run, all 10 tasks, adam, seed 0):
#   sbatch slurm_td_mt1.sh td_worldmt1.py 0 0 fixed
# Example (continual delays, 3 seeds × 10 tasks = 30 jobs):
#   sbatch --array=0-29 slurm_td_mt1.sh td_worldmt1.py 4 4 continual

# --- Configuration ---
VENV_DIR="../../../.venv"
script_name="${1:-td_worldmt1.py}"
max_obs_delay="${2:-0}"
max_act_delay="${3:-0}"
delay_mode="${4:-fixed}"
resample_every="${5:-10000}"

# MT1 tasks, optimizers, seeds
tasks=("reach-v3" "push-v3" "pick-place-v3" "door-open-v3" "drawer-open-v3" \
       "drawer-close-v3" "button-press-topdown-v3" "peg-insert-side-v3" \
       "window-open-v3" "window-close-v3")
optimizers=("adam")
seeds=(0)

num_tasks=${#tasks[@]}
num_optimizers=${#optimizers[@]}
num_seeds=${#seeds[@]}

# Compute indices from SLURM_ARRAY_TASK_ID
task_idx=$(( SLURM_ARRAY_TASK_ID % num_tasks ))
opt_idx=$(( (SLURM_ARRAY_TASK_ID / num_tasks) % num_optimizers ))
seed_idx=$(( SLURM_ARRAY_TASK_ID / (num_tasks * num_optimizers) ))

task="${tasks[$task_idx]}"
optimizer="${optimizers[$opt_idx]}"
seed="${seeds[$seed_idx]}"

output_filename="task_${SLURM_ARRAY_TASK_ID}_${task}_${optimizer}_obs${max_obs_delay}_act${max_act_delay}_${delay_mode}_seed_${seed}.out"

# WandB settings
wandb_entity="lucmc"
wandb_project="TD MT1 results"

# --- Setup environment (uv or venv) ---
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

if [ ! -d "$VENV_DIR" ]; then
    uv venv "$VENV_DIR" --python 3.12
fi
source "$VENV_DIR/bin/activate"
uv pip install -e "../../.[cuda12]"

# --- Run experiment ---
echo "Running $script_name for task=$task optimizer=$optimizer seed=$seed obs_delay=$max_obs_delay act_delay=$max_act_delay mode=$delay_mode"
python "../$script_name" \
       --task-name "$task" \
       --optimizer "$optimizer" \
       --seed "$seed" \
       --max-obs-delay "$max_obs_delay" \
       --max-act-delay "$max_act_delay" \
       --delay-mode "$delay_mode" \
       --resample-every "$resample_every" \
       --wandb-entity "$wandb_entity" \
       --wandb-project "$wandb_project" \
       > "$output_filename" 2>&1
