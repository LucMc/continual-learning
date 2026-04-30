#!/bin/bash
#SBATCH --job-name=td_ant_continual04
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=24G
#SBATCH --partition=3090
#SBATCH --gpus=1
#SBATCH --time=02-23:00:00
#SBATCH -o slurm.%N.%j.out
#SBATCH -e slurm.%N.%j.err
#SBATCH --array=0-11   # 6 algos x 2 seeds = 12 jobs

# Time-delayed Ant continual-04 sweep on Surrey aisurrey.
#
# Each array task runs one (algo, seed) pair against the continual-04 config:
#   --overall-max-obs-delay 5 --overall-max-act-delay 5
#   --delay-mode task_boundary
#   --num-tasks 40 --steps-per-task 20000000     (800M total)
# i.e. delay sub-intervals sampled from [0,4]x[0,4] at task boundaries.
#
# Submit:
#   sbatch slurm_td_ant.sh
#
# Smoke task 0 only first:
#   sbatch --array=0 slurm_td_ant.sh

# --- Configuration ---
VENV_DIR="../../../.venv"
script_name="${1:-td_ant.py}"

algos=("adam" "regrama" "cpr" "redo" "cbp" "shrink_and_perturb")
seeds=(0 1)

num_algos=${#algos[@]}
num_seeds=${#seeds[@]}

algo_idx=$(( SLURM_ARRAY_TASK_ID % num_algos ))
seed_idx=$(( SLURM_ARRAY_TASK_ID / num_algos ))
algo="${algos[$algo_idx]}"
seed="${seeds[$seed_idx]}"

output_filename="td_ant_${SLURM_ARRAY_TASK_ID}_${algo}_seed_${seed}.out"

# continual-04 config (mirrors sky/launch_td_ant.sh:73-78).
overall_obs=5
overall_act=5
delay_mode="task_boundary"
num_tasks=40
steps_per_task=20000000
num_envs=2048

wandb_entity="lucmc"
wandb_project="TD Ant"

# --- Setup environment (uv) ---
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
echo "td_ant continual-04 algo=$algo seed=$seed (40 tasks x 20M = 800M steps)"
python "../$script_name" \
       --seed "$seed" \
       --overall-max-obs-delay "$overall_obs" \
       --overall-max-act-delay "$overall_act" \
       --delay-mode "$delay_mode" \
       --num-tasks "$num_tasks" \
       --steps-per-task "$steps_per_task" \
       --num-envs "$num_envs" \
       --optimizers "$algo" \
       --wandb-entity "$wandb_entity" \
       --wandb-project "$wandb_project" \
       > "$output_filename" 2>&1
