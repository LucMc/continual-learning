#!/bin/bash
#SBATCH --job-name=td_ant_tbconst
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --partition=3090
#SBATCH --gpus=1
#SBATCH --time=02-23:00:00
#SBATCH -o slurm.%N.%j.out
#SBATCH -e slurm.%N.%j.err
#SBATCH --array=0-11   # 6 algos x 2 seeds = 12 jobs

# Time-delayed Ant constant-per-task sweep on Surrey aisurrey.
#
# Each array task runs one (algo, seed) pair against:
#   --overall-max-obs-delay 9 --overall-max-act-delay 9
#   --delay-mode task_boundary_constant
#   --delay-info-mode none
#   --num-tasks 40 --steps-per-task 20000000     (800M total)
#
# Per task, a single (alpha, kappa) is drawn uniformly and independently from
# {0,...,8}^2; the delay is CONSTANT within the task (range width 1) but
# resampled at every task boundary. Augmented obs = [delayed_obs | act_buffer]
# only — no oracle delay channel — so the agent must infer the regime from
# action-history dynamics. Mirrors slurm_td_ant.sh's continual-04 step budget
# so step-axis comparisons are clean.
#
# Submit:
#   sbatch slurm_td_ant_const.sh
#
# Smoke task 0 only first:
#   sbatch --array=0 slurm_td_ant_const.sh

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

output_filename="td_ant_const_${SLURM_ARRAY_TASK_ID}_${algo}_seed_${seed}.out"

# Constant-per-task config — overall ranges define the i.i.d. uniform support.
overall_obs=9
overall_act=9
delay_mode="task_boundary_constant"
delay_info_mode="none"
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
echo "td_ant tbconst algo=$algo seed=$seed (40 tasks x 20M = 800M steps, info=$delay_info_mode, max=8,8)"
python "../$script_name" \
       --seed "$seed" \
       --overall-max-obs-delay "$overall_obs" \
       --overall-max-act-delay "$overall_act" \
       --delay-mode "$delay_mode" \
       --delay-info-mode "$delay_info_mode" \
       --num-tasks "$num_tasks" \
       --steps-per-task "$steps_per_task" \
       --num-envs "$num_envs" \
       --optimizers "$algo" \
       --wandb-entity "$wandb_entity" \
       --wandb-project "$wandb_project" \
       > "$output_filename" 2>&1
