#!/bin/bash
#SBATCH --job-name=td_ant_ramp20
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

# Time-delayed Ant alternating-ramp sweep on Surrey aisurrey.
#
# Each array task runs one (algo, seed) pair against the ramp config:
#   --delay-mode ramp --ramp-num-increments 20
#   --delay-info-mode none
#   --steps-per-task 20000000     (21 tasks x 20M = 420M total)
#
# Per-task delay schedule:
#   (0,0) (1,0) (1,1) (2,1) (2,2) ... (10,9) (10,10)
#
# Augmented obs = [delayed_obs | act_buffer]; the agent is NOT told the
# current alpha/kappa (delay_info_mode=none), so it has to infer the regime
# from action-history dynamics — the actual CL/plasticity benchmark.
#
# Submit:
#   sbatch slurm_td_ant_ramp.sh
#
# Smoke task 0 only first:
#   sbatch --array=0 slurm_td_ant_ramp.sh

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

output_filename="td_ant_ramp_${SLURM_ARRAY_TASK_ID}_${algo}_seed_${seed}.out"

# Ramp config — overall_max_*_delay are auto-derived from the schedule.
delay_mode="ramp"
delay_info_mode="none"
ramp_num_increments=20
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
echo "td_ant ramp algo=$algo seed=$seed (21 tasks x 20M = 420M steps, info=$delay_info_mode)"
python "../$script_name" \
       --seed "$seed" \
       --delay-mode "$delay_mode" \
       --delay-info-mode "$delay_info_mode" \
       --ramp-num-increments "$ramp_num_increments" \
       --steps-per-task "$steps_per_task" \
       --num-envs "$num_envs" \
       --optimizers "$algo" \
       --wandb-entity "$wandb_entity" \
       --wandb-project "$wandb_project" \
       > "$output_filename" 2>&1
