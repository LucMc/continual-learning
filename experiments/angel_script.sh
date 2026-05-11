#!/bin/bash
#SBATCH --job-name=crl-layernorm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --partition=3090
#SBATCH --gpus=1
#SBATCH --time=00-03:00:00
#SBATCH -o slurm.%N.%j.out
#SBATCH -e slurm.%N.%j.err
#SBATCH --array=1-59 # num_aglos * num_seeds * num_scripts - 1 59

# --- Configuration ---
VENV_DIR=".venv"
script_names=("slippery_ant_layernorm.py" "slippery_humanoid_layernorm.py")

# Algorithms and seeds
algos=("adam" "adamw")
seeds=("0" "1" "2" "3" "4" "5" "6" "7" "8" "9" "10" "11" "12" "13" "14")

num_algos=${#algos[@]}
num_seeds=${#seeds[@]}
num_scripts=${#script_names[@]}

# Compute indices from SLURM_ARRAY_TASK_ID
script_idx=$(( SLURM_ARRAY_TASK_ID / (num_algos * num_seeds) ))
remain=$(( SLURM_ARRAY_TASK_ID % (num_algos * num_seeds) ))

algo_idx=$(( remain / num_seeds ))
seed_idx=$(( remain % num_seeds ))

algo="${algos[$algo_idx]}"
seed="${seeds[$seed_idx]}"
script_name="${script_names[$script_idx]}"

output_filename="script_${script_name}_${SLURM_ARRAY_TASK_ID}_${algo}_seed_${seed}.out"

# WandB settings (fill in your values)
wandb_entity="lucmc"
wandb_project="crl_layernorm"

# --- Setup environment (uv or venv) ---
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

mkdir -p "$(dirname "$VENV_DIR")"
LOCKFILE="$VENV_DIR.lock"

exec 9>"$LOCKFILE"
flock 9
if [ ! -d "$VENV_DIR" ]; then
    uv venv "$VENV_DIR" --python 3.12
    source "$VENV_DIR/bin/activate"
    uv pip install -e ".[cuda12]"
    deactivate
fi
flock -u 9

source "$VENV_DIR/bin/activate"

# --- Run experiment ---
echo "Running $script_name for algo=$algo seed=$seed"
python "experiments/$script_name" --include "$algo" --seed "$seed" \
       --wandb-entity "$wandb_entity" --wandb-project "$wandb_project" \
       > "$output_filename" 2>&1
