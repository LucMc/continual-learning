#!/bin/bash
#SBATCH --job-name=metaworld_sweep
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --partition=3090
#SBATCH --gpus=1
#SBATCH --time=03-00:00:00
#SBATCH -o slurm.%N.%j.out
#SBATCH -e slurm.%N.%j.err

# Usage: sbatch --array=0-N slurm_metaworld_sweep.sh <algo> [wandb_entity] [wandb_project] [seed]
# To get N, run: python sweep_metaworld.py --algo <algo> --get-count
#
# Examples:
#   python sweep_metaworld.py --algo regrama --get-count
#   sbatch --array=0-134 slurm_metaworld_sweep.sh regrama lucmc metaworld_sac 0
#
#   python sweep_metaworld.py --algo bro --get-count
#   sbatch --array=0-29 slurm_metaworld_sweep.sh bro lucmc metaworld_bro 0

VENV_DIR="../../../.venv"
algo="${1:-adam}"
wandb_entity="${2:-lucmc}"
wandb_project="${3:-metaworld_sweep}"
seed="${4:-0}"

output_filename="${algo}_cfg_${SLURM_ARRAY_TASK_ID}_seed_${seed}_job_${SLURM_JOB_ID}.out"

# --- Setup environment ---
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

if [ ! -d "$VENV_DIR" ]; then
    uv venv "$VENV_DIR" --python 3.12
fi
source "$VENV_DIR/bin/activate"

# --- Run sweep config ---
echo "Running $algo config $SLURM_ARRAY_TASK_ID seed $seed"
python sweep_metaworld.py \
    --algo "$algo" \
    --config-id "$SLURM_ARRAY_TASK_ID" \
    --seed "$seed" \
    --wandb-entity "$wandb_entity" \
    --wandb-project "$wandb_project" \
    > "$output_filename" 2>&1
