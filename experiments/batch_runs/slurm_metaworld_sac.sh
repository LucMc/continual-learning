#!/bin/bash
#SBATCH --job-name=metaworld_sac
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --partition=3090
#SBATCH --gpus=1
#SBATCH --time=03-00:00:00
#SBATCH -o slurm.%N.%j.out
#SBATCH -e slurm.%N.%j.err
#SBATCH --array=0-31   # 16 algos * 2 seeds = 32 jobs

# Usage: sbatch slurm_metaworld_sac.sh [wandb_entity] [wandb_project]
# Example: sbatch --array=0-31 slurm_metaworld_sac.sh lucmc metaworld_sac

VENV_DIR="../../.venv"
wandb_entity="${1:-lucmc}"
wandb_project="${2:-metaworld_sac}"

# Algorithms and seeds
algos=("adam" "redo" "regrama" "cbp" "ccbp" "ccbpl" "ccbph" "shrink_and_perturb" "muon" "muon_redo" "muon_regrama" "muon_cbp" "muon_ccbp" "muon_ccbpl" "muon_ccbph" "muon_shrink_and_perturb")
seeds=(0 1)

num_algos=${#algos[@]}
num_seeds=${#seeds[@]}

# Compute indices from SLURM_ARRAY_TASK_ID
algo_idx=$(( SLURM_ARRAY_TASK_ID % num_algos ))
seed_idx=$(( SLURM_ARRAY_TASK_ID / num_algos ))

algo="${algos[$algo_idx]}"
seed="${seeds[$seed_idx]}"

output_filename="sac_${algo}_seed_${seed}_job_${SLURM_JOB_ID}.out"

# --- Setup environment ---
PROJECT_ROOT="$(cd "$VENV_DIR/.." && pwd)"
source "$VENV_DIR/bin/activate"

# --- Run SAC experiment ---
echo "Running SAC on MetaWorld MT10 with algo=$algo seed=$seed"
python "$PROJECT_ROOT/experiments/metaworld_sac.py" \
    --include "$algo" \
    --seed "$seed" \
    --wandb-entity "$wandb_entity" \
    --wandb-project "$wandb_project" \
    > "$output_filename" 2>&1
