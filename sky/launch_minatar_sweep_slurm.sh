#!/bin/bash
# Launch MinAtar hyperparameter sweep on Surrey SLURM cluster via SkyPilot.
#
# For each algo, queries the number of configs from sweep_minatar.py,
# then launches one SkyPilot job per (algo, config_id) pair.
#
# Usage:
#   export WANDB_API_KEY="your-key"
#   ./sky/launch_minatar_sweep_slurm.sh <wandb_entity> <wandb_project> [--algos algo1,algo2,...] [--seed N]
#
# Examples:
#   ./sky/launch_minatar_sweep_slurm.sh my-team my-project
#   ./sky/launch_minatar_sweep_slurm.sh my-team my-project --algos adam,cpr
#   ./sky/launch_minatar_sweep_slurm.sh my-team my-project --algos cpr --seed 42
#
# Monitor:
#   sky queue                              # list running jobs
#   sky logs minatar-sweep-adam-0           # view logs for a job
#   sky status                             # cluster status

set -euo pipefail

WANDB_ENTITY="${1:?Usage: $0 <wandb_entity> <wandb_project> [--algos algo1,algo2,...] [--seed N]}"
WANDB_PROJECT="${2:?Usage: $0 <wandb_entity> <wandb_project> [--algos algo1,algo2,...] [--seed N]}"
shift 2

# Defaults
ALGOS_STR="adam,muon,redo,regrama,cbp,cpr,shrink_and_perturb"
SEED="0"

# Parse optional flags
while [[ $# -gt 0 ]]; do
    case "$1" in
        --algos) ALGOS_STR="$2"; shift 2 ;;
        --seed)  SEED="$2";      shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [ -z "${WANDB_API_KEY:-}" ]; then
    echo "Error: WANDB_API_KEY environment variable must be set."
    echo "Get it from https://wandb.ai/authorize"
    exit 1
fi

IFS=',' read -ra ALGOS <<< "$ALGOS_STR"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
YAML_PATH="$SCRIPT_DIR/minatar_sweep_slurm.yaml"
SWEEP_SCRIPT="experiments/batch_runs/sweep_minatar.py"

TOTAL_JOBS=0

echo "Launching MinAtar hyperparameter sweep on Surrey SLURM (aisurrey)"
echo "  Entity:  $WANDB_ENTITY"
echo "  Project: $WANDB_PROJECT"
echo "  Seed:    $SEED"
echo "  Algos:   ${ALGOS[*]}"
echo ""

for algo in "${ALGOS[@]}"; do
    # Get number of configs for this algo
    COUNT=$(python "$SWEEP_SCRIPT" --algo "$algo" --get-count 2>/dev/null | grep "Total configurations:" | awk '{print $NF}')
    MAX_ID=$((COUNT - 1))

    echo "=== $algo: $COUNT configs (IDs 0-$MAX_ID) ==="

    for config_id in $(seq 0 "$MAX_ID"); do
        echo "  Launching: $algo config $config_id"
        sky launch "$YAML_PATH" \
            -c "minatar-sweep-${algo}-${config_id}" \
            -d -y \
            --env ALGO="$algo" \
            --env CONFIG_ID="$config_id" \
            --env SEED="$SEED" \
            --env WANDB_API_KEY="$WANDB_API_KEY" \
            --env WANDB_ENTITY="$WANDB_ENTITY" \
            --env WANDB_PROJECT="$WANDB_PROJECT"
        TOTAL_JOBS=$((TOTAL_JOBS + 1))
    done
    echo ""
done

echo "All jobs submitted: $TOTAL_JOBS total."
echo "  sky queue                              — list jobs"
echo "  sky logs minatar-sweep-<algo>-<id>     — view logs"
echo "  sky status                             — cluster status"
