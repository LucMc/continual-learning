#!/bin/bash
# Launch MinAtar hyperparameter sweep on Vast.ai via SkyPilot.
#
# Launches one cluster per algorithm, running all configs for that algo
# sequentially on a single GPU.
#
# Usage:
#   export WANDB_API_KEY="your-key"
#   ./sky/launch_minatar_sweep_vast.sh <wandb_entity> <wandb_project> [--algos algo1,algo2,...] [--seed N]
#
# Examples:
#   ./sky/launch_minatar_sweep_vast.sh my-team my-project
#   ./sky/launch_minatar_sweep_vast.sh my-team my-project --algos adam,cpr
#   ./sky/launch_minatar_sweep_vast.sh my-team my-project --algos cpr --seed 42
#
# Monitor:
#   sky queue                                # list running jobs
#   sky logs minatar-sweep-adam              # view logs for a job
#   sky status                               # cluster status

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
YAML_PATH="$SCRIPT_DIR/minatar_sweep_vast.yaml"
SWEEP_SCRIPT="experiments/batch_runs/sweep_minatar.py"

TOTAL_JOBS=0

echo "Launching MinAtar hyperparameter sweep on Vast.ai (RTX5090)"
echo "  Entity:  $WANDB_ENTITY"
echo "  Project: $WANDB_PROJECT"
echo "  Seed:    $SEED"
echo "  Algos:   ${ALGOS[*]}"
echo ""

for algo in "${ALGOS[@]}"; do
    # Get number of configs for this algo
    COUNT=$(python "$SWEEP_SCRIPT" --algo "$algo" --get-count 2>/dev/null | grep "Total configurations:" | awk '{print $NF}')
    MAX_ID=$((COUNT - 1))

    echo "Launching: $algo ($COUNT configs, IDs 0-$MAX_ID)"
    sky launch "$YAML_PATH" \
        -c "minatar-sweep-${algo}" \
        -d -y --down \
        --env ALGO="$algo" \
        --env SEED="$SEED" \
        --env CONFIG_START="0" \
        --env CONFIG_END="$MAX_ID" \
        --env WANDB_API_KEY="$WANDB_API_KEY" \
        --env WANDB_ENTITY="$WANDB_ENTITY" \
        --env WANDB_PROJECT="$WANDB_PROJECT"
    echo "  -> Submitted minatar-sweep-${algo}"
    echo ""
    TOTAL_JOBS=$((TOTAL_JOBS + 1))
done

echo "All jobs submitted: $TOTAL_JOBS clusters."
echo "  sky queue                          — list jobs"
echo "  sky logs minatar-sweep-<algo>      — view logs"
echo "  sky status                         — cluster status"
