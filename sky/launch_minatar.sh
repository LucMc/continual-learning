#!/bin/bash
# Launch MinAtar optimizer sweep jobs on Vast.ai via SkyPilot.
#
# Usage:
#   export WANDB_API_KEY="your-key"
#   ./sky/launch_minatar.sh <wandb_entity> <wandb_project> [--algos algo1,algo2,...] [--seeds 0,1,2,...]
#
# Examples:
#   ./sky/launch_minatar.sh my-team my-project
#   ./sky/launch_minatar.sh my-team my-project --algos adam,cpr
#   ./sky/launch_minatar.sh my-team my-project --algos adam --seeds 0,1,2
#
# Monitor:
#   sky queue              # list running jobs
#   sky logs minatar-adam   # view logs for a job
#   sky status             # cluster status

set -euo pipefail

WANDB_ENTITY="${1:?Usage: $0 <wandb_entity> <wandb_project> [--algos algo1,algo2,...] [--seeds 0,1,2,...]}"
WANDB_PROJECT="${2:?Usage: $0 <wandb_entity> <wandb_project> [--algos algo1,algo2,...] [--seeds 0,1,2,...]}"
shift 2

# Defaults
ALGOS_STR="adam,redo,regrama,cbp,cpr,shrink_and_perturb"
SEEDS_STR="0,1,2,3,4"

# Parse optional flags
while [[ $# -gt 0 ]]; do
    case "$1" in
        --algos) ALGOS_STR="$2"; shift 2 ;;
        --seeds) SEEDS_STR="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [ -z "${WANDB_API_KEY:-}" ]; then
    echo "Error: WANDB_API_KEY environment variable must be set."
    echo "Get it from https://wandb.ai/authorize"
    exit 1
fi

# Convert comma-separated strings to arrays
IFS=',' read -ra ALGOS <<< "$ALGOS_STR"
# Convert comma-separated seeds to space-separated (for the YAML run loop)
SEEDS="${SEEDS_STR//,/ }"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
YAML_PATH="$SCRIPT_DIR/minatar.yaml"

echo "Launching ${#ALGOS[@]} MinAtar jobs on Vast.ai RTX5060"
echo "  Entity:  $WANDB_ENTITY"
echo "  Project: $WANDB_PROJECT"
echo "  Algos:   ${ALGOS[*]}"
echo "  Seeds:   $SEEDS"
echo ""

for algo in "${ALGOS[@]}"; do
    echo "Launching: $algo"
    sky launch "$YAML_PATH" \
        -c "minatar-${algo}" \
        -d -y --down \
        --env ALGO="$algo" \
        --env SEEDS="$SEEDS" \
        --env WANDB_API_KEY="$WANDB_API_KEY" \
        --env WANDB_ENTITY="$WANDB_ENTITY" \
        --env WANDB_PROJECT="$WANDB_PROJECT"
    echo "  -> Submitted minatar-${algo}"
    echo ""
done

echo "All jobs submitted."
echo "  sky queue    — list jobs"
echo "  sky logs minatar-<algo> — view logs"
echo "  sky status   — cluster status"
