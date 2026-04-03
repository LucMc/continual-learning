#!/bin/bash
# Launch single-task MinAtar experiments on Vast.ai via SkyPilot.
#
# Runs each MinAtar game (space_invaders, asterix, seaquest) independently
# as a single-task baseline, one sky job per (algo, task) pair.
#
# Usage:
#   export WANDB_API_KEY="your-key"
#   ./sky/launch_minatar_single.sh <wandb_entity> <wandb_project> [--algos algo1,algo2,...] [--seeds 0,1,2,...] [--tasks task1,task2,...]
#
# Examples:
#   ./sky/launch_minatar_single.sh my-team my-project
#   ./sky/launch_minatar_single.sh my-team my-project --algos adam,cpr
#   ./sky/launch_minatar_single.sh my-team my-project --algos adam --tasks asterix
#
# Monitor:
#   sky queue                          # list running jobs
#   sky logs minatar-adam-asterix      # view logs for a job
#   sky status                         # cluster status

set -euo pipefail

WANDB_ENTITY="${1:?Usage: $0 <wandb_entity> <wandb_project> [--algos ...] [--seeds ...] [--tasks ...]}"
WANDB_PROJECT="${2:?Usage: $0 <wandb_entity> <wandb_project> [--algos ...] [--seeds ...] [--tasks ...]}"
shift 2

# Defaults
ALGOS_STR="adam,redo,regrama,cbp,cpr,shrink_and_perturb"
SEEDS_STR="0,1,2,3,4"
TASKS_STR="space_invaders,asterix,seaquest"

# Parse optional flags
while [[ $# -gt 0 ]]; do
    case "$1" in
        --algos) ALGOS_STR="$2"; shift 2 ;;
        --seeds) SEEDS_STR="$2"; shift 2 ;;
        --tasks) TASKS_STR="$2"; shift 2 ;;
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
IFS=',' read -ra TASKS <<< "$TASKS_STR"
SEEDS="${SEEDS_STR//,/ }"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
YAML_PATH="$SCRIPT_DIR/minatar_single.yaml"

NUM_JOBS=$(( ${#ALGOS[@]} * ${#TASKS[@]} ))
echo "Launching $NUM_JOBS single-task MinAtar jobs on Vast.ai RTX5090"
echo "  Entity:  $WANDB_ENTITY"
echo "  Project: $WANDB_PROJECT"
echo "  Algos:   ${ALGOS[*]}"
echo "  Tasks:   ${TASKS[*]}"
echo "  Seeds:   $SEEDS"
echo ""

for task in "${TASKS[@]}"; do
    for algo in "${ALGOS[@]}"; do
        echo "Launching: $algo / $task"
        sky launch "$YAML_PATH" \
            -c "minatar-${algo}-${task}" \
            -d -y --down \
            --env ALGO="$algo" \
            --env TASK="$task" \
            --env SEEDS="$SEEDS" \
            --env WANDB_API_KEY="$WANDB_API_KEY" \
            --env WANDB_ENTITY="$WANDB_ENTITY" \
            --env WANDB_PROJECT="$WANDB_PROJECT"
        echo "  -> Submitted minatar-${algo}-${task}"
        echo ""
    done
done

echo "All $NUM_JOBS jobs submitted."
echo "  sky queue                         — list jobs"
echo "  sky logs minatar-<algo>-<task>    — view logs"
echo "  sky status                        — cluster status"
