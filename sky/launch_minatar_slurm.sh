#!/bin/bash
# Launch all 6 MinAtar optimizer sweep jobs on Surrey SLURM cluster via SkyPilot.
#
# Usage:
#   export WANDB_API_KEY="your-key"
#   ./sky/launch_minatar_slurm.sh <wandb_entity> <wandb_project>
#
# Monitor:
#   sky queue              # list running jobs
#   sky logs minatar-slurm-adam   # view logs for a job
#   sky status             # cluster status

set -euo pipefail

WANDB_ENTITY="${1:?Usage: $0 <wandb_entity> <wandb_project>}"
WANDB_PROJECT="${2:?Usage: $0 <wandb_entity> <wandb_project>}"

if [ -z "${WANDB_API_KEY:-}" ]; then
    echo "Error: WANDB_API_KEY environment variable must be set."
    echo "Get it from https://wandb.ai/authorize"
    exit 1
fi

ALGOS=("adam" "redo" "regrama" "cbp" "cpr" "shrink_and_perturb")
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
YAML_PATH="$SCRIPT_DIR/minatar_slurm.yaml"

echo "Launching ${#ALGOS[@]} MinAtar jobs on Surrey SLURM (aisurrey)"
echo "  Entity:  $WANDB_ENTITY"
echo "  Project: $WANDB_PROJECT"
echo ""

for algo in "${ALGOS[@]}"; do
    echo "Launching: $algo"
    sky launch "$YAML_PATH" \
        -c "minatar-slurm-${algo}" \
        -d -y \
        --env ALGO="$algo" \
        --env WANDB_API_KEY="$WANDB_API_KEY" \
        --env WANDB_ENTITY="$WANDB_ENTITY" \
        --env WANDB_PROJECT="$WANDB_PROJECT"
    echo "  -> Submitted minatar-slurm-${algo}"
    echo ""
done

echo "All jobs submitted."
echo "  sky queue    — list jobs"
echo "  sky logs minatar-slurm-<algo> — view logs"
echo "  sky status   — cluster status"
