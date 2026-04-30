#!/bin/bash
# Launch the time-delayed Ant experiment on Vast.ai via SkyPilot.
#
# One cluster per (config, seed). Configs:
#   --config 0,0           obs_delay=0, act_delay=0, fixed, 2x20M=40M steps
#   --config 1,1           obs_delay=1, act_delay=1, fixed, 2x20M=40M steps
#   --config continual-08  task_boundary sub-sampling from [0,8] x [0,8],
#                          20 tasks x 20M, adam only by default
#
# Usage:
#   export WANDB_API_KEY=...
#   ./sky/launch_td_ant.sh <wandb_entity> <wandb_project> --config 0,0
#   ./sky/launch_td_ant.sh <wandb_entity> <wandb_project> --config continual-08
#
# Other flags:
#   --seeds "0,1"            comma-separated seeds (default: "0,1")
#   --optimizers "adam,cbp"  comma-separated optimizers
#                            (default: "adam"; for continual-08 sweeps you'll
#                             usually pass "adam,regrama,cpr,redo,cbp,shrink_and_perturb")
#   --num-envs N             override num_envs (default 2048)
#
# Monitor:
#   sky queue
#   sky logs <cluster>
#   sky status

set -euo pipefail

WANDB_ENTITY="${1:?Usage: $0 <wandb_entity> <wandb_project> --config <0,0|1,1|continual-08> [options]}"
WANDB_PROJECT="${2:?Usage: $0 <wandb_entity> <wandb_project> --config <0,0|1,1|continual-08> [options]}"
shift 2

CONFIG=""
SEEDS_STR="0,1"
OPTIMIZERS="adam"
NUM_ENVS="2048"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config) CONFIG="$2"; shift 2 ;;
        --seeds) SEEDS_STR="$2"; shift 2 ;;
        --optimizers) OPTIMIZERS="$2"; shift 2 ;;
        --num-envs) NUM_ENVS="$2"; shift 2 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

if [ -z "$CONFIG" ]; then
    echo "Error: --config is required (0,0 | 1,1 | continual-08)" >&2
    exit 1
fi
if [ -z "${WANDB_API_KEY:-}" ]; then
    echo "Error: WANDB_API_KEY environment variable must be set." >&2
    echo "Get it from https://wandb.ai/authorize" >&2
    exit 1
fi

case "$CONFIG" in
    0,0)
        OVERALL_OBS=1; OVERALL_ACT=1
        FIXED_OBS=0; FIXED_ACT=0
        DELAY_MODE=fixed
        NUM_TASKS=2; STEPS_PER_TASK=20000000
        TAG="0obs0act-fixed"
        ;;
    1,1)
        OVERALL_OBS=2; OVERALL_ACT=2
        FIXED_OBS=1; FIXED_ACT=1
        DELAY_MODE=fixed
        NUM_TASKS=2; STEPS_PER_TASK=20000000
        TAG="1obs1act-fixed"
        ;;
    continual-04)
        OVERALL_OBS=5; OVERALL_ACT=5
        FIXED_OBS=0; FIXED_ACT=0  # ignored by task_boundary mode
        DELAY_MODE=task_boundary
        NUM_TASKS=20; STEPS_PER_TASK=20000000
        TAG="continual04"
        ;;
    continual-08)
        OVERALL_OBS=9; OVERALL_ACT=9
        FIXED_OBS=0; FIXED_ACT=0  # ignored by task_boundary mode
        DELAY_MODE=task_boundary
        NUM_TASKS=20; STEPS_PER_TASK=20000000
        TAG="continual08"
        ;;
    *)
        echo "Unknown --config: $CONFIG (expected 0,0 | 1,1 | continual-04 | continual-08)" >&2
        exit 1
        ;;
esac

# Derive a short optimizer tag for cluster naming so adam vs cpr clusters
# don't collide when we run them in parallel.
case "$OPTIMIZERS" in
    adam) OPT_TAG="" ;;     # default — keep cluster name short
    *)    OPT_TAG="-${OPTIMIZERS//,/_}" ;;
esac

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
YAML_PATH="$SCRIPT_DIR/td_ant.yaml"
IFS=',' read -ra SEEDS <<< "$SEEDS_STR"

echo "Launching td-ant: config=$CONFIG seeds=${SEEDS[*]} optimizers=$OPTIMIZERS"
echo "  Entity: $WANDB_ENTITY    Project: $WANDB_PROJECT"
echo "  overall=[obs=$OVERALL_OBS, act=$OVERALL_ACT]  fixed=[obs=$FIXED_OBS, act=$FIXED_ACT]"
echo "  mode=$DELAY_MODE  tasks=$NUM_TASKS  steps_per_task=$STEPS_PER_TASK  num_envs=$NUM_ENVS"
echo ""

for SEED in "${SEEDS[@]}"; do
    CLUSTER="td-ant-${TAG}${OPT_TAG}-s${SEED}"
    echo "Launching $CLUSTER"
    sky launch "$YAML_PATH" \
        -c "$CLUSTER" \
        -d -y --down -i 5 \
        --env OVERALL_MAX_OBS_DELAY="$OVERALL_OBS" \
        --env OVERALL_MAX_ACT_DELAY="$OVERALL_ACT" \
        --env DELAY_MODE="$DELAY_MODE" \
        --env FIXED_OBS_DELAY="$FIXED_OBS" \
        --env FIXED_ACT_DELAY="$FIXED_ACT" \
        --env NUM_TASKS="$NUM_TASKS" \
        --env STEPS_PER_TASK="$STEPS_PER_TASK" \
        --env NUM_ENVS="$NUM_ENVS" \
        --env OPTIMIZERS="$OPTIMIZERS" \
        --env SEEDS="$SEED" \
        --env WANDB_API_KEY="$WANDB_API_KEY" \
        --env WANDB_ENTITY="$WANDB_ENTITY" \
        --env WANDB_PROJECT="$WANDB_PROJECT"
    echo "  -> Submitted $CLUSTER"
    echo ""
done

echo "All $CONFIG jobs submitted."
echo "  sky queue"
echo "  sky logs td-ant-${TAG}-s<seed>"
echo "  sky status"
