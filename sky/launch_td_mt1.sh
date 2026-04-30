#!/bin/bash
# Launch time-delayed MetaWorld MT1 sweep on Vast.ai via SkyPilot.
#
# Sweeps over delay configs × delay-embedding types for a single (task,
# optimizer): one cluster per (max_obs_delay, max_act_delay, delay_emb_type)
# combo; each cluster runs all seeds sequentially.
# Wandb groups runs by "<act>act<obs>obs_<delay_mode>_<delay_emb_type>" so
# augmented (one_hot) and unseen (none) curves auto-group separately in the UI.
#
# Usage:
#   export WANDB_API_KEY="your-key"
#   ./sky/launch_td_mt1.sh <wandb_entity> <wandb_project> [options]
#
# Options:
#   --task <name>         MT10 task name (default: reach-v3)
#   --optimizer <name>    optimizer (default: adam)
#   --delays <list>       semicolon-separated obs,act pairs
#                         (default: "0,0;1,0;0,1;4,4;8,8")
#   --seeds <list>        comma-separated seeds (default: "0")
#   --delay-mode <mode>   fixed | multi-task | continual (default: fixed)
#   --delay-emb-types <l> comma-separated list of one_hot|none
#                         (default: "one_hot,none" — augmented + unseen)
#   --total-steps <N>     total env steps (default: 5000000)
#
# Examples:
#   ./sky/launch_td_mt1.sh my-team td-mt1
#   ./sky/launch_td_mt1.sh my-team td-mt1 --task push-v3 --optimizer cbp
#   ./sky/launch_td_mt1.sh my-team td-mt1 --delays "0,0;3,3" --seeds "0,1"
#
# Monitor:
#   sky queue                                — list jobs
#   sky logs td-mt1-reach-obs0-act0          — view logs (example)
#   sky status                               — cluster status

set -euo pipefail

WANDB_ENTITY="${1:?Usage: $0 <wandb_entity> <wandb_project> [options]}"
WANDB_PROJECT="${2:?Usage: $0 <wandb_entity> <wandb_project> [options]}"
shift 2

# Defaults
TASK="reach-v3"
OPTIMIZER="adam"
DELAYS_STR="0,0;1,0;0,1;4,4;8,8"
SEEDS_STR="0"
DELAY_MODE="fixed"
EMB_TYPES_STR="one_hot,none"
TOTAL_STEPS="5000000"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --task) TASK="$2"; shift 2 ;;
        --optimizer) OPTIMIZER="$2"; shift 2 ;;
        --delays) DELAYS_STR="$2"; shift 2 ;;
        --seeds) SEEDS_STR="$2"; shift 2 ;;
        --delay-mode) DELAY_MODE="$2"; shift 2 ;;
        --delay-emb-types) EMB_TYPES_STR="$2"; shift 2 ;;
        --total-steps) TOTAL_STEPS="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [ -z "${WANDB_API_KEY:-}" ]; then
    echo "Error: WANDB_API_KEY environment variable must be set."
    echo "Get it from https://wandb.ai/authorize"
    exit 1
fi

# Convert seeds to space-separated for the YAML run loop
SEEDS="${SEEDS_STR//,/ }"

# Short task name for cluster naming (strip -v3 suffix)
TASK_SHORT="${TASK%-v3}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
YAML_PATH="$SCRIPT_DIR/td_mt1.yaml"

# Parse delay configs (semicolon-separated obs,act pairs) and emb types
IFS=';' read -ra DELAY_CONFIGS <<< "$DELAYS_STR"
IFS=',' read -ra EMB_TYPES <<< "$EMB_TYPES_STR"

NUM_JOBS=$(( ${#DELAY_CONFIGS[@]} * ${#EMB_TYPES[@]} ))

echo "Launching $NUM_JOBS TD-MT1 jobs on Vast.ai"
echo "  Entity:     $WANDB_ENTITY"
echo "  Project:    $WANDB_PROJECT"
echo "  Task:       $TASK"
echo "  Optimizer:  $OPTIMIZER"
echo "  Delay mode: $DELAY_MODE"
echo "  Delays:     $DELAYS_STR"
echo "  Emb types:  $EMB_TYPES_STR"
echo "  Seeds:      $SEEDS"
echo "  Steps:      $TOTAL_STEPS"
echo ""

for cfg in "${DELAY_CONFIGS[@]}"; do
    OBS_DELAY="${cfg%,*}"
    ACT_DELAY="${cfg#*,}"
    for EMB in "${EMB_TYPES[@]}"; do
        # Short tag for cluster name: aug = one_hot, unseen = none
        case "$EMB" in
            one_hot) EMB_TAG="aug" ;;
            none)    EMB_TAG="unseen" ;;
            *)       EMB_TAG="$EMB" ;;
        esac
        CLUSTER="td-mt1-${TASK_SHORT}-${EMB_TAG}-obs${OBS_DELAY}-act${ACT_DELAY}"

        echo "Launching: obs=$OBS_DELAY act=$ACT_DELAY emb=$EMB  -> $CLUSTER"
        sky launch "$YAML_PATH" \
            -c "$CLUSTER" \
            -d -y --down -i 5 \
            --env TASK="$TASK" \
            --env OPTIMIZER="$OPTIMIZER" \
            --env MAX_OBS_DELAY="$OBS_DELAY" \
            --env MAX_ACT_DELAY="$ACT_DELAY" \
            --env DELAY_MODE="$DELAY_MODE" \
            --env DELAY_EMB_TYPE="$EMB" \
            --env SEEDS="$SEEDS" \
            --env TOTAL_STEPS="$TOTAL_STEPS" \
            --env WANDB_API_KEY="$WANDB_API_KEY" \
            --env WANDB_ENTITY="$WANDB_ENTITY" \
            --env WANDB_PROJECT="$WANDB_PROJECT"
        echo "  -> Submitted $CLUSTER"
        echo ""
    done
done

echo "All jobs submitted."
echo "  sky queue    — list jobs"
echo "  sky logs td-mt1-${TASK_SHORT}-<aug|unseen>-obs<O>-act<A> — view logs"
echo "  sky status   — cluster status"
