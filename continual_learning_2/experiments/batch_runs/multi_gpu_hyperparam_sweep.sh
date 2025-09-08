#!/usr/bin/env bash
# Queue runner for sweep_slippery_ant.py
# One job per GPU from a user-provided list. Starts a new job when another finishes.
# Requires: bash >= 4, nvidia-smi (optional), python3 (or python).

set -euo pipefail

# ---------- defaults ----------
PYTHON_BIN="${PYTHON_BIN:-}"                 # Optional: export PYTHON_BIN=python to override
SCRIPT_PATH="sweep_slippery_ant.py"          # Default Python entrypoint
GPUS_CSV=""                                  # e.g. "0,1,3"
ALGOS_DEFAULT=("adam" "regrama" "redo" "cbp" "ccbp" "shrink_and_perturb")
ALGOS=()
SEEDS=("42")                                 # default seed list
CONFIG_IDS_STR=""                            # e.g. "0-7,9,12" (applies to all algos); if empty we enumerate all configs for each algo
WANDB_ENTITY=""                              # optional
WANDB_PROJECT=""                             # optional
LOG_DIR=""                                   # if empty -> logs/<timestamp>
DRY_RUN=0

# ---------- helpers ----------
usage() {
  cat <<'EOF'
Usage:
  ./gpu_queue_runner.sh --gpus 0,1,2 [--script sweep_slippery_ant.py]
                        [--algos "redo regrama"] [--seeds "0 1 2"]
                        [--config-ids "0-9,12,15"] [--wandb-entity ENTITY]
                        [--wandb-project PROJECT] [--log-dir DIR] [--dry-run]

What it does:
  * Builds a job queue of (algo, config_id, seed) using sweep_slippery_ant.py.
  * If --config-ids isn't provided, it enumerates ALL config IDs per algo by calling:
      python sweep_slippery_ant.py <algo> --list
  * Concurrency = number of GPU IDs you pass to --gpus (one job per listed GPU).
  * Starts a new job on a freed GPU as soon as one finishes.

Examples:
  # Use GPUs 0 and 2; all configs for 'redo' and 'regrama'; seeds 0..2
  ./gpu_queue_runner.sh --gpus 0,2 --algos "redo regrama" --seeds "0 1 2"

  # Use GPUs 0,1,2,3; run specific config IDs (0..7 and 12) across all algos, seed 42
  ./gpu_queue_runner.sh --gpus 0,1,2,3 --config-ids "0-7,12"

  # With Weights & Biases (positional args in your Python script)
  ./gpu_queue_runner.sh --gpus 0,1 --algos "adam" --seeds "0 1 2" \
      --wandb-entity myteam --wandb-project ant_sweep

Flags:
  --gpus           Comma- or space-separated list of GPU IDs (required).
  --script         Path to sweep_slippery_ant.py (default: sweep_slippery_ant.py).
  --algos          Space-separated algos. Default: all defined in your script.
  --seeds          Space-separated seeds. Default: "42".
  --config-ids     Comma/space list of IDs and/or ranges (e.g. "0-9,12"). If omitted,
                   the script asks sweep_slippery_ant.py for "Total configs" per algo.
  --wandb-entity   Optional entity (passed as 4th positional arg).
  --wandb-project  Optional project (passed as 5th positional arg).
  --log-dir        Where to write logs. Default: logs/<timestamp>.
  --dry-run        Build and print the queue, but don't run anything.

Env overrides:
  PYTHON_BIN       Set to 'python' or 'python3'. If unset, the script picks one.

Notes:
  * Each launched process sees only its assigned GPU via CUDA_VISIBLE_DEVICES,
    so inside the process the GPU is index 0 (common JAX/CUDA behavior).
  * Consider: export XLA_PYTHON_CLIENT_PREALLOCATE=false (JAX) if you want
    to avoid large upfront GPU memory reservations.
EOF
}

# Expand comma/space-separated tokens with optional ranges, e.g. "0-3,6 8-9"
expand_range_list() {
  local input="${1:-}"
  input="${input//,/ }"
  local out=()
  for tok in $input; do
    if [[ "$tok" =~ ^([0-9]+)-([0-9]+)$ ]]; then
      local a=${BASH_REMATCH[1]}
      local b=${BASH_REMATCH[2]}
      if (( a <= b )); then
        for ((i=a;i<=b;i++)); do out+=("$i"); done
      else
        for ((i=a;i>=b;i--)); do out+=("$i"); done
      fi
    elif [[ "$tok" =~ ^[0-9]+$ ]]; then
      out+=("$tok")
    elif [[ -n "$tok" ]]; then
      echo "Invalid numeric token in list: '$tok'" >&2
      exit 2
    fi
  done
  echo "${out[@]}"
}

count_configs_for_algo() {
  local algo="$1"
  local py="${PYTHON_BIN:-}"
  if [[ -z "$py" ]]; then
    if command -v python3 >/dev/null 2>&1; then py="python3"; else py="python"; fi
  fi
  local out
  if ! out="$("$py" "$SCRIPT_PATH" "$algo" --list 2>/dev/null)"; then
    echo "0"
    return
  fi
  local last
  last="$(echo "$out" | tail -n 1)"
  local n
  n="$(echo "$last" | awk -F': ' '/Total configs/ {print $2}')"
  if [[ -z "${n:-}" ]]; then
    # Fallback: count lines starting with "N:" (robust if format changes slightly)
    n="$(echo "$out" | awk -F: '/^[0-9]+:/{c++} END{print c+0}')"
  fi
  echo "${n:-0}"
}

timestamp() { date +"%Y-%m-%d %H:%M:%S"; }

# ---------- parse args ----------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus) GPUS_CSV="${2:-}"; shift 2 ;;
    --script) SCRIPT_PATH="${2:-}"; shift 2 ;;
    --algos) IFS=' ' read -r -a ALGOS <<< "${2:-}"; shift 2 ;;
    --seeds) IFS=' ' read -r -a SEEDS <<< "${2:-}"; shift 2 ;;
    --config-ids) CONFIG_IDS_STR="${2:-}"; shift 2 ;;
    --wandb-entity) WANDB_ENTITY="${2:-}"; shift 2 ;;
    --wandb-project) WANDB_PROJECT="${2:-}"; shift 2 ;;
    --log-dir) LOG_DIR="${2:-}"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "$GPUS_CSV" ]]; then
  echo "Error: --gpus is required." >&2
  usage
  exit 2
fi

if [[ ! -f "$SCRIPT_PATH" ]]; then
  echo "Error: Python script not found: $SCRIPT_PATH" >&2
  exit 2
fi

# Determine python bin
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if command -v python3 >/dev/null 2>&1; then PYTHON_BIN="python3"; else PYTHON_BIN="python"; fi
fi

# Parse GPUs
read -r -a GPU_LIST <<< "$(expand_range_list "$GPUS_CSV")"
if [[ ${#GPU_LIST[@]} -eq 0 ]]; then
  echo "No valid GPU IDs parsed from: '$GPUS_CSV'" >&2
  exit 2
fi

# Default algos
if [[ ${#ALGOS[@]} -eq 0 ]]; then
  ALGOS=("${ALGOS_DEFAULT[@]}")
fi

# Expand config IDs (if provided)
CONFIG_IDS_GLOBAL=()
if [[ -n "$CONFIG_IDS_STR" ]]; then
  read -r -a CONFIG_IDS_GLOBAL <<< "$(expand_range_list "$CONFIG_IDS_STR")"
  if [[ ${#CONFIG_IDS_GLOBAL[@]} -eq 0 ]]; then
    echo "No valid config IDs parsed from: '$CONFIG_IDS_STR'" >&2
    exit 2
  fi
fi

# Prepare logs
if [[ -z "$LOG_DIR" ]]; then
  LOG_DIR="logs/$(date +%Y%m%d_%H%M%S)"
fi
mkdir -p "$LOG_DIR"

# ---------- build job queue ----------
# Each job: "algo|config_id|seed"
JOB_QUEUE=()

for algo in "${ALGOS[@]}"; do
  if [[ ${#CONFIG_IDS_GLOBAL[@]} -gt 0 ]]; then
    mapfile -t cfgs < <(printf "%s\n" "${CONFIG_IDS_GLOBAL[@]}")
  else
    # Ask the python file how many configs exist for this algo
    total_cfgs="$(count_configs_for_algo "$algo")"
    if ! [[ "$total_cfgs" =~ ^[0-9]+$ ]] || [[ "$total_cfgs" -le 0 ]]; then
      echo "Warning: Could not discover configs for algo '$algo' (got '$total_cfgs'). Skipping." >&2
      continue
    fi
    cfgs=()
    for ((c=0;c<total_cfgs;c++)); do cfgs+=("$c"); done
  fi

  for cfg in "${cfgs[@]}"; do
    for seed in "${SEEDS[@]}"; do
      JOB_QUEUE+=("${algo}|${cfg}|${seed}")
    done
  done
done

TOTAL_JOBS=${#JOB_QUEUE[@]}
if [[ $TOTAL_JOBS -eq 0 ]]; then
  echo "No jobs to run. Check your --algos / --config-ids / --seeds." >&2
  exit 1
fi

echo "[$(timestamp)] Built queue with $TOTAL_JOBS jobs."
echo "  GPUs: ${GPU_LIST[*]}"
echo "  Algos: ${ALGOS[*]}"
echo "  Seeds: ${SEEDS[*]}"
if [[ ${#CONFIG_IDS_GLOBAL[@]} -gt 0 ]]; then
  echo "  Config IDs (global): ${CONFIG_IDS_GLOBAL[*]}"
else
  echo "  Config IDs: enumerated per algo via --list"
fi
echo "  Logs: $LOG_DIR"

if [[ $DRY_RUN -eq 1 ]]; then
  echo
  echo "Dry run – showing first 20 jobs:"
  for ((i=0;i<20 && i<TOTAL_JOBS;i++)); do
    IFS='|' read -r a c s <<< "${JOB_QUEUE[$i]}"
    echo "  [$((i+1))/$TOTAL_JOBS] algo=$a cfg=$c seed=$s"
  done
  exit 0
fi

# ---------- scheduler ----------
# One running PID per GPU slot
declare -A RUNNING_PIDS  # key: GPU_ID -> PID

# Clean up background children on exit / Ctrl-C
cleanup() {
  echo
  echo "[$(timestamp)] Caught exit; terminating child processes..."
  for g in "${GPU_LIST[@]}"; do
    if [[ -n "${RUNNING_PIDS[$g]:-}" ]]; then
      if kill -0 "${RUNNING_PIDS[$g]}" 2>/dev/null; then
        kill "${RUNNING_PIDS[$g]}" 2>/dev/null || true
      fi
    fi
  done
}
trap cleanup EXIT INT TERM

# Helper: check and free finished slots
reap_finished() {
  local g pid
  for g in "${GPU_LIST[@]}"; do
    pid="${RUNNING_PIDS[$g]:-}"
    if [[ -n "$pid" ]]; then
      if ! kill -0 "$pid" 2>/dev/null; then
        unset RUNNING_PIDS["$g"]
        echo "[$(timestamp)] GPU $g finished (pid $pid freed)."
      fi
    fi
  done
}

next_free_gpu() {
  local g
  for g in "${GPU_LIST[@]}"; do
    if [[ -z "${RUNNING_PIDS[$g]:-}" ]]; then
      echo "$g"
      return 0
    fi
  done
  echo ""  # none
  return 1
}

launch_job_on_gpu() {
  local gpu="$1"
  local algo="$2"
  local cfg="$3"
  local seed="$4"

  # Build positional args for the python script
  # <algo> <config_id> [seed] [wandb_entity] [wandb_project]
  local -a argv
  argv+=("$algo" "$cfg" "$seed")
  if [[ -n "$WANDB_ENTITY" ]]; then
    argv+=("$WANDB_ENTITY")
  fi
  if [[ -n "$WANDB_PROJECT" ]]; then
    # If project supplied but entity not, pass empty entity to occupy the slot
    if [[ ${#argv[@]} -lt 4 ]]; then argv+=(""); fi
    argv+=("$WANDB_PROJECT")
  fi

  local tag="${algo}_cfg${cfg}_s${seed}_gpu${gpu}"
  local log_file="${LOG_DIR}/${tag}.log"

  echo "[$(timestamp)] Launching: ${tag}"
  echo "  -> $PYTHON_BIN $SCRIPT_PATH ${argv[*]}" | tee -a "$log_file"

  # Recommended for JAX multi-run ergonomics (optional; uncomment if you prefer)
  # export XLA_PYTHON_CLIENT_PREALLOCATE=false

  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" "$SCRIPT_PATH" "${argv[@]}" >>"$log_file" 2>&1 &
  local pid=$!
  RUNNING_PIDS["$gpu"]="$pid"
  echo "  pid=$pid  log=$log_file"
}

# Main dispatch loop
idx=0
while (( idx < TOTAL_JOBS )); do
  reap_finished
  free_gpu="$(next_free_gpu || true)"
  if [[ -n "$free_gpu" ]]; then
    IFS='|' read -r A C S <<< "${JOB_QUEUE[$idx]}"
    echo "[Dispatch $((idx+1))/$TOTAL_JOBS] algo=$A cfg=$C seed=$S -> GPU $free_gpu"
    launch_job_on_gpu "$free_gpu" "$A" "$C" "$S"
    ((idx++))
    # Gentle pacing so logs are readable & PIDs register
    sleep 0.3
  else
    # No slot free yet; wait a bit and re-check
    sleep 2
  fi
done

echo "[$(timestamp)] All jobs dispatched. Waiting for completion..."
# Wait until all slots free
while :; do
  reap_finished
  all_free=1
  for g in "${GPU_LIST[@]}"; do
    if [[ -n "${RUNNING_PIDS[$g]:-}" ]]; then all_free=0; break; fi
  done
  (( all_free == 1 )) && break
  sleep 2
done

echo "[$(timestamp)] All experiments completed. Logs: $LOG_DIR"

