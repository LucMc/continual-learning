#!/bin/bash

# Usage: ./run_experiments.sh [script_name] [max_jobs_per_gpu]
# Example: ./run_experiments.sh perm_mist.py 2
# Defaults: script_name=slippery_ant.py, max_jobs_per_gpu=1 (one job per GPU)

script_name="${1:-slippery_ant.py}"
num_gpus=$(nvidia-smi --list-gpus | wc -l)
max_jobs_per_gpu="${2:-1}"

algos=("redo" "regrama" "adam" "cbp")
seeds=(0 1 2 3 4)

echo "Running $script_name with up to $max_jobs_per_gpu jobs per GPU across $num_gpus GPUs"

total_concurrent_jobs=$((num_gpus * max_jobs_per_gpu))

job_count=0

for seed in "${seeds[@]}"; do
  for alg in "${algos[@]}"; do
    gpu_id=$(( job_count % num_gpus ))
    echo "Launching $alg seed $seed on GPU $gpu_id"

    CUDA_VISIBLE_DEVICES=$gpu_id python "../$script_name" $WANDB_CFG --include "$alg" --seed "$seed" &

    ((job_count++))

    # Limit the total concurrent jobs to num_gpus * max_jobs_per_gpu
    if (( job_count % total_concurrent_jobs == 0 )); then
      wait
    fi
  done
done

wait

echo "All experiments completed."
