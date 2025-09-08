#!/bin/bash
gpus=(0 1 2 3)
algo="redo"

# Get total number of configs for this algo
total_configs=$(python sweep_slippery_ant.py --algo $algo --list-configs | tail -n 1 | grep -o '[0-9]*$')

echo "Running $total_configs configs for $algo across ${#gpus[@]} GPUs"

# Run configs in parallel across GPUs
config_id=0
while [ $config_id -lt $total_configs ]; do
    for gpu in "${gpus[@]}"; do
        if [ $config_id -lt $total_configs ]; then
            echo "Starting config $config_id on GPU $gpu"
            CUDA_VISIBLE_DEVICES=$gpu python sweep_slippery_ant.py --algo $algo --config-id $config_id $WANDB_CFG &
            ((config_id++))
        fi
    done
    wait  # Wait for current batch to complete before starting next batch
done
