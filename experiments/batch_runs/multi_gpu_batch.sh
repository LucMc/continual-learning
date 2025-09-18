#!/bin/bash
# Usage: ./run_experiments.sh [script_name] [max_jobs_per_gpu] [vram_threshold]
# Example: ./run_experiments.sh perm_mist.py 2 50
# Defaults: script_name=slippery_ant.py, max_jobs_per_gpu=1, vram_threshold=50
# If you haven't set the WANDB_CFG env variable do it below
# WANDB_CFG="--wandb_entity=... --wandb_project=..."

script_name="${1:-slippery_ant.py}"
max_jobs_per_gpu="${2:-1}"
vram_threshold="${3:-50}"  # VRAM usage threshold in percentage

num_gpus=$(nvidia-smi --list-gpus | wc -l)
algos=("redo" "regrama" "adam" "cbp" "ccbp" "shrink_and_perturb")
seeds=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15)

echo "Running $script_name with up to $max_jobs_per_gpu jobs per GPU across $num_gpus GPUs"
echo "VRAM threshold: ${vram_threshold}% (only using GPUs with less than this usage)"

# Function to check VRAM usage for a specific GPU
get_gpu_vram_usage() {
    local gpu_id=$1
    local memory_info=$(nvidia-smi -i $gpu_id --query-gpu=memory.used,memory.total --format=csv,noheader,nounits)
    local used_mem=$(echo $memory_info | cut -d',' -f1 | tr -d ' ')
    local total_mem=$(echo $memory_info | cut -d',' -f2 | tr -d ' ')
    
    if [ "$total_mem" -gt 0 ]; then
        echo $(( (used_mem * 100) / total_mem ))
    else
        echo 100  # Return 100% if we can't get memory info
    fi
}

# Function to find an available GPU
find_available_gpu() {
    local available_gpus=()
    
    for ((gpu_id=0; gpu_id<num_gpus; gpu_id++)); do
        local vram_usage=$(get_gpu_vram_usage $gpu_id)
        local jobs_on_gpu=${gpu_job_count[$gpu_id]:-0}
        
        if [ "$vram_usage" -lt "$vram_threshold" ] && [ "$jobs_on_gpu" -lt "$max_jobs_per_gpu" ]; then
            available_gpus+=($gpu_id)
        fi
    done
    
    if [ ${#available_gpus[@]} -gt 0 ]; then
        # Return GPU with lowest VRAM usage among available ones
        local best_gpu=${available_gpus[0]}
        local best_usage=$(get_gpu_vram_usage $best_gpu)
        
        for gpu_id in "${available_gpus[@]:1}"; do
            local usage=$(get_gpu_vram_usage $gpu_id)
            if [ "$usage" -lt "$best_usage" ]; then
                best_gpu=$gpu_id
                best_usage=$usage
            fi
        done
        echo $best_gpu
    else
        echo -1  # No available GPU
    fi
}

# Function to wait for a GPU to become available
wait_for_available_gpu() {
    local gpu_id=-1
    local wait_count=0
    
    while [ "$gpu_id" -eq -1 ]; do
        gpu_id=$(find_available_gpu)
        
        if [ "$gpu_id" -eq -1 ]; then
            if [ $((wait_count % 6)) -eq 0 ]; then  # Print status every 30 seconds
                echo "Waiting for GPU availability (checked ${wait_count} times)..."
                echo "Current GPU status:"
                for ((i=0; i<num_gpus; i++)); do
                    local vram_usage=$(get_gpu_vram_usage $i)
                    local jobs=${gpu_job_count[$i]:-0}
                    echo "  GPU $i: VRAM usage=${vram_usage}%, running jobs=${jobs}/${max_jobs_per_gpu}"
                done
            fi
            sleep 5
            ((wait_count++))
            
            # Check if any background jobs have finished
            jobs_finished=0
            for pid in "${!job_pids[@]}"; do
                if ! kill -0 "$pid" 2>/dev/null; then
                    # Job finished, decrease counter for its GPU
                    gpu_of_job=${job_pids[$pid]}
                    ((gpu_job_count[$gpu_of_job]--))
                    unset job_pids[$pid]
                    jobs_finished=1
                fi
            done
        fi
    done
    
    echo $gpu_id
}

# Initialize job tracking arrays
declare -A gpu_job_count  # Track number of jobs per GPU
declare -A job_pids        # Track PIDs and their assigned GPUs

for ((i=0; i<num_gpus; i++)); do
    gpu_job_count[$i]=0
done

# Main execution loop
total_jobs=$((${#seeds[@]} * ${#algos[@]}))
job_num=0

for seed in "${seeds[@]}"; do
    for alg in "${algos[@]}"; do
        ((job_num++))
        
        # Find or wait for an available GPU
        gpu_id=$(wait_for_available_gpu)
        vram_usage=$(get_gpu_vram_usage $gpu_id)
        
        echo "[Job $job_num/$total_jobs] Launching $alg seed $seed on GPU $gpu_id (VRAM usage: ${vram_usage}%)"
        
        # Launch the job
        CUDA_VISIBLE_DEVICES=$gpu_id python "../$script_name" $WANDB_CFG --include "$alg" --seed "$seed" &
        pid=$!
        
        # Track the job
        job_pids[$pid]=$gpu_id
        ((gpu_job_count[$gpu_id]++))
        
        # Small delay to avoid race conditions
        sleep 0.5
    done
done

# Wait for all remaining jobs to complete
echo "All jobs dispatched. Waiting for completion..."
wait

echo "All experiments completed."

# Final status report
echo "Final GPU statistics:"
for ((i=0; i<num_gpus; i++)); do
    vram_usage=$(get_gpu_vram_usage $i)
    echo "  GPU $i: Final VRAM usage=${vram_usage}%"
done
