#!/bin/bash

# Exit on any error
set -e

# --- 1. Input Handling and Validation ---
# Check if the number of runs was provided
if [ -z "$1" ]; then
    echo "Error: Please provide the number of runs."
    echo "Usage: $0 <number_of_runs>"
    exit 1
fi

# Store the number of runs from the first argument
num_runs=$1

# Validate that the input is a number between 1 and 20
if ! [[ "$num_runs" =~ ^[0-9]+$ ]] || [ "$num_runs" -lt 1 ] || [ "$num_runs" -gt 20 ]; then
    echo "Error: Number of runs must be an integer between 1 and 20."
    exit 1
fi

echo "Preparing to run experiments for the first $num_runs seed(s)..."
echo ""

# --- 2. Define Experiment Parameters ---
# Define the full array of 20 seeds for consistency
# Using seq for convenience, but you can list them manually if you prefer non-sequential seeds.
seeds=($(seq 1 20))

# Define the array of dormant reset methods
dormant_reset_methods=("cbp" "ccbp" "ccbp2" "none")

# Name of your Python script
python_script="cont_ppo.py"


# --- 3. Select the Batch of Seeds ---
# Slice the seeds array to get the first N seeds based on user input
# The syntax is ${array[@]:start_index:count}
selected_seeds=("${seeds[@]:0:$num_runs}")


# --- 4. Main Execution Loop ---
# Loop through each selected seed and reset method
for seed in "${selected_seeds[@]}"
do
    for reset_method in "${dormant_reset_methods[@]}"
    do
        echo "Running with seed: $seed, dormant-reset-method: $reset_method"
        python "$python_script" --seed "$seed" --dormant-reset-method "$reset_method" --log
        
        echo "Seed $seed with reset method $reset_method completed successfully"
        echo "-----------------------------------"
    done
done

echo "Batch of $num_runs seed(s) completed!"
