#!/bin/bash

# Exit on any error
set -e

# Define the array of seeds
seeds=(1 2 3 4 5)

# Define the array of dormant reset methods
dormant_reset_methods=("cbp" "ccbp" "ccbp2" "none")

# Name of your Python script
python_script="ppo_brax_cont.py"

# Loop through each seed and reset method
for seed in "${seeds[@]}"
do
    for reset_method in "${dormant_reset_methods[@]}"
    do
        echo "Running with seed: $seed, dormant-reset-method: $reset_method"
        python "$python_script" --seed "$seed" --dormant-reset-method "$reset_method" --log
        
        echo "Seed $seed with reset method $reset_method completed successfully"
        echo "-----------------------------------"
    done
done

echo "All combinations completed!"
