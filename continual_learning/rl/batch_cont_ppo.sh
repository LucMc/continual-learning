#!/bin/bash

# Exit on any error
set -e

# --- 0. Usage Function ---
# A function to print help/usage information
usage() {
    echo "Usage: $0 -s <number_of_runs> [-m <method1> <method2> ...]"
    echo ""
    echo "Options:"
    echo "  -s <number>   (Required) The number of seeds to run, from 1 to 20."
    echo "  -m <methods>  (Optional) A space-separated list of methods to run."
    echo "                If not provided, all default methods will be run."
    echo ""
    echo "Default Methods: cbp ccbp ccbp2 none"
    echo "Example: $0 -s 5"
    echo "Example: $0 -s 3 -m cbp none"
    exit 1
}

# --- 1. Define Defaults and Initialize Variables ---
num_runs=""
# Define the full array of available methods
all_methods=("cbp" "ccbp" "ccbp2" "none")
# This array will hold the methods we actually run (either all or a subset)
methods_to_run=()

# --- 2. Parse Command-Line Arguments ---
# Loop until all arguments have been processed
while [[ $# -gt 0 ]]; do
    case "$1" in
        -s)
            # Check if a value was provided for -s
            if [[ -z "$2" ]] || [[ "$2" =~ ^- ]]; then
                echo "Error: Option -s requires an argument." >&2
                usage
            fi
            num_runs="$2"
            shift 2 # Move past the flag and its value
            ;;
        -m)
            shift # Move past the -m flag
            # Capture all subsequent arguments until we find another flag (starting with -) or run out of args
            while [[ $# -gt 0 ]] && ! [[ "$1" =~ ^- ]]; do
                methods_to_run+=("$1")
                shift
            done
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Error: Unknown option '$1'" >&2
            usage
            ;;
    esac
done

# --- 3. Input Handling and Validation ---
# If user did not provide a list of methods, use all of them by default
if [ ${#methods_to_run[@]} -eq 0 ]; then
    echo "No methods specified with -m, running all default methods."
    methods_to_run=("${all_methods[@]}")
fi

# Check if the number of runs was provided (mandatory)
if [ -z "$num_runs" ]; then
    echo "Error: The -s <number_of_runs> option is mandatory." >&2
    usage
fi

# Validate that the number of runs is an integer between 1 and 20
if ! [[ "$num_runs" =~ ^[0-9]+$ ]] || [ "$num_runs" -lt 1 ] || [ "$num_runs" -gt 20 ]; then
    echo "Error: Number of runs must be an integer between 1 and 20." >&2
    exit 1
fi

echo "Preparing to run experiments for the first $num_runs seed(s)..."
echo "Methods to run: ${methods_to_run[*]}"
echo ""


# --- 4. Define Experiment Parameters ---
# Define the full array of 20 seeds for consistency
seeds=($(seq 1 20))

# Name of your Python script
python_script="cont_ppo.py"


# --- 5. Select the Batch of Seeds ---
# Slice the seeds array to get the first N seeds based on user input
selected_seeds=("${seeds[@]:0:$num_runs}")


# --- 6. Main Execution Loop ---
# Loop through each selected seed and reset method
for seed in "${selected_seeds[@]}"
do
    for reset_method in "${methods_to_run[@]}"
    do
        echo "Running with seed: $seed, dormant-reset-method: $reset_method"
        python "$python_script" --seed "$seed" --dormant-reset-method "$reset_method" --log
        
        echo "Seed $seed with reset method $reset_method completed successfully"
        echo "-----------------------------------"
    done
done

echo "Batch of $num_runs seed(s) completed!"
