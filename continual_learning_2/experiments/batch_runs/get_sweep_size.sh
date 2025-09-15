#!/bin/bash

# Helper script to get the number of configurations for a given algorithm
# Usage: ./get_sweep_size.sh <algo>

if [ $# -eq 0 ]; then
    echo "Usage: $0 <algorithm>"
    echo "Available algorithms: adam, regrama, redo, cbp, ccbp, shrink_and_perturb"
    exit 1
fi

algo="$1"

# Get the total number of configurations
total=$(python sweep_slippery_ant.py --algo "$algo" --list-configs | tail -1 | grep -o '[0-9]\+')

if [ -z "$total" ]; then
    echo "Error: Could not determine configuration count for algorithm '$algo'"
    exit 1
fi

# Array indices are 0-based, so max index is total-1
max_index=$((total - 1))

echo "Algorithm: $algo"
echo "Total configurations: $total"
echo "SLURM array range: 0-$max_index"
echo ""
echo "To submit the sweep, run:"
echo "sbatch --array=0-$max_index slurm_hyperparameter_sweep.sh $algo"