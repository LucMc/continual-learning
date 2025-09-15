#!/bin/bash

# Example usage of the enhanced analyze.py script with combined network metrics

# Set your W&B config (should be set in environment)
# WANDB_CFG="--wandb-entity your_entity"

# Slippery Ant Results - Individual metrics (original behavior)
echo "Plotting individual metrics..."
python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric charts/mean_episodic_return --output-dir plots/main --ext png

# Slippery Ant Results - Combined network metrics (NEW FUNCTIONALITY)
echo "Plotting combined network metrics..."
python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final \
    --metrics dormant,linearized --combine-networks \
    --output-dir plots/main --ext png

# Alternative: Plot all network metrics separately but in one command
echo "Plotting all network metrics separately..."
python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final \
    --metrics nn/value_dormant_neurons/total_ratio,nn/actor_dormant_neurons/total_ratio,nn/value_linearised_neurons/total_ratio,nn/actor_linearised_neurons/total_ratio \
    --output-dir plots/main --ext png

# Combined with performance metric
echo "Plotting performance with combined network metrics..."
python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final \
    --metrics charts/mean_episodic_return,dormant,linearized --combine-networks \
    --output-dir plots/main --ext png

echo "All plots generated!"
echo ""
echo "Usage examples:"
echo "1. Single metric (original):     --metric charts/mean_episodic_return"
echo "2. Multiple metrics (spaces):    --metrics metric1 metric2 metric3"
echo "3. Multiple metrics (comma):     --metrics 'metric1,metric2,metric3'"
echo "4. Combined networks:            --metrics dormant,linearized --combine-networks"
echo "5. Mix of both:                  --metrics charts/mean_episodic_return,dormant --combine-networks"