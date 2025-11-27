 # python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric nn/value_srank/layer_0_act --output-dir plots/main --ext png
 # python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric nn/actor_srank/main/layer_0_act --output-dir plots/main --ext png
 # python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric nn/value_srank/layer_1_act --output-dir plots/main --ext png
 # python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric nn/actor_srank/main/layer_1_act --output-dir plots/main --ext png
 # python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric nn/value_srank/layer_2_act --output-dir plots/main --ext png
 # python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric nn/actor_srank/main/layer_2_act --output-dir plots/main --ext png
 # python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric nn/value_srank/layer_3_act --output-dir plots/main --ext png
 # python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric nn/actor_srank/main/layer_3_act --output-dir plots/main --ext png
 # python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metrics value_srank_hidden --output-dir plots/main --ext png
 # python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metrics actor_srank_hidden --output-dir plots/main --ext png
     # python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metrics srank_hidden --combine-networks --output-dir plots/main/combined --ext png --normalize-networks --plot-title "S-rank" --no-show-iqr
 # python analyze.py $WANDB_CFG --group slippery_humanoid_full6 --wandb_project crl_humanoid --metrics srank_hidden --output-dir plots/humanoid --ext png --plot-title "S-rank" --no-show-iqr
 # python analyze.py $WANDB_CFG --group cbp_test3 --wandb_project crl_experiments --metric charts/mean_episodic_return --output-dir plots/main --ext png --bar-chart --plot-title "cbp test4"
 #
     # python analyze.py $WANDB_CFG --group cbp_test3 --wandb_project crl_experiments --metrics srank_hidden --combine-networks --output-dir plots/main/combined --ext png --normalize-networks --plot-title "S-rank cbp" --no-show-iqr
   # python ablation_plot.py --wandb-entity lucmc --wandb_project crl_experiments --group cbp_test3 --metric charts/mean_episodic_return --grouping-mode parameter --split-by replacement_rate --ext png --output-dir plots/ablations/cbp --plot-title "test3"
   # python analyze.py --wandb_entity cavlab --group slippery_ant_full2 --wandb_project test_ant --metric charts/mean_episodic_return --output-dir plots/main --ext png --bar-chart --plot-title "Mean Episodic Return test"
     # python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metrics srank_hidden --combine-networks --output-dir plots/main/combined/unnorm --ext png --plot-title "S-rank" --no-show-iqr
     
 # Perm MNIST
 # python analyze.py $WANDB_CFG --group perm_mnist --wandb_project crl_final --metric metrics/eval_accuracy_ci --output-dir plots/perm_mnist --ext png --x-axis-max 2 --y-min 0.90 --no-show-iqr --plot-title "Mean Evaluation Accuracy"
 # python analyze.py $WANDB_CFG --group perm_mnist_final --wandb_project crl_final --metric nn/value_dormant_neurons/total_ratio --output-dir plots/perm_mnist --ext png
 # python analyze.py $WANDB_CFG --group perm_mnist_final --wandb_project crl_final --metric nn/actor_dormant_neurons/total_ratio --output-dir plots/perm_mnist --ext png
 # python analyze.py $WANDB_CFG --group perm_mnist_final --wandb_project crl_final --metric nn/value_linearised_neurons/total_ratio --output-dir plots/perm_mnist --ext png
 # python analyze.py $WANDB_CFG --group perm_mnist_final --wandb_project crl_final --metric nn/actor_linearised_neurons/total_ratio --output-dir plots/perm_mnist --ext png

     # python ablation_plot.py --wandb-entity lucmc --wandb_project crl_experiments --group slippery_ant_ccbp_sweep --metric nn/value_dormant_neurons/total_ratio --grouping-mode parameter --split-by replacement_rate --ext png --output-dir plots/ablations/ccbp
     # python ablation_plot.py --wandb-entity lucmc --wandb_project crl_experiments --group slippery_ant_ccbp_sweep --metric nn/actor_dormant_neurons/total_ratio --grouping-mode parameter --split-by replacement_rate --ext png --output-dir plots/ablations/ccbp
     # python ablation_plot.py --wandb-entity lucmc --wandb_project crl_experiments --group slippery_ant_ccbp_sweep --metric nn/value_linearised_neurons/total_ratio --grouping-mode parameter --split-by replacement_rate --ext png --output-dir plots/ablations/ccbp
     # python ablation_plot.py --wandb-entity lucmc --wandb_project crl_experiments --group slippery_ant_ccbp_sweep --metric nn/actor_linearised_neurons/total_ratio --grouping-mode parameter --split-by replacement_rate --ext png --output-dir plots/ablations/ccbp
     #
     # python ablation_plot.py --wandb-entity lucmc --wandb_project crl_experiments --group slippery_ant_ccbp_sweep --metric nn/vf_gradient_norm --grouping-mode parameter --split-by replacement_rate --ext png --output-dir plots/ablations/ccbp --plot-title "Value Network Gradient Norm"
     # python ablation_plot.py --wandb-entity lucmc --wandb_project crl_experiments --group slippery_ant_ccbp_sweep --metric nn/policy_gradient_norm --grouping-mode parameter --split-by replacement_rate --ext png --output-dir plots/ablations/ccbp --plot-title "Policy Network Gradient Norm"
     #
     # python analyze.py --wandb-entity lucmc --wandb_project crl_experiments --group slippery_ant_ccbp_sweep --metrics dormant,linearized --combine-networks --ext png --output-dir plots/ablations/ccbp
     #
     # python ablation_plot.py --wandb-entity lucmc --wandb_project crl_experiments --group slippery_ant_ccbp_sweep --metric charts/mean_episodic_return --grouping-mode parameter --split-by replacement_rate --ext png --output-dir plots/ablations/ccbp
     # python ablation_plot.py --wandb-entity lucmc --wandb_project crl_experiments --group slippery_ant_ccbp_sweep --metric charts/mean_episodic_return --grouping-mode parameter --split-by decay_rate --ext png --output-dir plots/ablations/ccbp
     # python ablation_plot.py --wandb-entity lucmc --wandb_project crl_experiments --group slippery_ant_ccbp_sweep --metric charts/mean_episodic_return --grouping-mode parameter --split-by sharpness --ext png --output-dir plots/ablations/ccbp
     # python ablation_plot.py --wandb-entity lucmc --wandb_project crl_experiments --group slippery_ant_ccbp_sweep --metric charts/mean_episodic_return --grouping-mode parameter --split-by threshold --ext png --output-dir plots/ablations/ccbp
     # python ablation_plot.py --wandb-entity lucmc --wandb_project crl_experiments --group slippery_ant_ccbp_sweep --metric charts/mean_episodic_return --grouping-mode parameter --split-by update_frequency --ext png --output-dir plots/ablations/ccbp
     
     # CBP
     # python ablation_plot.py --wandb-entity lucmc --wandb_project crl_experiments --group slippery_ant_cbp_sweep --metric charts/mean_episodic_return --grouping-mode parameter --split-by replacement_rate --ext png --output-dir plots/ablations/cbp
     # python ablation_plot.py --wandb-entity lucmc --wandb_project crl_experiments --group slippery_ant_cbp_sweep --metric charts/mean_episodic_return --grouping-mode parameter --split-by maturity_threshold --ext png --output-dir plots/ablations/cbp
     # python ablation_plot.py --wandb-entity lucmc --wandb_project crl_experiments --group slippery_ant_cbp_sweep --metric charts/mean_episodic_return --grouping-mode parameter --split-by decay_rate --ext png --output-dir plots/ablations/cbp
     #
     # # ReDo
     # python ablation_plot.py --wandb-entity lucmc --wandb_project crl_experiments --group slippery_ant_redo_sweep --metric charts/mean_episodic_return --grouping-mode parameter --split-by score_threshold --ext png --output-dir plots/ablations/redo
     # python ablation_plot.py --wandb-entity lucmc --wandb_project crl_experiments --group slippery_ant_redo_sweep --metric charts/mean_episodic_return --grouping-mode parameter --split-by update_frequency --ext png --output-dir plots/ablations/redo
     #
     # # ReGrama
     # python ablation_plot.py --wandb-entity lucmc --wandb_project crl_experiments --group slippery_ant_regrama_sweep --metric charts/mean_episodic_return --grouping-mode parameter --split-by score_threshold --ext png --output-dir plots/ablations/regrama
     # python ablation_plot.py --wandb-entity lucmc --wandb_project crl_experiments --group slippery_ant_regrama_sweep --metric charts/mean_episodic_return --grouping-mode parameter --split-by update_frequency --ext png --output-dir plots/ablations/regrama
     #
     # shrink_pertub
     # python ablation_plot.py --wandb-entity lucmc --wandb_project crl_experiments --group slippery_ant_shrink_and_perturb_sweep --metric charts/mean_episodic_return --grouping-mode parameter --split-by shrink --ext png --output-dir plots/ablations/sp
     # python ablation_plot.py --wandb-entity lucmc --wandb_project crl_experiments --group slippery_ant_shrink_and_perturb_sweep --metric charts/mean_episodic_return --grouping-mode parameter --split-by pertub --ext png --output-dir plots/ablations/sp
     # python ablation_plot.py --wandb-entity lucmc --wandb_project crl_experiments --group slippery_ant_shrink_and_perturb_sweep --metric charts/mean_episodic_return --grouping-mode parameter --split-by every_n --ext png --output-dir plots/ablations/sp
     #
     # Just for which runs are best
     # python ablation_plot.py --wandb-entity lucmc --wandb_project crl_experiments --group slippery_ant_cbp_sweep --metric charts/mean_episodic_return --grouping-mode config --ext png --output-dir plots/ablations/all --short-labels --ranking-criteria average --top-k 5
     # python ablation_plot.py --wandb-entity lucmc --wandb_project crl_experiments --group slippery_ant_redo_sweep --metric charts/mean_episodic_return --grouping-mode config --ext png --output-dir plots/ablations/all --short-labels --ranking-criteria average --top-k 5
     # python ablation_plot.py --wandb-entity lucmc --wandb_project crl_experiments --group slippery_ant_regrama_sweep --metric charts/mean_episodic_return --grouping-mode config --ext png --output-dir plots/ablations/all --short-labels --ranking-criteria average --top-k 5
     
 # python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric nn/value_srank/layer_0_act --output-dir plots/main --ext png
 # python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric nn/actor_srank/main/layer_0_act --output-dir plots/main --ext png
 # python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric nn/value_srank/layer_1_act --output-dir plots/main --ext png
 # python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric nn/actor_srank/main/layer_1_act --output-dir plots/main --ext png
 # python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric nn/value_srank/layer_2_act --output-dir plots/main --ext png
 # python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric nn/actor_srank/main/layer_2_act --output-dir plots/main --ext png
 # python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric nn/value_srank/layer_3_act --output-dir plots/main --ext png
 # python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric nn/actor_srank/main/layer_3_act --output-dir plots/main --ext png
 # python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric nn/value_srank/output_act --output-dir plots/main --ext png
 # python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric nn/actor_srank/main/output_act --output-dir plots/main --ext png
 # python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metrics value_srank_hidden --output-dir plots/main --ext png
 # python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metrics actor_srank_hidden --output-dir plots/main --ext png
 # python analyze.py $WANDB_CFG  --group slippery_ant_bigbatch --wandb_project crl_experiments --metrics srank_hidden --combine-networks --output-dir plots/bigbatch/combined --ext png  --plot-title "S-rank" --no-show-iqr
    
################################################################################################################################################################################################################
 # Slippery Ant Results
 python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric charts/mean_episodic_return --output-dir plots/main --ext png --bar-chart --plot-title "Mean Episodic Return"
 
 # python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric nn/value_dormant_neurons/total_ratio --output-dir plots/main --ext png
 # python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric nn/actor_dormant_neurons/total_ratio --output-dir plots/main --ext png
 # python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric nn/value_linearised_neurons/total_ratio --output-dir plots/main --ext png
 # python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric nn/actor_linearised_neurons/total_ratio --output-dir plots/main --ext png
 # python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric nn/policy_parameter_norm --output-dir plots/main --ext png
 # python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric nn/vf_parameter_norm --output-dir plots/main --ext png
 # python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric nn/vf_gradient_norm --output-dir plots/main --ext png
 # python backup2_analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric nn/policy_gradient_norm --output-dir plots/main --ext png --no-show-iqr --plot-title "Policy Network Gradient Norm"
 #
 # # Large batch size adam experiment
 # python backup2_analyze.py $WANDB_CFG --group slippery_ant_bigbatch --wandb_project crl_experiments --metric charts/mean_episodic_return --output-dir plots/bigbatch --plot-title "Mean Episodic Return" 
 # python backup2_analyze.py $WANDB_CFG  --group slippery_ant_bigbatch --wandb_project crl_experiments --metric dormant --combine-networks --output-dir plots/bigbatch/combined --ext png  --plot-title "Dormant Neurons" 
 # python backup2_analyze.py $WANDB_CFG  --group slippery_ant_bigbatch --wandb_project crl_experiments --metric linearized --combine-networks --output-dir plots/bigbatch/combined --ext png  --plot-title "Linearized Neurons" 
 # python backup2_analyze.py $WANDB_CFG  --group slippery_ant_bigbatch --wandb_project crl_experiments --metrics gradient_norm --combine-networks --output-dir plots/bigbatch/combined --ext png --plot-title "Gradient Norm" 
 # python analyze.py $WANDB_CFG --group slippery_ant_bigbatch --wandb_project crl_experiments --metric nn/value_dormant_neurons/total_ratio --output-dir plots/bigbatch
 # python analyze.py $WANDB_CFG --group slippery_ant_bigbatch --wandb_project crl_experiments --metric nn/actor_dormant_neurons/total_ratio --output-dir plots/bigbatch
 # python analyze.py $WANDB_CFG --group slippery_ant_bigbatch --wandb_project crl_experiments --metric nn/value_linearised_neurons/total_ratio --output-dir plots/bigbatch
 # python analyze.py $WANDB_CFG --group slippery_ant_bigbatch --wandb_project crl_experiments --metric nn/actor_linearised_neurons/total_ratio --output-dir plots/bigbatch
 
 # # Dormant/collapse bar chart
 # python plot_policy_collapse.py --wandb-entity lucmc --group ccbp_server --wandb_project crl_final --metric charts/mean_episodic_return --collapse_threshold 8000 --output-dir plots/collapse_bar --overall --min-consecutive-below 10


 # Ablations
     # CCBP
     # python ablation_plot.py --wandb-entity lucmc --wandb_project crl_experiments --group slippery_ant_transforms --metric charts/mean_episodic_return --grouping-mode parameter --split-by transform_type --ext png --output-dir plots/ablations/ccbp --plot-title "Transformation Function Comparison" --no-show-iqr --no-show-metric-in-legend
     # python ablation_plot.py --wandb-entity lucmc --wandb_project crl_experiments --group ccbp_replacement_rate --metric charts/mean_episodic_return --grouping-mode parameter --split-by replacement_rate --ext png --output-dir plots/ablations/ccbp --plot-title "Stability-Plasticity Tuning" --no-show-iqr --y-tick-count 6
     # python ablation_plot.py --wandb-entity lucmc --wandb_project crl_experiments --group slippery_ant_ccbp_sharpness_sweep --metric charts/mean_episodic_return --grouping-mode parameter --split-by sharpness --ext png --output-dir plots/ablations/ccbp --plot-title "Sharpness Ablation" --no-show-iqr --sort-by-value

     # Combined
     # python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric dormant --combine-networks --output-dir plots/main/combined/unnorm --ext png --plot-title "Dormant Neurons" --no-show-iqr
     # python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric linearized --combine-networks --output-dir plots/main/combined/unnorm --ext png --plot-title "Linearized Neurons" --no-show-iqr
     # python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metrics gradient_norm --combine-networks --output-dir plots/main/combined/unnorm --ext png --plot-title "Gradient Norm" --no-show-iqr

     # python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric dormant --combine-networks --output-dir plots/main/combined --ext png --normalize-networks --plot-title "Dormant Neurons" --no-show-iqr
     # python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric linearized --combine-networks --output-dir plots/main/combined --ext png --normalize-networks --plot-title "Linearized Neurons" --no-show-iqr
     # python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metrics gradient_norm --combine-networks --output-dir plots/main/combined --ext png --normalize-networks --plot-title "Gradient Norm" --no-show-iqr
     #
     # python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metrics gradient_norm --combine-networks --output-dir plots/main/combined --ext png --normalize-networks --plot-title "Gradient Norm"
     # python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metrics parameter_norm --combine-networks --output-dir plots/main/combined --ext png --plot-title "Parameter Norm" --no-show-iqr
     # python analyze.py $WANDB_CFG --group slippery_humanoid_full6 --wandb_project crl_humanoid --metrics parameter_norm --combine-networks --output-dir plots/main/combined/humanoid --ext png --plot-title "Parameter Norm" --no-show-iqr
     #
     ## python analyze.py --wandb-entity lucmc --wandb_project crl_experiments --group slippery_ant_transforms --metrics srank_hidden --combine-networks --output-dir plots/main/combined --ext png --normalize-networks --plot-title "S-rank transforms" --no-show-iqr     
 
 # Slippery Humanoid
 # python analyze.py $WANDB_CFG --group slippery_humanoid_full6 --wandb_project crl_humanoid --metric charts/mean_episodic_return --output-dir plots/humanoid --ext png --bar-chart --plot-title "Mean Episodic Return" --no-show-iqr
 # python analyze.py $WANDB_CFG --group slippery_humanoid_test --wandb_project crl_experiments --metric charts/mean_episodic_return --output-dir plots/humanoid --ext png --bar-chart
 # python analyze.py $WANDB_CFG --group slippery_humanoid_full6 --wandb_project crl_experiments --metric charts/mean_episodic_return --output-dir plots/humanoid --ext png --bar-chart --plot-title "Episodic Return"
 
 # python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric nn/value_dormant_neurons/total_ratio --output-dir plots/main --ext png
 # python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric nn/actor_dormant_neurons/total_ratio --output-dir plots/main --ext png
 # python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric nn/value_linearised_neurons/total_ratio --output-dir plots/main --ext png
 # python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric nn/actor_linearised_neurons/total_ratio --output-dir plots/main --ext png
 # python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric nn/policy_parameter_norm --output-dir plots/main --ext png
 # python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric nn/vf_parameter_norm --output-dir plots/main --ext png
 # python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric nn/vf_gradient_norm --output-dir plots/main --ext png --no-show-iqr --plot-title "Value Network Gradient Norm"
 # python python backup2_analyze.py $WANDB_CFG --group slippery_ant_bigbatch --wandb_project crl_experiments --metric charts/mean_episodic_return --output-dir plots/bigbatch  $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric nn/policy_gradient_norm --output-dir plots/main --ext png --no-show-iqr --plot-title "Policy Network Gradient Norm"
 #
 # python analyze.py $WANDB_CFG --group slippery_humanoid_full6 --wandb_project crl_humanoid --metrics dormant --combine-networks --output-dir plots/humanoid --ext png --plot-title "Dormant Neurons" --no-show-iqr
 # python analyze.py $WANDB_CFG --group slippery_humanoid_full6 --wandb_project crl_humanoid --metrics linearized --combine-networks --output-dir plots/humanoid --ext png --plot-title "Linearized Neurons" --no-show-iqr
 # python analyze.py $WANDB_CFG --group slippery_humanoid_full6 --wandb_project crl_humanoid --metrics gradient_norm --combine-networks --output-dir plots/humanoid --ext png --plot-title "Gradient Norm" --no-show-iqr
 # Threshold ablation
  # python compare_thresholds.py --wandb-entity lucmc --wandb-project crl_humanoid --group luc_humanoid_tau --metric charts/mean_episodic_return --threshold-1 1.0 --threshold-2 0.95 --pattern-1 "ccbp_br_adam_sb_lr_smaller_net_*" --pattern-2 "ccbp_bigger_rollout_new_hparams_*" --bar-chart --base-text-size 20.0 --y-min -2000 --plot-title "SlipperyHumanoid threshold comparison"
  # python compare_thresholds.py --wandb-entity lucmc --wandb-project crl_final --group ant_tau_1 --metric charts/mean_episodic_return --threshold-1 1.0 --threshold-2 0.95 --pattern-1 "ccbp_smaller_*" --pattern-2 "ccbp_s*-copy" --bar-chart --base-text-size 20.0 --y-min -2000 --plot-title "SlipperyAnt threshold comparison"
