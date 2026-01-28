# #!/bin/bash
#
# ################################################################################
# # Legacy/Additional S-rank metrics (recovered from original commented section)
# ################################################################################
# # With IQR
# python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric nn/value_srank/layer_0_act --output-dir plots/main --ext png --plot-title "cpr_server_nn_value_srank_layer_0_act"
# python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric nn/actor_srank/main/layer_0_act --output-dir plots/main --ext png --plot-title "cpr_server_nn_actor_srank_main_layer_0_act"
# python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric nn/value_srank/layer_1_act --output-dir plots/main --ext png --plot-title "cpr_server_nn_value_srank_layer_1_act"
# python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric nn/actor_srank/main/layer_1_act --output-dir plots/main --ext png --plot-title "cpr_server_nn_actor_srank_main_layer_1_act"
# python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric nn/value_srank/layer_2_act --output-dir plots/main --ext png --plot-title "cpr_server_nn_value_srank_layer_2_act"
# python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric nn/actor_srank/main/layer_2_act --output-dir plots/main --ext png --plot-title "cpr_server_nn_actor_srank_main_layer_2_act"
# python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric nn/value_srank/layer_3_act --output-dir plots/main --ext png --plot-title "cpr_server_nn_value_srank_layer_3_act"
# python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric nn/actor_srank/main/layer_3_act --output-dir plots/main --ext png --plot-title "cpr_server_nn_actor_srank_main_layer_3_act"
# python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metrics value_srank_hidden --output-dir plots/main --ext png --plot-title "cpr_server_value_srank_hidden"
# python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metrics actor_srank_hidden --output-dir plots/main --ext png --plot-title "cpr_server_actor_srank_hidden"
# python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metrics srank_hidden --combine-networks --output-dir plots/main/combined --ext png --plot-title "S-rank"
# python analyze.py $WANDB_CFG --group slippery_humanoid_full6 --wandb_project crl_humanoid --metrics srank_hidden --output-dir plots/humanoid --ext png --plot-title "S-rank Humanoid"
# # python analyze.py $WANDB_CFG --group cbp_test3 --wandb_project crl_experiments --metric charts/mean_episodic_return --output-dir plots/main --ext png --bar-chart --y-min 0 --plot-title "cbp test4"
# # python analyze.py $WANDB_CFG --group cbp_test3 --wandb_project crl_experiments --metrics srank_hidden --combine-networks --output-dir plots/main/combined --ext png --plot-title "S-rank cbp"
# # python ablation_plot.py --wandb-entity lucmc --wandb_project crl_experiments --group cbp_test3 --metric charts/mean_episodic_return --grouping-mode parameter --split-by replacement_rate --ext png --output-dir plots/ablations/cbp --plot-title "test3"
# # python analyze.py --wandb_entity cavlab --group slippery_ant_full2 --wandb_project test_ant --metric charts/mean_episodic_return --output-dir plots/main --ext png --bar-chart --y-min 0 --plot-title "Mean Episodic Return test"
# python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metrics srank_hidden --combine-networks --output-dir plots/main/combined/unnorm --ext png --plot-title "S-rank unnorm"
#
# # Without IQR
# python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric nn/value_srank/layer_0_act --output-dir plots/main/no_iqr --ext png --plot-title "cpr_server_nn_value_srank_layer_0_act" --no-show-iqr
# python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric nn/actor_srank/main/layer_0_act --output-dir plots/main/no_iqr --ext png --plot-title "cpr_server_nn_actor_srank_main_layer_0_act" --no-show-iqr
# python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric nn/value_srank/layer_1_act --output-dir plots/main/no_iqr --ext png --plot-title "cpr_server_nn_value_srank_layer_1_act" --no-show-iqr
# python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric nn/actor_srank/main/layer_1_act --output-dir plots/main/no_iqr --ext png --plot-title "cpr_server_nn_actor_srank_main_layer_1_act" --no-show-iqr
# python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric nn/value_srank/layer_2_act --output-dir plots/main/no_iqr --ext png --plot-title "cpr_server_nn_value_srank_layer_2_act" --no-show-iqr
# python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric nn/actor_srank/main/layer_2_act --output-dir plots/main/no_iqr --ext png --plot-title "cpr_server_nn_actor_srank_main_layer_2_act" --no-show-iqr
# python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric nn/value_srank/layer_3_act --output-dir plots/main/no_iqr --ext png --plot-title "cpr_server_nn_value_srank_layer_3_act" --no-show-iqr
# python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric nn/actor_srank/main/layer_3_act --output-dir plots/main/no_iqr --ext png --plot-title "cpr_server_nn_actor_srank_main_layer_3_act" --no-show-iqr
# python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metrics value_srank_hidden --output-dir plots/main/no_iqr --ext png --plot-title "cpr_server_value_srank_hidden" --no-show-iqr
# python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metrics actor_srank_hidden --output-dir plots/main/no_iqr --ext png --plot-title "cpr_server_actor_srank_hidden" --no-show-iqr
# python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metrics srank_hidden --combine-networks --output-dir plots/main/combined/no_iqr --ext png --plot-title "S-rank" --no-show-iqr
# python analyze.py $WANDB_CFG --group slippery_humanoid_full6 --wandb_project crl_humanoid --metrics srank_hidden --output-dir plots/humanoid/no_iqr --ext png --plot-title "S-rank Humanoid" --no-show-iqr
# # python analyze.py $WANDB_CFG --group cbp_test3 --wandb_project crl_experiments --metric charts/mean_episodic_return --output-dir plots/main/no_iqr --ext png --bar-chart --y-min 0 --plot-title "cbp test4" --no-show-iqr
# # python analyze.py $WANDB_CFG --group cbp_test3 --wandb_project crl_experiments --metrics srank_hidden --combine-networks --output-dir plots/main/combined/no_iqr --ext png --plot-title "S-rank cbp" --no-show-iqr
# # python ablation_plot.py --wandb-entity lucmc --wandb_project crl_experiments --group cbp_test3 --metric charts/mean_episodic_return --grouping-mode parameter --split-by replacement_rate --ext png --output-dir plots/ablations/cbp/no_iqr --plot-title "test3" --no-show-iqr
# # python analyze.py --wandb_entity cavlab --group slippery_ant_full2 --wandb_project test_ant --metric charts/mean_episodic_return --output-dir plots/main/no_iqr --ext png --bar-chart --y-min 0 --plot-title "Mean Episodic Return test" --no-show-iqr
# python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metrics srank_hidden --combine-networks --output-dir plots/main/combined/unnorm/no_iqr --ext png --plot-title "S-rank unnorm" --no-show-iqr
#
# ################################################################################
# # Perm MNIST
# ################################################################################
# # With IQR
# python analyze.py $WANDB_CFG --group perm_mnist --wandb_project crl_final --metric metrics/eval_accuracy_ci --output-dir plots/perm_mnist --ext png --x-axis-max 2 --y-min 0.90 --plot-title "Mean Evaluation Accuracy"
# python analyze.py $WANDB_CFG --group perm_mnist_final --wandb_project crl_final --metric nn/value_dormant_neurons/total_ratio --output-dir plots/perm_mnist --ext png --plot-title "perm_mnist_value_dormant_neurons"
# python analyze.py $WANDB_CFG --group perm_mnist_final --wandb_project crl_final --metric nn/actor_dormant_neurons/total_ratio --output-dir plots/perm_mnist --ext png --plot-title "perm_mnist_actor_dormant_neurons"
# python analyze.py $WANDB_CFG --group perm_mnist_final --wandb_project crl_final --metric nn/value_linearised_neurons/total_ratio --output-dir plots/perm_mnist --ext png --plot-title "perm_mnist_value_linearised_neurons"
# python analyze.py $WANDB_CFG --group perm_mnist_final --wandb_project crl_final --metric nn/actor_linearised_neurons/total_ratio --output-dir plots/perm_mnist --ext png --plot-title "perm_mnist_actor_linearised_neurons"
#
# # Without IQR
# python analyze.py $WANDB_CFG --group perm_mnist --wandb_project crl_final --metric metrics/eval_accuracy_ci --output-dir plots/perm_mnist/no_iqr --ext png --x-axis-max 2 --y-min 0.90 --plot-title "Mean Evaluation Accuracy" --no-show-iqr
# python analyze.py $WANDB_CFG --group perm_mnist_final --wandb_project crl_final --metric nn/value_dormant_neurons/total_ratio --output-dir plots/perm_mnist/no_iqr --ext png --plot-title "perm_mnist_value_dormant_neurons" --no-show-iqr
# python analyze.py $WANDB_CFG --group perm_mnist_final --wandb_project crl_final --metric nn/actor_dormant_neurons/total_ratio --output-dir plots/perm_mnist/no_iqr --ext png --plot-title "perm_mnist_actor_dormant_neurons" --no-show-iqr
# python analyze.py $WANDB_CFG --group perm_mnist_final --wandb_project crl_final --metric nn/value_linearised_neurons/total_ratio --output-dir plots/perm_mnist/no_iqr --ext png --plot-title "perm_mnist_value_linearised_neurons" --no-show-iqr
# python analyze.py $WANDB_CFG --group perm_mnist_final --wandb_project crl_final --metric nn/actor_linearised_neurons/total_ratio --output-dir plots/perm_mnist/no_iqr --ext png --plot-title "perm_mnist_actor_linearised_neurons" --no-show-iqr
#
# ################################################################################
# # CPR Sweep Ablations (slippery_ant_ccbp_sweep)
# ################################################################################
# # With IQR
# python ablation_plot.py --wandb-entity lucmc --wandb_project crl_experiments --group slippery_ant_ccbp_sweep --metric nn/value_dormant_neurons/total_ratio --grouping-mode parameter --split-by replacement_rate --ext png --output-dir plots/ablations/cpr --plot-title "CPR Sweep Value Dormant by Replacement Rate"
# python ablation_plot.py --wandb-entity lucmc --wandb_project crl_experiments --group slippery_ant_ccbp_sweep --metric nn/actor_dormant_neurons/total_ratio --grouping-mode parameter --split-by replacement_rate --ext png --output-dir plots/ablations/cpr --plot-title "CPR Sweep Actor Dormant by Replacement Rate"
# python ablation_plot.py --wandb-entity lucmc --wandb_project crl_experiments --group slippery_ant_ccbp_sweep --metric nn/value_linearised_neurons/total_ratio --grouping-mode parameter --split-by replacement_rate --ext png --output-dir plots/ablations/cpr --plot-title "CPR Sweep Value Linearised by Replacement Rate"
# python ablation_plot.py --wandb-entity lucmc --wandb_project crl_experiments --group slippery_ant_ccbp_sweep --metric nn/actor_linearised_neurons/total_ratio --grouping-mode parameter --split-by replacement_rate --ext png --output-dir plots/ablations/cpr --plot-title "CPR Sweep Actor Linearised by Replacement Rate"
# python ablation_plot.py --wandb-entity lucmc --wandb_project crl_experiments --group slippery_ant_ccbp_sweep --metric nn/vf_gradient_norm --grouping-mode parameter --split-by replacement_rate --ext png --output-dir plots/ablations/cpr --plot-title "CPR Sweep Value Network Gradient Norm"
# python ablation_plot.py --wandb-entity lucmc --wandb_project crl_experiments --group slippery_ant_ccbp_sweep --metric nn/policy_gradient_norm --grouping-mode parameter --split-by replacement_rate --ext png --output-dir plots/ablations/cpr --plot-title "CPR Sweep Policy Network Gradient Norm"
# python analyze.py --wandb-entity lucmc --wandb_project crl_experiments --group slippery_ant_ccbp_sweep --metrics dormant --combine-networks --ext png --output-dir plots/ablations/cpr --plot-title "CPR Sweep Dormant Combined"
# python analyze.py --wandb-entity lucmc --wandb_project crl_experiments --group slippery_ant_ccbp_sweep --metrics linearized --combine-networks --ext png --output-dir plots/ablations/cpr --plot-title "CPR Sweep Linearized Combined"
# python ablation_plot.py --wandb-entity lucmc --wandb_project crl_experiments --group slippery_ant_ccbp_sweep --metric charts/mean_episodic_return --grouping-mode parameter --split-by replacement_rate --ext png --output-dir plots/ablations/cpr --plot-title "CPR Sweep Return by Replacement Rate"
# python ablation_plot.py --wandb-entity lucmc --wandb_project crl_experiments --group slippery_ant_ccbp_sweep --metric charts/mean_episodic_return --grouping-mode parameter --split-by decay_rate --ext png --output-dir plots/ablations/cpr --plot-title "CPR Sweep Return by Decay Rate"
# python ablation_plot.py --wandb-entity lucmc --wandb_project crl_experiments --group slippery_ant_ccbp_sweep --metric charts/mean_episodic_return --grouping-mode parameter --split-by sharpness --ext png --output-dir plots/ablations/cpr --plot-title "CPR Sweep Return by Sharpness"
# python ablation_plot.py --wandb-entity lucmc --wandb_project crl_experiments --group slippery_ant_ccbp_sweep --metric charts/mean_episodic_return --grouping-mode parameter --split-by threshold --ext png --output-dir plots/ablations/cpr --plot-title "CPR Sweep Return by Threshold"
# python ablation_plot.py --wandb-entity lucmc --wandb_project crl_experiments --group slippery_ant_ccbp_sweep --metric charts/mean_episodic_return --grouping-mode parameter --split-by update_frequency --ext png --output-dir plots/ablations/cpr --plot-title "CPR Sweep Return by Update Frequency"
#
# # Without IQR
# python ablation_plot.py --wandb-entity lucmc --wandb_project crl_experiments --group slippery_ant_ccbp_sweep --metric nn/value_dormant_neurons/total_ratio --grouping-mode parameter --split-by replacement_rate --ext png --output-dir plots/ablations/cpr/no_iqr --plot-title "CPR Sweep Value Dormant by Replacement Rate" --no-show-iqr
# python ablation_plot.py --wandb-entity lucmc --wandb_project crl_experiments --group slippery_ant_ccbp_sweep --metric nn/actor_dormant_neurons/total_ratio --grouping-mode parameter --split-by replacement_rate --ext png --output-dir plots/ablations/cpr/no_iqr --plot-title "CPR Sweep Actor Dormant by Replacement Rate" --no-show-iqr
# python ablation_plot.py --wandb-entity lucmc --wandb_project crl_experiments --group slippery_ant_ccbp_sweep --metric nn/value_linearised_neurons/total_ratio --grouping-mode parameter --split-by replacement_rate --ext png --output-dir plots/ablations/cpr/no_iqr --plot-title "CPR Sweep Value Linearised by Replacement Rate" --no-show-iqr
# python ablation_plot.py --wandb-entity lucmc --wandb_project crl_experiments --group slippery_ant_ccbp_sweep --metric nn/actor_linearised_neurons/total_ratio --grouping-mode parameter --split-by replacement_rate --ext png --output-dir plots/ablations/cpr/no_iqr --plot-title "CPR Sweep Actor Linearised by Replacement Rate" --no-show-iqr
# python ablation_plot.py --wandb-entity lucmc --wandb_project crl_experiments --group slippery_ant_ccbp_sweep --metric nn/vf_gradient_norm --grouping-mode parameter --split-by replacement_rate --ext png --output-dir plots/ablations/cpr/no_iqr --plot-title "CPR Sweep Value Network Gradient Norm" --no-show-iqr
# python ablation_plot.py --wandb-entity lucmc --wandb_project crl_experiments --group slippery_ant_ccbp_sweep --metric nn/policy_gradient_norm --grouping-mode parameter --split-by replacement_rate --ext png --output-dir plots/ablations/cpr/no_iqr --plot-title "CPR Sweep Policy Network Gradient Norm" --no-show-iqr
# python analyze.py --wandb-entity lucmc --wandb_project crl_experiments --group slippery_ant_ccbp_sweep --metrics dormant --combine-networks --ext png --output-dir plots/ablations/cpr/no_iqr --plot-title "CPR Sweep Dormant Combined" --no-show-iqr
# python analyze.py --wandb-entity lucmc --wandb_project crl_experiments --group slippery_ant_ccbp_sweep --metrics linearized --combine-networks --ext png --output-dir plots/ablations/cpr/no_iqr --plot-title "CPR Sweep Linearized Combined" --no-show-iqr
# python ablation_plot.py --wandb-entity lucmc --wandb_project crl_experiments --group slippery_ant_ccbp_sweep --metric charts/mean_episodic_return --grouping-mode parameter --split-by replacement_rate --ext png --output-dir plots/ablations/cpr/no_iqr --plot-title "CPR Sweep Return by Replacement Rate" --no-show-iqr
# python ablation_plot.py --wandb-entity lucmc --wandb_project crl_experiments --group slippery_ant_ccbp_sweep --metric charts/mean_episodic_return --grouping-mode parameter --split-by decay_rate --ext png --output-dir plots/ablations/cpr/no_iqr --plot-title "CPR Sweep Return by Decay Rate" --no-show-iqr
# python ablation_plot.py --wandb-entity lucmc --wandb_project crl_experiments --group slippery_ant_ccbp_sweep --metric charts/mean_episodic_return --grouping-mode parameter --split-by sharpness --ext png --output-dir plots/ablations/cpr/no_iqr --plot-title "CPR Sweep Return by Sharpness" --no-show-iqr
# python ablation_plot.py --wandb-entity lucmc --wandb_project crl_experiments --group slippery_ant_ccbp_sweep --metric charts/mean_episodic_return --grouping-mode parameter --split-by threshold --ext png --output-dir plots/ablations/cpr/no_iqr --plot-title "CPR Sweep Return by Threshold" --no-show-iqr
# python ablation_plot.py --wandb-entity lucmc --wandb_project crl_experiments --group slippery_ant_ccbp_sweep --metric charts/mean_episodic_return --grouping-mode parameter --split-by update_frequency --ext png --output-dir plots/ablations/cpr/no_iqr --plot-title "CPR Sweep Return by Update Frequency" --no-show-iqr
#
# ################################################################################
# # CBP Ablations (slippery_ant_cbp_sweep)
# ################################################################################
# # With IQR
# python ablation_plot.py --wandb-entity lucmc --wandb_project crl_experiments --group slippery_ant_cbp_sweep --metric charts/mean_episodic_return --grouping-mode parameter --split-by replacement_rate --ext png --output-dir plots/ablations/cbp --plot-title "CBP Replacement Rate"
# python ablation_plot.py --wandb-entity lucmc --wandb_project crl_experiments --group slippery_ant_cbp_sweep --metric charts/mean_episodic_return --grouping-mode parameter --split-by maturity_threshold --ext png --output-dir plots/ablations/cbp --plot-title "CBP Maturity Threshold"
# python ablation_plot.py --wandb-entity lucmc --wandb_project crl_experiments --group slippery_ant_cbp_sweep --metric charts/mean_episodic_return --grouping-mode parameter --split-by decay_rate --ext png --output-dir plots/ablations/cbp --plot-title "CBP Decay Rate"
#
# # Without IQR
# python ablation_plot.py --wandb-entity lucmc --wandb_project crl_experiments --group slippery_ant_cbp_sweep --metric charts/mean_episodic_return --grouping-mode parameter --split-by replacement_rate --ext png --output-dir plots/ablations/cbp/no_iqr --plot-title "CBP Replacement Rate" --no-show-iqr
# python ablation_plot.py --wandb-entity lucmc --wandb_project crl_experiments --group slippery_ant_cbp_sweep --metric charts/mean_episodic_return --grouping-mode parameter --split-by maturity_threshold --ext png --output-dir plots/ablations/cbp/no_iqr --plot-title "CBP Maturity Threshold" --no-show-iqr
# python ablation_plot.py --wandb-entity lucmc --wandb_project crl_experiments --group slippery_ant_cbp_sweep --metric charts/mean_episodic_return --grouping-mode parameter --split-by decay_rate --ext png --output-dir plots/ablations/cbp/no_iqr --plot-title "CBP Decay Rate" --no-show-iqr
#
# ################################################################################
# # ReDo Ablations
# ################################################################################
# # With IQR
# python ablation_plot.py --wandb-entity lucmc --wandb_project crl_experiments --group slippery_ant_redo_sweep --metric charts/mean_episodic_return --grouping-mode parameter --split-by score_threshold --ext png --output-dir plots/ablations/redo --plot-title "ReDo Score Threshold"
# python ablation_plot.py --wandb-entity lucmc --wandb_project crl_experiments --group slippery_ant_redo_sweep --metric charts/mean_episodic_return --grouping-mode parameter --split-by update_frequency --ext png --output-dir plots/ablations/redo --plot-title "ReDo Update Frequency"
#
# # Without IQR
# python ablation_plot.py --wandb-entity lucmc --wandb_project crl_experiments --group slippery_ant_redo_sweep --metric charts/mean_episodic_return --grouping-mode parameter --split-by score_threshold --ext png --output-dir plots/ablations/redo/no_iqr --plot-title "ReDo Score Threshold" --no-show-iqr
# python ablation_plot.py --wandb-entity lucmc --wandb_project crl_experiments --group slippery_ant_redo_sweep --metric charts/mean_episodic_return --grouping-mode parameter --split-by update_frequency --ext png --output-dir plots/ablations/redo/no_iqr --plot-title "ReDo Update Frequency" --no-show-iqr
#
# ################################################################################
# # ReGraMa Ablations
# ################################################################################
# # With IQR
# python ablation_plot.py --wandb-entity lucmc --wandb_project crl_experiments --group slippery_ant_regrama_sweep --metric charts/mean_episodic_return --grouping-mode parameter --split-by score_threshold --ext png --output-dir plots/ablations/regrama --plot-title "ReGraMa Score Threshold"
# python ablation_plot.py --wandb-entity lucmc --wandb_project crl_experiments --group slippery_ant_regrama_sweep --metric charts/mean_episodic_return --grouping-mode parameter --split-by update_frequency --ext png --output-dir plots/ablations/regrama --plot-title "ReGraMa Update Frequency"
#
# # Without IQR
# python ablation_plot.py --wandb-entity lucmc --wandb_project crl_experiments --group slippery_ant_regrama_sweep --metric charts/mean_episodic_return --grouping-mode parameter --split-by score_threshold --ext png --output-dir plots/ablations/regrama/no_iqr --plot-title "ReGraMa Score Threshold" --no-show-iqr
# python ablation_plot.py --wandb-entity lucmc --wandb_project crl_experiments --group slippery_ant_regrama_sweep --metric charts/mean_episodic_return --grouping-mode parameter --split-by update_frequency --ext png --output-dir plots/ablations/regrama/no_iqr --plot-title "ReGraMa Update Frequency" --no-show-iqr
#
# ################################################################################
# # Shrink & Perturb Ablations
# ################################################################################
# # With IQR
# python ablation_plot.py --wandb-entity lucmc --wandb_project crl_experiments --group slippery_ant_shrink_and_perturb_sweep --metric charts/mean_episodic_return --grouping-mode parameter --split-by shrink --ext png --output-dir plots/ablations/sp --plot-title "SP Shrink"
# python ablation_plot.py --wandb-entity lucmc --wandb_project crl_experiments --group slippery_ant_shrink_and_perturb_sweep --metric charts/mean_episodic_return --grouping-mode parameter --split-by pertub --ext png --output-dir plots/ablations/sp --plot-title "SP Perturb"
# python ablation_plot.py --wandb-entity lucmc --wandb_project crl_experiments --group slippery_ant_shrink_and_perturb_sweep --metric charts/mean_episodic_return --grouping-mode parameter --split-by every_n --ext png --output-dir plots/ablations/sp --plot-title "SP Every N"
#
# # Without IQR
# python ablation_plot.py --wandb-entity lucmc --wandb_project crl_experiments --group slippery_ant_shrink_and_perturb_sweep --metric charts/mean_episodic_return --grouping-mode parameter --split-by shrink --ext png --output-dir plots/ablations/sp/no_iqr --plot-title "SP Shrink" --no-show-iqr
# python ablation_plot.py --wandb-entity lucmc --wandb_project crl_experiments --group slippery_ant_shrink_and_perturb_sweep --metric charts/mean_episodic_return --grouping-mode parameter --split-by pertub --ext png --output-dir plots/ablations/sp/no_iqr --plot-title "SP Perturb" --no-show-iqr
# python ablation_plot.py --wandb-entity lucmc --wandb_project crl_experiments --group slippery_ant_shrink_and_perturb_sweep --metric charts/mean_episodic_return --grouping-mode parameter --split-by every_n --ext png --output-dir plots/ablations/sp/no_iqr --plot-title "SP Every N" --no-show-iqr
#
# ################################################################################
# # Top-K Config Ablations (for determining best runs)
# ################################################################################
# # With IQR
# python ablation_plot.py --wandb-entity lucmc --wandb_project crl_experiments --group slippery_ant_cbp_sweep --metric charts/mean_episodic_return --grouping-mode config --ext png --output-dir plots/ablations/all --short-labels --ranking-criteria average --top-k 5 --plot-title "CBP Top5 Config"
# python ablation_plot.py --wandb-entity lucmc --wandb_project crl_experiments --group slippery_ant_redo_sweep --metric charts/mean_episodic_return --grouping-mode config --ext png --output-dir plots/ablations/all --short-labels --ranking-criteria average --top-k 5 --plot-title "ReDo Top5 Config"
# python ablation_plot.py --wandb-entity lucmc --wandb_project crl_experiments --group slippery_ant_regrama_sweep --metric charts/mean_episodic_return --grouping-mode config --ext png --output-dir plots/ablations/all --short-labels --ranking-criteria average --top-k 5 --plot-title "ReGraMa Top5 Config"
#
# # Without IQR
# python ablation_plot.py --wandb-entity lucmc --wandb_project crl_experiments --group slippery_ant_cbp_sweep --metric charts/mean_episodic_return --grouping-mode config --ext png --output-dir plots/ablations/all/no_iqr --short-labels --ranking-criteria average --top-k 5 --plot-title "CBP Top5 Config" --no-show-iqr
# python ablation_plot.py --wandb-entity lucmc --wandb_project crl_experiments --group slippery_ant_redo_sweep --metric charts/mean_episodic_return --grouping-mode config --ext png --output-dir plots/ablations/all/no_iqr --short-labels --ranking-criteria average --top-k 5 --plot-title "ReDo Top5 Config" --no-show-iqr
# python ablation_plot.py --wandb-entity lucmc --wandb_project crl_experiments --group slippery_ant_regrama_sweep --metric charts/mean_episodic_return --grouping-mode config --ext png --output-dir plots/ablations/all/no_iqr --short-labels --ranking-criteria average --top-k 5 --plot-title "ReGraMa Top5 Config" --no-show-iqr
#
# ################################################################################
# # S-rank metrics - Layer-wise
# ################################################################################
# # With IQR
# python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric nn/value_srank/layer_0_act --output-dir plots/main --ext png --plot-title "cpr_server_value_srank_layer_0"
# python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric nn/actor_srank/main/layer_0_act --output-dir plots/main --ext png --plot-title "cpr_server_actor_srank_layer_0"
# python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric nn/value_srank/layer_1_act --output-dir plots/main --ext png --plot-title "cpr_server_value_srank_layer_1"
# python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric nn/actor_srank/main/layer_1_act --output-dir plots/main --ext png --plot-title "cpr_server_actor_srank_layer_1"
# python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric nn/value_srank/layer_2_act --output-dir plots/main --ext png --plot-title "cpr_server_value_srank_layer_2"
# python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric nn/actor_srank/main/layer_2_act --output-dir plots/main --ext png --plot-title "cpr_server_actor_srank_layer_2"
# python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric nn/value_srank/layer_3_act --output-dir plots/main --ext png --plot-title "cpr_server_value_srank_layer_3"
# python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric nn/actor_srank/main/layer_3_act --output-dir plots/main --ext png --plot-title "cpr_server_actor_srank_layer_3"
# python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric nn/value_srank/output_act --output-dir plots/main --ext png --plot-title "cpr_server_value_srank_output"
# python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric nn/actor_srank/main/output_act --output-dir plots/main --ext png --plot-title "cpr_server_actor_srank_output"
# python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metrics value_srank_hidden --output-dir plots/main --ext png --plot-title "cpr_server_value_srank_hidden_main"
# python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metrics actor_srank_hidden --output-dir plots/main --ext png --plot-title "cpr_server_actor_srank_hidden_main"
# python analyze.py $WANDB_CFG --group slippery_ant_bigbatch --wandb_project crl_experiments --metrics srank_hidden --combine-networks --output-dir plots/bigbatch/combined --ext png --plot-title "S-rank Bigbatch"
#
# # Without IQR
# python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric nn/value_srank/layer_0_act --output-dir plots/main/no_iqr --ext png --plot-title "cpr_server_value_srank_layer_0" --no-show-iqr
# python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric nn/actor_srank/main/layer_0_act --output-dir plots/main/no_iqr --ext png --plot-title "cpr_server_actor_srank_layer_0" --no-show-iqr
# python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric nn/value_srank/layer_1_act --output-dir plots/main/no_iqr --ext png --plot-title "cpr_server_value_srank_layer_1" --no-show-iqr
# python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric nn/actor_srank/main/layer_1_act --output-dir plots/main/no_iqr --ext png --plot-title "cpr_server_actor_srank_layer_1" --no-show-iqr
# python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric nn/value_srank/layer_2_act --output-dir plots/main/no_iqr --ext png --plot-title "cpr_server_value_srank_layer_2" --no-show-iqr
# python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric nn/actor_srank/main/layer_2_act --output-dir plots/main/no_iqr --ext png --plot-title "cpr_server_actor_srank_layer_2" --no-show-iqr
# python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric nn/value_srank/layer_3_act --output-dir plots/main/no_iqr --ext png --plot-title "cpr_server_value_srank_layer_3" --no-show-iqr
# python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric nn/actor_srank/main/layer_3_act --output-dir plots/main/no_iqr --ext png --plot-title "cpr_server_actor_srank_layer_3" --no-show-iqr
# python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric nn/value_srank/output_act --output-dir plots/main/no_iqr --ext png --plot-title "cpr_server_value_srank_output" --no-show-iqr
# python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric nn/actor_srank/main/output_act --output-dir plots/main/no_iqr --ext png --plot-title "cpr_server_actor_srank_output" --no-show-iqr
# python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metrics value_srank_hidden --output-dir plots/main/no_iqr --ext png --plot-title "cpr_server_value_srank_hidden_main" --no-show-iqr
# python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metrics actor_srank_hidden --output-dir plots/main/no_iqr --ext png --plot-title "cpr_server_actor_srank_hidden_main" --no-show-iqr
# python analyze.py $WANDB_CFG --group slippery_ant_bigbatch --wandb_project crl_experiments --metrics srank_hidden --combine-networks --output-dir plots/bigbatch/combined/no_iqr --ext png --plot-title "S-rank Bigbatch" --no-show-iqr
#
# ################################################################################
# # Slippery Ant Results
# ################################################################################
# # With IQR
# python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric charts/mean_episodic_return --output-dir plots/main --ext png --bar-chart --y-min 0 --plot-title "Mean Episodic Return"
# python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric nn/value_dormant_neurons/total_ratio --output-dir plots/main --ext png --plot-title "cpr_server_value_dormant_neurons"
# python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric nn/actor_dormant_neurons/total_ratio --output-dir plots/main --ext png --plot-title "cpr_server_actor_dormant_neurons"
# python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric nn/value_linearised_neurons/total_ratio --output-dir plots/main --ext png --plot-title "cpr_server_value_linearised_neurons"
# python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric nn/actor_linearised_neurons/total_ratio --output-dir plots/main --ext png --plot-title "cpr_server_actor_linearised_neurons"
# python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric nn/policy_parameter_norm --output-dir plots/main --ext png --plot-title "cpr_server_policy_parameter_norm"
# python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric nn/vf_parameter_norm --output-dir plots/main --ext png --plot-title "cpr_server_vf_parameter_norm"
# python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric nn/vf_gradient_norm --output-dir plots/main --ext png --plot-title "Value Network Gradient Norm"
# python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric nn/policy_gradient_norm --output-dir plots/main --ext png --plot-title "Policy Network Gradient Norm"
# # python backup2_analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric nn/policy_gradient_norm --output-dir plots/main --ext png --plot-title "Policy Network Gradient Norm"
#
# # Without IQR
# python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric charts/mean_episodic_return --output-dir plots/main/no_iqr --ext png --bar-chart --y-min 0 --plot-title "Mean Episodic Return" --no-show-iqr
# python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric nn/value_dormant_neurons/total_ratio --output-dir plots/main/no_iqr --ext png --plot-title "cpr_server_value_dormant_neurons" --no-show-iqr
# python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric nn/actor_dormant_neurons/total_ratio --output-dir plots/main/no_iqr --ext png --plot-title "cpr_server_actor_dormant_neurons" --no-show-iqr
# python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric nn/value_linearised_neurons/total_ratio --output-dir plots/main/no_iqr --ext png --plot-title "cpr_server_value_linearised_neurons" --no-show-iqr
# python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric nn/actor_linearised_neurons/total_ratio --output-dir plots/main/no_iqr --ext png --plot-title "cpr_server_actor_linearised_neurons" --no-show-iqr
# python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric nn/policy_parameter_norm --output-dir plots/main/no_iqr --ext png --plot-title "cpr_server_policy_parameter_norm" --no-show-iqr
# python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric nn/vf_parameter_norm --output-dir plots/main/no_iqr --ext png --plot-title "cpr_server_vf_parameter_norm" --no-show-iqr
# python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric nn/vf_gradient_norm --output-dir plots/main/no_iqr --ext png --plot-title "Value Network Gradient Norm" --no-show-iqr
# python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric nn/policy_gradient_norm --output-dir plots/main/no_iqr --ext png --plot-title "Policy Network Gradient Norm" --no-show-iqr
# # python backup2_analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric nn/policy_gradient_norm --output-dir plots/main/no_iqr --ext png --plot-title "Policy Network Gradient Norm" --no-show-iqr
#
# ################################################################################
# # Large batch size adam experiment
# ################################################################################
# # With IQR
# python backup2_analyze.py $WANDB_CFG --group slippery_ant_bigbatch --wandb_project crl_experiments --metric charts/mean_episodic_return --output-dir plots/bigbatch --plot-title "Mean Episodic Return Bigbatch"
# python backup2_analyze.py $WANDB_CFG --group slippery_ant_bigbatch --wandb_project crl_experiments --metric dormant --combine-networks --output-dir plots/bigbatch/combined --ext png --plot-title "Dormant Neurons Bigbatch"
# python backup2_analyze.py $WANDB_CFG --group slippery_ant_bigbatch --wandb_project crl_experiments --metric linearized --combine-networks --output-dir plots/bigbatch/combined --ext png --plot-title "Linearized Neurons Bigbatch"
# python backup2_analyze.py $WANDB_CFG --group slippery_ant_bigbatch --wandb_project crl_experiments --metrics gradient_norm --combine-networks --output-dir plots/bigbatch/combined --ext png --plot-title "Gradient Norm Bigbatch"
# python analyze.py $WANDB_CFG --group slippery_ant_bigbatch --wandb_project crl_experiments --metric nn/value_dormant_neurons/total_ratio --output-dir plots/bigbatch --plot-title "bigbatch_value_dormant_neurons"
# python analyze.py $WANDB_CFG --group slippery_ant_bigbatch --wandb_project crl_experiments --metric nn/actor_dormant_neurons/total_ratio --output-dir plots/bigbatch --plot-title "bigbatch_actor_dormant_neurons"
# python analyze.py $WANDB_CFG --group slippery_ant_bigbatch --wandb_project crl_experiments --metric nn/value_linearised_neurons/total_ratio --output-dir plots/bigbatch --plot-title "bigbatch_value_linearised_neurons"
# python analyze.py $WANDB_CFG --group slippery_ant_bigbatch --wandb_project crl_experiments --metric nn/actor_linearised_neurons/total_ratio --output-dir plots/bigbatch --plot-title "bigbatch_actor_linearised_neurons"
#
# # Without IQR
# python backup2_analyze.py $WANDB_CFG --group slippery_ant_bigbatch --wandb_project crl_experiments --metric charts/mean_episodic_return --output-dir plots/bigbatch/no_iqr --plot-title "Mean Episodic Return Bigbatch" --no-show-iqr
# python backup2_analyze.py $WANDB_CFG --group slippery_ant_bigbatch --wandb_project crl_experiments --metric dormant --combine-networks --output-dir plots/bigbatch/combined/no_iqr --ext png --plot-title "Dormant Neurons Bigbatch" --no-show-iqr
# python backup2_analyze.py $WANDB_CFG --group slippery_ant_bigbatch --wandb_project crl_experiments --metric linearized --combine-networks --output-dir plots/bigbatch/combined/no_iqr --ext png --plot-title "Linearized Neurons Bigbatch" --no-show-iqr
# python backup2_analyze.py $WANDB_CFG --group slippery_ant_bigbatch --wandb_project crl_experiments --metrics gradient_norm --combine-networks --output-dir plots/bigbatch/combined/no_iqr --ext png --plot-title "Gradient Norm Bigbatch" --no-show-iqr
# python analyze.py $WANDB_CFG --group slippery_ant_bigbatch --wandb_project crl_experiments --metric nn/value_dormant_neurons/total_ratio --output-dir plots/bigbatch/no_iqr --plot-title "bigbatch_value_dormant_neurons" --no-show-iqr
# python analyze.py $WANDB_CFG --group slippery_ant_bigbatch --wandb_project crl_experiments --metric nn/actor_dormant_neurons/total_ratio --output-dir plots/bigbatch/no_iqr --plot-title "bigbatch_actor_dormant_neurons" --no-show-iqr
# python analyze.py $WANDB_CFG --group slippery_ant_bigbatch --wandb_project crl_experiments --metric nn/value_linearised_neurons/total_ratio --output-dir plots/bigbatch/no_iqr --plot-title "bigbatch_value_linearised_neurons" --no-show-iqr
# python analyze.py $WANDB_CFG --group slippery_ant_bigbatch --wandb_project crl_experiments --metric nn/actor_linearised_neurons/total_ratio --output-dir plots/bigbatch/no_iqr --plot-title "bigbatch_actor_linearised_neurons" --no-show-iqr
#
# ################################################################################
# # Dormant/collapse bar chart (no IQR applicable)
# ################################################################################
# python plot_policy_collapse.py --wandb-entity lucmc --group ccbp_server --wandb_project crl_final --metric charts/mean_episodic_return --collapse_threshold 8000 --output-dir plots/collapse_bar --overall --min-consecutive-below 10 --output-name cpr_server
#
# ################################################################################
# # Ablations - CPR (additional)
# ################################################################################
# # With IQR
# python ablation_plot.py --wandb-entity lucmc --wandb_project crl_experiments --group slippery_ant_transforms --metric charts/mean_episodic_return --grouping-mode parameter --split-by transform_type --ext png --output-dir plots/ablations/cpr --plot-title "Transformation Function Comparison" --no-show-metric-in-legend
# python ablation_plot.py --wandb-entity lucmc --wandb_project crl_experiments --group ccbp_replacement_rate --metric charts/mean_episodic_return --grouping-mode parameter --split-by replacement_rate --ext png --output-dir plots/ablations/cpr --plot-title "Stability-Plasticity Tuning" --y-tick-count 6
# python ablation_plot.py --wandb-entity lucmc --wandb_project crl_experiments --group slippery_ant_ccbp_sharpness_sweep --metric charts/mean_episodic_return --grouping-mode parameter --split-by sharpness --ext png --output-dir plots/ablations/cpr --plot-title "Sharpness Ablation" --sort-by-value
#
# # Without IQR
# python ablation_plot.py --wandb-entity lucmc --wandb_project crl_experiments --group slippery_ant_transforms --metric charts/mean_episodic_return --grouping-mode parameter --split-by transform_type --ext png --output-dir plots/ablations/cpr/no_iqr --plot-title "Transformation Function Comparison" --no-show-metric-in-legend --no-show-iqr
# python ablation_plot.py --wandb-entity lucmc --wandb_project crl_experiments --group ccbp_replacement_rate --metric charts/mean_episodic_return --grouping-mode parameter --split-by replacement_rate --ext png --output-dir plots/ablations/cpr/no_iqr --plot-title "Stability-Plasticity Tuning" --y-tick-count 6 --no-show-iqr
# python ablation_plot.py --wandb-entity lucmc --wandb_project crl_experiments --group slippery_ant_ccbp_sharpness_sweep --metric charts/mean_episodic_return --grouping-mode parameter --split-by sharpness --ext png --output-dir plots/ablations/cpr/no_iqr --plot-title "Sharpness Ablation" --sort-by-value --no-show-iqr
#
# ################################################################################
# # Combined network plots (unnormalized)
# ################################################################################
# # With IQR
# python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric dormant --combine-networks --output-dir plots/main/combined/unnorm --ext png --plot-title "Dormant Neurons"
# python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric linearized --combine-networks --output-dir plots/main/combined/unnorm --ext png --plot-title "Linearized Neurons"
python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metrics gradient_norm --combine-networks --output-dir plots/main/combined/unnorm --ext png --plot-title "CPR Gradient Norm"
# python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metrics srank_hidden --combine-networks --output-dir plots/main/combined/unnorm --ext png --plot-title "S-rank unnorm main"
#
# # Without IQR
# python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric dormant --combine-networks --output-dir plots/main/combined/unnorm/no_iqr --ext png --plot-title "Dormant Neurons" --no-show-iqr
# python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric linearized --combine-networks --output-dir plots/main/combined/unnorm/no_iqr --ext png --plot-title "Linearized Neurons" --no-show-iqr
python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metrics gradient_norm --combine-networks --output-dir plots/main/combined/unnorm/no_iqr --ext png --plot-title "CPR Gradient Norm" --no-show-iqr
# python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metrics srank_hidden --combine-networks --output-dir plots/main/combined/unnorm/no_iqr --ext png --plot-title "S-rank unnorm main" --no-show-iqr
#
# ################################################################################
# # Combined network plots (normalized)
# ################################################################################
# # With IQR
# python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric dormant --combine-networks --output-dir plots/main/combined --ext png --plot-title "Dormant Neurons"
# python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric linearized --combine-networks --output-dir plots/main/combined --ext png --plot-title "Linearized Neurons"
python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metrics gradient_norm --combine-networks --output-dir plots/main/combined --ext png --plot-title "CPR Gradient Norm"
# python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metrics srank_hidden --combine-networks --output-dir plots/main/combined --ext png --plot-title "S-rank"
# python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metrics parameter_norm --combine-networks --output-dir plots/main/combined --ext png --plot-title "Parameter Norm"
# python analyze.py $WANDB_CFG --group slippery_humanoid_full6 --wandb_project crl_humanoid --metrics parameter_norm --combine-networks --output-dir plots/main/combined/humanoid --ext png --plot-title "Parameter Norm Humanoid"
# python analyze.py --wandb-entity lucmc --wandb_project crl_experiments --group slippery_ant_transforms --metrics srank_hidden --combine-networks --output-dir plots/main/combined --ext png --plot-title "S-rank transforms"
#
# # Without IQR
# python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric dormant --combine-networks --output-dir plots/main/combined/no_iqr --ext png --plot-title "Dormant Neurons" --no-show-iqr
# python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric linearized --combine-networks --output-dir plots/main/combined/no_iqr --ext png --plot-title "Linearized Neurons" --no-show-iqr
python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metrics gradient_norm --combine-networks --output-dir plots/main/combined/no_iqr --ext png --plot-title "CPR Gradient Norm" --no-show-iqr
# python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metrics srank_hidden --combine-networks --output-dir plots/main/combined/no_iqr --ext png --plot-title "S-rank" --no-show-iqr
# python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metrics parameter_norm --combine-networks --output-dir plots/main/combined/no_iqr --ext png --plot-title "Parameter Norm" --no-show-iqr
# python analyze.py $WANDB_CFG --group slippery_humanoid_full6 --wandb_project crl_humanoid --metrics parameter_norm --combine-networks --output-dir plots/main/combined/humanoid/no_iqr --ext png --plot-title "Parameter Norm Humanoid" --no-show-iqr
# python analyze.py --wandb-entity lucmc --wandb_project crl_experiments --group slippery_ant_transforms --metrics srank_hidden --combine-networks --output-dir plots/main/combined/no_iqr --ext png --plot-title "S-rank transforms" --no-show-iqr
#
# ################################################################################
# # Slippery Humanoid
# ################################################################################
# # With IQR
python analyze.py $WANDB_CFG --group slippery_humanoid_full6 --wandb_project crl_humanoid --metric charts/mean_episodic_return --output-dir plots/humanoid --ext png --bar-chart --y-min 0 --plot-title "Mean Episodic Return"
python analyze.py $WANDB_CFG --group slippery_humanoid_test --wandb_project crl_experiments --metric charts/mean_episodic_return --output-dir plots/humanoid --ext png --bar-chart --y-min 0 --plot-title "Humanoid Test"
python analyze.py $WANDB_CFG --group slippery_humanoid_full6 --wandb_project crl_experiments --metric charts/mean_episodic_return --output-dir plots/humanoid --ext png --bar-chart --y-min 0 --plot-title "Episodic Return Humanoid"
python analyze.py $WANDB_CFG --group slippery_humanoid_full6 --wandb_project crl_humanoid --metrics dormant --combine-networks --output-dir plots/humanoid --ext png --plot-title "Dormant Neurons Humanoid"
python analyze.py $WANDB_CFG --group slippery_humanoid_full6 --wandb_project crl_humanoid --metrics linearized --combine-networks --output-dir plots/humanoid --ext png --plot-title "Linearized Neurons Humanoid"
python analyze.py $WANDB_CFG --group slippery_humanoid_full6 --wandb_project crl_humanoid --metrics gradient_norm --combine-networks --output-dir plots/humanoid --ext png --plot-title "CPR Gradient Norm Humanoid"

# Without IQR
python analyze.py $WANDB_CFG --group slippery_humanoid_full6 --wandb_project crl_humanoid --metric charts/mean_episodic_return --output-dir plots/humanoid/no_iqr --ext png --bar-chart --y-min 0 --plot-title "Mean Episodic Return" --no-show-iqr
python analyze.py $WANDB_CFG --group slippery_humanoid_test --wandb_project crl_experiments --metric charts/mean_episodic_return --output-dir plots/humanoid/no_iqr --ext png --bar-chart --y-min 0 --plot-title "Humanoid Test" --no-show-iqr
python analyze.py $WANDB_CFG --group slippery_humanoid_full6 --wandb_project crl_experiments --metric charts/mean_episodic_return --output-dir plots/humanoid/no_iqr --ext png --bar-chart --y-min 0 --plot-title "Episodic Return Humanoid" --no-show-iqr
python analyze.py $WANDB_CFG --group slippery_humanoid_full6 --wandb_project crl_humanoid --metrics dormant --combine-networks --output-dir plots/humanoid/no_iqr --ext png --plot-title "Dormant Neurons Humanoid" --no-show-iqr
python analyze.py $WANDB_CFG --group slippery_humanoid_full6 --wandb_project crl_humanoid --metrics linearized --combine-networks --output-dir plots/humanoid/no_iqr --ext png --plot-title "Linearized Neurons Humanoid" --no-show-iqr
python analyze.py $WANDB_CFG --group slippery_humanoid_full6 --wandb_project crl_humanoid --metrics gradient_norm --combine-networks --output-dir plots/humanoid/no_iqr --ext png --plot-title "CPR Gradient Norm Humanoid" --no-show-iqr

################################################################################
# Duplicate metrics (additional Slippery Ant)
################################################################################
# With IQR
python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric nn/value_dormant_neurons/total_ratio --output-dir plots/main --ext png --plot-title "cpr_server_value_dormant_neurons_2"
python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric nn/actor_dormant_neurons/total_ratio --output-dir plots/main --ext png --plot-title "cpr_server_actor_dormant_neurons_2"
python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric nn/value_linearised_neurons/total_ratio --output-dir plots/main --ext png --plot-title "cpr_server_value_linearised_neurons_2"
python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric nn/actor_linearised_neurons/total_ratio --output-dir plots/main --ext png --plot-title "cpr_server_actor_linearised_neurons_2"
python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric nn/policy_parameter_norm --output-dir plots/main --ext png --plot-title "cpr_server_policy_parameter_norm_2"
python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric nn/vf_parameter_norm --output-dir plots/main --ext png --plot-title "cpr_server_vf_parameter_norm_2"
python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric nn/vf_gradient_norm --output-dir plots/main --ext png --plot-title "CPR Value Network Gradient Norm"
python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric nn/policy_gradient_norm --output-dir plots/main --ext png --plot-title "CPR Policy Network Gradient Norm"

# Without IQR
python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric nn/value_dormant_neurons/total_ratio --output-dir plots/main/no_iqr --ext png --plot-title "cpr_server_value_dormant_neurons_2" --no-show-iqr
python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric nn/actor_dormant_neurons/total_ratio --output-dir plots/main/no_iqr --ext png --plot-title "cpr_server_actor_dormant_neurons_2" --no-show-iqr
python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric nn/value_linearised_neurons/total_ratio --output-dir plots/main/no_iqr --ext png --plot-title "cpr_server_value_linearised_neurons_2" --no-show-iqr
python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric nn/actor_linearised_neurons/total_ratio --output-dir plots/main/no_iqr --ext png --plot-title "cpr_server_actor_linearised_neurons_2" --no-show-iqr
python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric nn/policy_parameter_norm --output-dir plots/main/no_iqr --ext png --plot-title "cpr_server_policy_parameter_norm_2" --no-show-iqr
python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric nn/vf_parameter_norm --output-dir plots/main/no_iqr --ext png --plot-title "cpr_server_vf_parameter_norm_2" --no-show-iqr
python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric nn/vf_gradient_norm --output-dir plots/main/no_iqr --ext png --plot-title "CPR Value Network Gradient Norm" --no-show-iqr
python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric nn/policy_gradient_norm --output-dir plots/main/no_iqr --ext png --plot-title "CPR Policy Network Gradient Norm" --no-show-iqr

# ################################################################################
# # Threshold ablation
# ################################################################################
# # With IQR
# python compare_thresholds.py --wandb-entity lucmc --wandb-project crl_humanoid --group luc_humanoid_tau --metric charts/mean_episodic_return --threshold-1 1.0 --threshold-2 0.95 --pattern-1 "ccbp_br_adam_sb_lr_smaller_net_*" --pattern-2 "ccbp_bigger_rollout_new_hparams_*" --bar-chart --base-text-size 20.0 --y-min -2000 --plot-title "SlipperyHumanoid threshold comparison"
# python compare_thresholds.py --wandb-entity lucmc --wandb-project crl_final --group ant_tau_1 --metric charts/mean_episodic_return --threshold-1 1.0 --threshold-2 0.95 --pattern-1 "ccbp_smaller_*" --pattern-2 "ccbp_s*-copy" --bar-chart --base-text-size 20.0 --y-min -2000 --plot-title "SlipperyAnt threshold comparison"
#
# # Without IQR
# python compare_thresholds.py --wandb-entity lucmc --wandb-project crl_humanoid --group luc_humanoid_tau --metric charts/mean_episodic_return --threshold-1 1.0 --threshold-2 0.95 --pattern-1 "ccbp_br_adam_sb_lr_smaller_net_*" --pattern-2 "ccbp_bigger_rollout_new_hparams_*" --bar-chart --base-text-size 20.0 --y-min -2000 --plot-title "SlipperyHumanoid threshold comparison" --no-show-iqr
# python compare_thresholds.py --wandb-entity lucmc --wandb-project crl_final --group ant_tau_1 --metric charts/mean_episodic_return --threshold-1 1.0 --threshold-2 0.95 --pattern-1 "ccbp_smaller_*" --pattern-2 "ccbp_s*-copy" --bar-chart --base-text-size 20.0 --y-min -2000 --plot-title "SlipperyAnt threshold comparison" --no-show-iqr
