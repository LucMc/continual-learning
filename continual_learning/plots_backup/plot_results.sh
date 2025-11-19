 # Slippery Ant Results
 # python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric charts/mean_episodic_return --output-dir plots/main --ext png --bar-chart --plot-title "Mean Episodic Return2" --no-show-iqr 
 # python analyze.py $WANDB_CFG --group cbp_new --wandb_project crl_experiments --metric charts/mean_episodic_return --output-dir plots/main --ext png --bar-chart --plot-title "cbp test3"
 # python analyze.py --wandb_entity cavlab --group slippery_ant_full2 --wandb_project test_ant --metric charts/mean_episodic_return --output-dir plots/main --ext png --bar-chart --plot-title "Mean Episodic Return test"
 #
 # python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric nn/value_dormant_neurons/total_ratio --output-dir plots/main --ext png
 # python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric nn/actor_dormant_neurons/total_ratio --output-dir plots/main --ext png
 # python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric nn/value_linearised_neurons/total_ratio --output-dir plots/main --ext png
 # python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric nn/actor_linearised_neurons/total_ratio --output-dir plots/main --ext png
 # python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric nn/policy_parameter_norm --output-dir plots/main --ext png
 # python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric nn/vf_parameter_norm --output-dir plots/main --ext png
 # python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric nn/vf_gradient_norm --output-dir plots/main --ext png
 # python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric nn/policy_gradient_norm --output-dir plots/main --ext png
 #
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
 #
 # # Large batch size adam experiment
 # python analyze.py $WANDB_CFG --group standard_large_batchsize --wandb_project crl_final --metric charts/mean_episodic_return --output-dir plots/bigbatch
 # python analyze.py $WANDB_CFG --group standard_large_batchsize --wandb_project crl_final --metric nn/value_dormant_neurons/total_ratio --output-dir plots/bigbatch
 # python analyze.py $WANDB_CFG --group standard_large_batchsize --wandb_project crl_final --metric nn/actor_dormant_neurons/total_ratio --output-dir plots/bigbatch
 # python analyze.py $WANDB_CFG --group standard_large_batchsize --wandb_project crl_final --metric nn/value_linearised_neurons/total_ratio --output-dir plots/bigbatch
 # python analyze.py $WANDB_CFG --group standard_large_batchsize --wandb_project crl_final --metric nn/actor_linearised_neurons/total_ratio --output-dir plots/bigbatch
 #
 # # Dormant/collapse bar chart
 # python plot_policy_collapse.py --wandb-entity lucmc --group ccbp_server --wandb_project crl_final --metric charts/mean_episodic_return --collapse_threshold 9000 --output-dir plots/collapse_bar

 # Perm MNIST
 # python analyze.py $WANDB_CFG --group perm_mnist_final --wandb_project crl_final --metric metrics/eval_accuracy_ci --output-dir plots/perm_mnist --ext png
 # python analyze.py $WANDB_CFG --group perm_mnist_final --wandb_project crl_final --metric nn/value_dormant_neurons/total_ratio --output-dir plots/perm_mnist --ext png
 # python analyze.py $WANDB_CFG --group perm_mnist_final --wandb_project crl_final --metric nn/actor_dormant_neurons/total_ratio --output-dir plots/perm_mnist --ext png
 # python analyze.py $WANDB_CFG --group perm_mnist_final --wandb_project crl_final --metric nn/value_linearised_neurons/total_ratio --output-dir plots/perm_mnist --ext png
 # python analyze.py $WANDB_CFG --group perm_mnist_final --wandb_project crl_final --metric nn/actor_linearised_neurons/total_ratio --output-dir plots/perm_mnist --ext png

 # Ablations
     # CCBP
     python ablation_plot.py --wandb-entity lucmc --wandb_project crl_experiments --group slippery_ant_transforms --metric charts/mean_episodic_return --grouping-mode parameter --split-by transform_type --ext png --output-dir plots/ablations/ccbp --plot-title "Transformation Function Comparison" --no-show-iqr
     # python ablation_plot.py --wandb-entity lucmc --wandb_project crl_experiments --group ccbp_replacement_rate --metric charts/mean_episodic_return --grouping-mode parameter --split-by replacement_rate --ext png --output-dir plots/ablations/ccbp --plot-title "Stability-Plastiticy Tuning"

     # python ablation_plot.py --wandb-entity lucmc --wandb_project crl_experiments --group slippery_ant_ccbp_sweep --metric nn/value_dormant_neurons/total_ratio --grouping-mode parameter --split-by replacement_rate --ext png --output-dir plots/ablations/ccbp
     # python ablation_plot.py --wandb-entity lucmc --wandb_project crl_experiments --group slippery_ant_ccbp_sweep --metric nn/actor_dormant_neurons/total_ratio --grouping-mode parameter --split-by replacement_rate --ext png --output-dir plots/ablations/ccbp
     # python ablation_plot.py --wandb-entity lucmc --wandb_project crl_experiments --group slippery_ant_ccbp_sweep --metric nn/value_linearised_neurons/total_ratio --grouping-mode parameter --split-by replacement_rate --ext png --output-dir plots/ablations/ccbp
     # python ablation_plot.py --wandb-entity lucmc --wandb_project crl_experiments --group slippery_ant_ccbp_sweep --metric nn/actor_linearised_neurons/total_ratio --grouping-mode parameter --split-by replacement_rate --ext png --output-dir plots/ablations/ccbp
     #
     # python ablation_plot.py --wandb-entity lucmc --wandb_project crl_experiments --group slippery_ant_ccbp_sweep --metric nn/vf_gradient_norm --grouping-mode parameter --split-by replacement_rate --ext png --output-dir plots/ablations/ccbp
     # python ablation_plot.py --wandb-entity lucmc --wandb_project crl_experiments --group slippery_ant_ccbp_sweep --metric nn/policy_gradient_norm --grouping-mode parameter --split-by replacement_rate --ext png --output-dir plots/ablations/ccbp
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
     
     # Combined
     # python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric dormant --combine-networks --output-dir plots/main/combined --ext png --normalize-networks --plot-title "Dormant Neurons" --no-show-iqr
     # python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric linearized --combine-networks --output-dir plots/main/combined --ext png --normalize-networks --plot-title "Linearized Neurons" --no-show-iqr
     # python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metrics actor_srank_hidden --combine-networks --output-dir plots/main/combined --ext png --normalize-networks --plot-title "S-rank" --no-show-iqr
     # python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metrics gradient_norm --combine-networks --output-dir plots/main/combined --ext png --normalize-networks --plot-title "Gradient Norm" --no-show-iqr
     
 
 # Slippery Humanoid
 # python analyze.py $WANDB_CFG --group slippery_humanoid_test --wandb_project crl_experiments --metric charts/mean_episodic_return --output-dir plots/humanoid --ext png --bar-chart
 # python analyze.py $WANDB_CFG --group slippery_humanoid_full6 --wandb_project crl_humanoid --metric charts/mean_episodic_return --output-dir plots/humanoid --ext png --bar-chart --plot-title "Mean Episodic Return"
 # python analyze.py $WANDB_CFG --group slippery_humanoid_full6 --wandb_project crl_experiments --metric charts/mean_episodic_return --output-dir plots/humanoid --ext png --bar-chart --plot-title "Episodic Return"
 
 # python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric nn/value_dormant_neurons/total_ratio --output-dir plots/main --ext png
 # python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric nn/actor_dormant_neurons/total_ratio --output-dir plots/main --ext png
 # python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric nn/value_linearised_neurons/total_ratio --output-dir plots/main --ext png
 # python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric nn/actor_linearised_neurons/total_ratio --output-dir plots/main --ext png
 # python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric nn/policy_parameter_norm --output-dir plots/main --ext png
 # python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric nn/vf_parameter_norm --output-dir plots/main --ext png
 # python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric nn/vf_gradient_norm --output-dir plots/main --ext png
 # python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric nn/policy_gradient_norm --output-dir plots/main --ext png
 #
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
