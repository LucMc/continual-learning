 # python analyze.py $WANDB_CFG --group slippery_ant_full --metric charts/mean_episodic_return --output-dir plots/main # Maybe make yaml files for each experiment or something?
 # python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric charts/mean_episodic_return --output-dir plots/main # Maybe make yaml files for each experiment or something?
 
 # Slippery Ant Rseults
 # python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric charts/mean_episodic_return --output-dir plots/main --ext png
 # python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric nn/value_dormant_neurons/total_ratio --output-dir plots/main --ext png
 # python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric nn/actor_dormant_neurons/total_ratio --output-dir plots/main --ext png
 # python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric nn/value_linearised_neurons/total_ratio --output-dir plots/main --ext png
 # python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric nn/actor_linearised_neurons/total_ratio --output-dir plots/main --ext png
 # python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric nn/policy_parameter_norm --output-dir plots/main --ext png
 # python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric nn/vf_parameter_norm --output-dir plots/main --ext png
 # python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric nn/vf_gradient_norm --output-dir plots/main --ext png
 # python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric nn/policy_gradient_norm --output-dir plots/main --ext png

 # python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric nn/value_srank/layer_0_act --output-dir plots/main --ext png
 # python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric nn/policy_srank/layer_0_act --output-dir plots/main --ext png
 # python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric nn/value_srank/layer_1_act --output-dir plots/main --ext png
 # python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric nn/policy_srank/layer_1_act --output-dir plots/main --ext png
 # python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric nn/value_srank/layer_2_act --output-dir plots/main --ext png
 # python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric nn/policy_srank/layer_2_act --output-dir plots/main --ext png
 # python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric nn/value_srank/layer_3_act --output-dir plots/main --ext png
 # python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric nn/policy_srank/layer_3_act --output-dir plots/main --ext png

 # Large batch size adam experiment
 # python analyze.py $WANDB_CFG --group standard_large_batchsize --wandb_project crl_final --metric charts/mean_episodic_return --output-dir plots/bigbatch
 # python analyze.py $WANDB_CFG --group standard_large_batchsize --wandb_project crl_final --metric nn/value_dormant_neurons/total_ratio --output-dir plots/bigbatch
 # python analyze.py $WANDB_CFG --group standard_large_batchsize --wandb_project crl_final --metric nn/actor_dormant_neurons/total_ratio --output-dir plots/bigbatch
 # python analyze.py $WANDB_CFG --group standard_large_batchsize --wandb_project crl_final --metric nn/value_linearised_neurons/total_ratio --output-dir plots/bigbatch
 # python analyze.py $WANDB_CFG --group standard_large_batchsize --wandb_project crl_final --metric nn/actor_linearised_neurons/total_ratio --output-dir plots/bigbatch

 # Dormant/collapse bar chart
 # python plot_policy_collapse.py --wandb-entity lucmc --group ccbp_server --wandb_project crl_final --metric charts/mean_episodic_return --collapse_threshold 9000 --output-dir plots/collapse_bar

 # Perm MNIST
 python analyze.py $WANDB_CFG --group perm_mnist_final --wandb_project crl_final --metric metrics/eval_accuracy_ci --output-dir plots/perm_mnist --ext png
 # python analyze.py $WANDB_CFG --group perm_mnist_final --wandb_project crl_final --metric nn/value_dormant_neurons/total_ratio --output-dir plots/perm_mnist --ext png
 # python analyze.py $WANDB_CFG --group perm_mnist_final --wandb_project crl_final --metric nn/actor_dormant_neurons/total_ratio --output-dir plots/perm_mnist --ext png
 # python analyze.py $WANDB_CFG --group perm_mnist_final --wandb_project crl_final --metric nn/value_linearised_neurons/total_ratio --output-dir plots/perm_mnist --ext png
 # python analyze.py $WANDB_CFG --group perm_mnist_final --wandb_project crl_final --metric nn/actor_linearised_neurons/total_ratio --output-dir plots/perm_mnist --ext png

 # Ablations
     # CCBP
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

     # shrink_pertub
     # python ablation_plot.py --wandb-entity lucmc --wandb_project crl_experiments --group slippery_ant_shrink_perturb_sweep --metric charts/mean_episodic_return --grouping-mode parameter --split-by shrink --ext png --output-dir plots/ablations/sp
     # python ablation_plot.py --wandb-entity lucmc --wandb_project crl_experiments --group slippery_ant_shrink_perturb_sweep --metric charts/mean_episodic_return --grouping-mode parameter --split-by pertub --ext png --output-dir plots/ablations/sp
     
     # Just for which runs are best
     # python ablation_plot.py --wandb-entity lucmc --wandb_project crl_experiments --group slippery_ant_cbp_sweep --metric charts/mean_episodic_return --grouping-mode config --ext png --output-dir plots/ablations/all --short-labels --ranking-criteria average --top-k 5
     # python ablation_plot.py --wandb-entity lucmc --wandb_project crl_experiments --group slippery_ant_redo_sweep --metric charts/mean_episodic_return --grouping-mode config --ext png --output-dir plots/ablations/all --short-labels --ranking-criteria average --top-k 5
     # python ablation_plot.py --wandb-entity lucmc --wandb_project crl_experiments --group slippery_ant_regrama_sweep --metric charts/mean_episodic_return --grouping-mode config --ext png --output-dir plots/ablations/all --short-labels --ranking-criteria average --top-k 5
     # Combined
     # python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metrics dormant,linearized --combine-networks --output-dir plots/main --ext png

