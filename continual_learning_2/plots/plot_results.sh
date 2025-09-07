 # python analyze.py $WANDB_CFG --group slippery_ant_full --metric charts/mean_episodic_return --output-dir plots/main # Maybe make yaml files for each experiment or something?
 # python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric charts/mean_episodic_return --output-dir plots/main # Maybe make yaml files for each experiment or something?
 #
 python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric charts/mean_episodic_return --output-dir plots/main
 python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric nn/value_dormant_neurons/total_ratio --output-dir plots/main
 python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric nn/actor_dormant_neurons/total_ratio --output-dir plots/main
 python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric nn/value_linearised_neurons/total_ratio --output-dir plots/main
 python analyze.py $WANDB_CFG --group ccbp_server --wandb_project crl_final --metric nn/actor_linearised_neurons/total_ratio --output-dir plots/main
 
 # Large batch size adam experiment
 # python analyze.py $WANDB_CFG --group standard_large_batchsize --wandb_project crl_final --metric charts/mean_episodic_return --output-dir plots/bigbatch
 # python analyze.py $WANDB_CFG --group standard_large_batchsize --wandb_project crl_final --metric nn/value_dormant_neurons/total_ratio --output-dir plots/bigbatch
 # python analyze.py $WANDB_CFG --group standard_large_batchsize --wandb_project crl_final --metric nn/actor_dormant_neurons/total_ratio --output-dir plots/bigbatch
 # python analyze.py $WANDB_CFG --group standard_large_batchsize --wandb_project crl_final --metric nn/value_linearised_neurons/total_ratio --output-dir plots/bigbatch
 # python analyze.py $WANDB_CFG --group standard_large_batchsize --wandb_project crl_final --metric nn/actor_linearised_neurons/total_ratio --output-dir plots/bigbatch

 python plot_policy_collapse.py --wandb-entity lucmc --group ccbp_server --wandb_project crl_final --metric charts/mean_episodic_return --collapse_threshold 9000 --output-dir plots/collapse_bar
