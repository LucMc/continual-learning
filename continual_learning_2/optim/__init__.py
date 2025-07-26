import optax

from continual_learning_2.configs.optim import AdamConfig, OptimizerConfig


def get_optimizer(config: OptimizerConfig):
    if isinstance(config, AdamConfig):
        return optax.chain(
            # optax.clip_by_global_norm(1.0),
            optax.adam(
                config.learning_rate, b1=config.beta1, b2=config.beta2, eps=config.epsilon
            ),
        )
    else:
        raise ValueError(f"Unsupported optimizer config type: {type(config)}")
