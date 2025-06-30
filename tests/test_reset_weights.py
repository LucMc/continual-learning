import jax
import jax.numpy as jnp
from continual_learning_2.models import get_model
from continual_learning_2.optim import get_optimizer
from continual_learning_2.configs import MLPConfig, CBPConfig, AdamConfig
from continual_learning_2.utils.training import TrainState
import continual_learning_2.utils.optim as utils

def test_weight_reset():
    # Create dummy params
    model_conf = MLPConfig(output_size=2, hidden_size=4)
    optim_conf = CBPConfig(
        tx=AdamConfig(learning_rate=1e-3),
        decay_rate=0.9,
        replacement_rate=0.5
    )
    model = get_model(model_conf)
    optimizer = get_optimizer(optim_conf)
    
    key = jax.random.PRNGKey(0)
    dummy_input = jnp.zeros((1, 2))
    
    network = TrainState.create(
        apply_fn=jax.jit(model.apply, static_argnames=("training", "mutable")),
        params=model.init(key, dummy_input),
        tx=optimizer,
        kernel_init=model_conf.kernel_init,
        bias_init=model_conf.bias_init,
    )
    
    assert network.params is not None
    test_params = jax.tree.map(lambda x: jnp.ones_like(x), network.params["params"])
    replacement_rate = 1

    # process params
    w, b, owm, exl = utils.process_params_with_outmag(test_params)

    """
    Weights are stored as [into_layer x out_layer].
    In a full reset the input weights (first dim) should be reinitialised,
    and output weights zeroed out. This test imagines initial weights are 0.5.
    By reseting all weights, we are setting all inbound (in layer_0) to 0.5 and all outbound (in layer_1)
    in the next layer to 0.
    """
    full_reset_mask = jax.tree.map(lambda x: jnp.ones_like(x)==1, b)
    initial_w = jax.tree.map(lambda x: jnp.ones_like(x)/2, w)

    new_weights, logs = utils.reset_weights(full_reset_mask, w, initial_w, replacement_rate)
    assert jnp.all(new_weights["layer_0"] == 0.5)
    assert jnp.all(new_weights["layer_1"] == 0.)
