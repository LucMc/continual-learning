import jax.numpy as jnp
import jax.tree_util as tu

reset_info = {
    'dense1': {'ages': jnp.zeros(10), 'logs': {'avg_age': jnp.zeros(10)}, 'mask': jnp.zeros(10)},
    'dense2': {'ages': jnp.zeros(10), 'logs': {'avg_age': jnp.zeros(10)}, 'mask': jnp.zeros(10)},
    'dense3': {'ages': jnp.zeros(10), 'logs': {'avg_age': jnp.zeros(10)}, 'mask': jnp.zeros(10)}
}

# Define the TreeDefs

l, tdef = tu.tree_flatten(transposed)
print("transposed:\n", tdef)
