from flax.core import FrozenDict
from jax.random import PRNGKey
from jaxtyping import Array, Float, PRNGKeyArray, PyTree
from flax.training.train_state import TrainState
from typing import Tuple
from chex import dataclass
import optax
import jax
import jax.random as random
import jax.numpy as jnp
from copy import deepcopy
from functools import partial

# class ContinualBackprop(optax.GradientTransformation):
#     def __init__(
#         self,
#         threshold: float = 0.1,
#         sparsity: float = 0.5,
#         begin_step: int = 0,
#         end_step: None | int = None,
#         frequency: int = 100,
#     ):
#         """Continual backpropergation optimiser.
#
#         Args:
#             threshold: Magnitude threshold below which to prune
#             sparsity: Target sparsity ratio (0 to 1)
#             begin_step: Step to begin pruning
#             end_step: Step to end pruning (None = no end)
#             frequency: How often to update pruning mask
#         """
#         self.threshold = threshold
#         self.sparsity = sparsity
#         self.begin_step = begin_step
#         self.end_step = end_step
#         self.frequency = frequency
#
#     def init(self, params):
#         """Initialize pruning state with mask of ones."""
#         return PruningState(mask=jax.tree_map(lambda x: jnp.ones_like(x), params))
#
#     def update_mask(self, params, state):
#         """Update pruning mask based on parameter magnitudes."""
#         # Calculate magnitude of parameters
#         magnitudes = jax.tree_map(lambda x: jnp.abs(x), params)
#
#         # Find threshold for desired sparsity
#         flat_magnitudes = jax.tree_util.tree_leaves(magnitudes)
#         flat_magnitudes = jnp.concatenate([x.ravel() for x in flat_magnitudes])
#         threshold = jnp.quantile(flat_magnitudes, self.sparsity)
#
#         # Create new mask
#         new_mask = jax.tree_map(
#             lambda x: jnp.where(jnp.abs(x) > threshold, 1.0, 0.0), params
#         )
#         return new_mask
#
#     def update(self, updates, state, params=None):
#         """Apply pruning transform to updates."""
#         del params
#
#         # Increment counter
#         new_count = state.count + 1
#
#         # Update mask if needed
#         should_update = (
#             (new_count >= self.begin_step)
#             & ((self.end_step is None) | (new_count <= self.end_step))
#             & (new_count % self.frequency == 0)
#         )
#
#         def update_fn(mask, update):
#             # Apply existing mask
#             pruned_update = update * mask
#             return pruned_update
#
#         # Apply mask to updates
#         new_updates = jax.tree_map(update_fn, state.mask, updates)
#
#         return new_updates, PruningState(mask=state.mask, count=new_count)

UTIL_TYPES = [
    "weight",
    "contribution",
    "adaptation",
    "zero_contribution",
    "adaptable_contribution",
    "feature_by_input",
]


class CBPTrainState(TrainState):
    def apply_gradients(self, *, grads, features, **kwargs):
        """TrainState that gives intermediates to optimizer and overwrites params with updates directly"""
        # Extract the params we want to optimize
        grads_with_opt = grads["params"]
        params_with_opt = self.params["params"]

        # Update with optimizer
        # updates, new_opt_state = self.tx.update(
        #     grads_with_opt, self.opt_state, params_with_opt, features=features
        # )
        updates, new_opt_state = continual_backprop().update(
            grads_with_opt,
            self.opt_state[0],
            params_with_opt,
            features=features["intermediates"]["activations"][0],
        )
        new_params_with_opt = optax.apply_updates(params_with_opt, updates)

        # Maintain the nested structure of params
        new_params = {"params": new_params_with_opt}

        return self.replace(
            step=self.step + 1,
            params=new_params,
            opt_state=new_opt_state,
            **kwargs,
        )


@dataclass
class CBPOptimState:
    # Things you shouldn't really mess with
    utilities: Float[Array, "#n_layers"]
    mean_feature_act: Float[Array, ""]
    ages: Array
    util_type_id: int
    accumulated_features_to_replace: int

    rng: PRNGKeyArray  # = random.PRNGKey(0)
    step_size: float = 0.001
    replacement_rate: float = 0.001
    decay_rate: float = 0.9
    maturity_threshold: int = 2  # 100
    accumulate: bool = False


# Give this the ContinualBackpropState params
def continual_backprop(
    util_type: str = "contribution", **kwargs
) -> optax.GradientTransformation:
    """
    Since when applying gradients you do params + updates, if we make the update
    to dead nodes (-weight value + reinit value) this would be the same as reinitialising
    """

    def process_params(params: FrozenDict):
        # seperates bias from params
        _params = deepcopy(params)#["params"]
        _params.pop("out_layer")  # As it doesn't pass through activation

        bias = {}
        weights = {}

        for layer_name in _params.keys():
            bias[layer_name] = _params[layer_name].pop("bias")
            weights[layer_name] = _params[layer_name].pop("kernel")

        return weights, bias

    def init(params: optax.Params):
        assert util_type in UTIL_TYPES, ValueError(
            f"Invalid util type, select from ({'|'.join(UTIL_TYPES)})"
        )
        weights, bias = process_params(params["params"])

        # Delete params?
        return CBPOptimState(
            utilities=jax.tree.map(
                lambda layer: jnp.ones_like(layer), weights
            ),  # All utils are 1 initially
            mean_feature_act=jnp.zeros(0),
            ages=jax.tree_map(
                lambda x: jnp.zeros_like(x),
                weights,  # should be bias
            ),  # 1 neuron has 1 bias and n_hidden weights
            util_type_id=UTIL_TYPES.index(
                util_type
            ),  # Replace with util function directly?
            accumulated_features_to_replace=0,
            rng=random.PRNGKey(0),
            **kwargs,
        )

    def gen_key_tree(key: PRNGKeyArray, tree: PyTree):
        """
        Creates a PyTree of random keys such that is can be traversed in the tree map and have
        a new key for each leaf.
        """
        leaves, treedef = jax.tree_flatten(tree)
        subkeys = jax.random.split(key, len(leaves))
        return jax.tree_unflatten(treedef, subkeys)

    # @jax.jit
    def update(
        updates: optax.Updates,  # Gradients
        state: optax.OptState,
        params: optax.Params | None = None,
        features: PyTree | None = None,
    ) -> tuple[optax.Updates, CBPOptimState]:
        # util_functions = [
        #     lambda x: output_weight_mag, # weight
        #     lambda x: weight_mag * features.abs().mean(dim=1), # contribution
        # lambda x: x, # adaptation
        # lambda x: x, # zero_contribution
        # lambda x: x, # adaptable_contribution
        # lambda x: x, # feature_by_input
        # ]
        # Calculate new_util based on util_type
        # util_function =

        # return -p + new_param  # (p > 0) * p
        def update_utility(
            layer: Array, utility: Array, ages: Array, key: PRNGKey, bound: float = 0.01
        ):
            """
            TODO:
            > loops through weights and bias, skip biases if not needed note this will need to happen for util initialisation too
            > features are after activation, can maybe just assume relu but remember last layer has no activation
            > make actual bound
            > Don't need an age for each connection, only each neuron in the network
            """

            # Maybe have a dictionary of the different util func transformations and then call the index in a cond
            new_param = layer * state.decay_rate

            features = layer  # Should be relu(wx+b), might have to overwrite apply updates like Angel said
            updated_utility = (state.decay_rate * layer) + (
                1 - state.decay_rate
            ) * jnp.abs(features) * jnp.sum(layer)

            key = random.PRNGKey(0)
            # get nodes over maturity threshold
            mature_features_idx = jnp.where(ages > state.maturity_threshold)[0]
            # mature_features_idx = jnp.argmax()
            n_to_replace = int(mature_features_idx.shape[0] * state.replacement_rate)
            mature_utils = updated_utility[mature_features_idx]
            idx_nodes_to_reset = mature_utils[
                jnp.argsort(mature_utils)[-n_to_replace:][::-1]
            ]

            # reset input AND output weights AND bias terms
            # resetting outbound connections [128] per node

            if len(idx_nodes_to_reset) > 0:
                _layer = layer.at[idx_nodes_to_reset].set(
                    random.uniform(key, layer.shape[1], float, -bound, bound)
                )
                _ages = ages.at[idx_nodes_to_reset].set(0.0)
            else:
                _layer = layer
                _ages = ages

            # For now let's just ignore activation and calculate the outputbased i+
            return (
                _layer,
            )  # ages_  # , ages+1, {"n_nodes_to_reset": len(nodes_to_reset)}, I do need to reutn ages but just for now

        def _continual_backprop(
            updates: optax.Updates,
        ) -> Tuple[optax.Updates, CBPOptimState]:
            """
            This feels like it could be made more efficient. Perhaps I should combine bias and weights into one 2D array and just do it for the
            one layer now before vmapping over multiple layers/tree vmapping latter
            """
            weights, bias = process_params(params)

            # because we need the next layers weight magnitude
            _ages = jax.tree.map(lambda x: x + 1, state.ages)
            _rng, util_key = random.split(state.rng)
            key_tree = gen_key_tree(util_key, weights)

            # update_utility
            new_weights = jax.tree.map(
                update_utility, weights, features, state.utilities, state.ages, key_tree
            )  # , next_layer_weight_sum) # Instead of applying to each neuron we split it into layers now

            # Would generating a rondom number and if it is bellow the threshold then rplace if elegible be better as it introduces more randomness?
            # num_new_features_to_replace = state.replacement_rate * eligable_features_to_replace
            # new_accumulated_features_to_replace += features_to_replace
            # I actually think it's this layers weight sum since this layer weights connect to next
            # See """calculate feature utility""" because it looks a little different, certainly need features

            # new_state = ContinualBackpropState(
            #     # utilities=_utilities,
            #     # mean_feature_act=_mean_feature_act,
            #     ages=_ages,
            #     # util_type_id=_util_type_id,
            #     # accumulated_features_to_replace=_accumulated_features_to_replace,
            #     rng=_rng,
            #     # step_size=_step_size,
            #     # replacement_rate=_replacement_rate,
            #     # decay_rate=_decay_rate,
            #     # maturity_threshold=_maturity_threshold,
            #     # accumulate=_accumulate,
            # )
            new_state = state

            return params, (new_state,)  # For now

        return _continual_backprop(updates)  # updates, ContinualBackpropState()

    return optax.GradientTransformation(init=init, update=update)


""" old code snippets:
    
            # bias_correction = 1 - state.decay_rate ** self.ages
            # bias_correction = jax.tree.map(
            #     lambda a: 1 - state.decay_rate**a, state.ages
            # )

            # layerwise_utility = jax.vmap(utility, in_axes=(0, None))(params) # Expect (layer_n, params) and map over layer_n
            # next_layer_weight_sum = jax.tree.map(lambda layer: layer.sum(), params) # Instead of applying to each neuron we split it into layers now
            # updated_utility = (state.decay_rate * layer) + (1-state.decay_rate) * jnp.abs(features) * next_layer_weight_sum
            # idx_nodes_to_reset = lax.top_k_idx(, )
"""
