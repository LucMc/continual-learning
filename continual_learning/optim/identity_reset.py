import optax

from flax import struct
from continual_learning.types import GradientTransformationExtraArgsReset
from jaxtyping import PyTree


class IdentityState(struct.PyTreeNode):
    logs: dict


def identity_reset(*args, **kwargs):
    """Identity reset method"""
    del args, kwargs

    def init_fn(params, *args, **kwargs):
        del params, args, kwargs
        return IdentityState(logs={})

    def update_fn(
        updates: optax.Updates,
        state: optax.OptState,
        params: optax.Params,
        features: PyTree,
        tx_state: optax.OptState,
    ) -> tuple[optax.Updates, optax.OptState, optax.OptState]:
        del updates, features
        return params, state, tx_state

    return GradientTransformationExtraArgsReset(init=init_fn, update=update_fn)  # pyright: ignore[reportArgumentType]
