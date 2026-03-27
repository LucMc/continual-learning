"""Discrete action space models for Discrete SAC.

CategoricalPolicy: outputs a Categorical distribution over discrete actions.
DiscreteQNetwork: twin Q-networks that take obs only and output Q(s, :) for all actions.
"""

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp

from continual_learning.configs.rl import PolicyNetworkConfig, QNetworkConfig


class CategoricalPolicy(nn.Module):
    """Policy for discrete action spaces.

    Outputs a Categorical distribution over n_actions discrete actions.
    The backbone MLP output_size is overridden to n_actions.
    """

    network: type[nn.Module]
    config: PolicyNetworkConfig
    n_actions: int
    input_spatial_shape: tuple[int, ...] | None = None

    @nn.compact
    def __call__(self, x: jax.Array, training: bool = False) -> distrax.Categorical:
        if self.input_spatial_shape is not None:
            x = x.reshape(*x.shape[:-1], *self.input_spatial_shape)
        network_config = self.config.network.replace(output_size=self.n_actions)
        logits = self.network(config=network_config, name="main")(x, training=training)
        return distrax.Categorical(logits=logits.astype(jnp.float32))


class DiscreteQNetwork(nn.Module):
    """Twin Q-network for discrete SAC.

    Takes observation only (no action concatenation) and outputs Q-values for
    all actions. Returns (q1, q2) each of shape (batch, n_actions).
    The backbone MLP output_size is overridden to n_actions.
    """

    network: type[nn.Module]
    config: QNetworkConfig
    n_actions: int
    input_spatial_shape: tuple[int, ...] | None = None

    @nn.compact
    def __call__(
        self, obs: jax.Array, training: bool = False
    ) -> tuple[jax.Array, jax.Array]:
        if self.input_spatial_shape is not None:
            obs = obs.reshape(*obs.shape[:-1], *self.input_spatial_shape)
        q1_config = self.config.network.replace(output_size=self.n_actions)
        q2_config = self.config.network.replace(output_size=self.n_actions)
        q1 = self.network(config=q1_config, name="q1")(obs, training=training)
        q2 = self.network(config=q2_config, name="q2")(obs, training=training)
        return q1.astype(jnp.float32), q2.astype(jnp.float32)
