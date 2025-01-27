from functools import partial
from typing import Dict, Tuple, Any

import jax
import jax.numpy as jnp

# from jax.experimental import optimizers
# from jax.experimental.optimizers import adam
import optax
from flax.training.train_state import TrainState


class BackpropJax():
    def __init__(self, net, step_size=0.001, loss='mse', opt='sgd', beta_1=0.9, beta_2=0.999, weight_decay=0.0,
                 to_perturb=False, perturb_scale=0.1, device='cpu', momentum=0):
        self.net = net
        self.to_perturb = to_perturb
        self.perturb_scale = perturb_scale
        self.device = device

        # define the optimizer
        if opt == 'sgd':
            self.tx = optax.sgd(learning_rate=step_size, momentum=momentum) # no weight decay option
        elif opt == 'adam':
            self.tx = optax.adam(learning_rate=step_size, b1=beta_1, b2=beta_2)
        elif opt == 'adamW':
            self.tx = optax.adamw(learning_rate=step_size, b1=beta_1, b2=beta_2)
        else:
            raise ValueError('Optimizer not supported')

        # Define loss function
        if loss == 'mse':
            self.loss_func = self._mse_loss
        elif loss == 'nll':
            self.loss_func = self._cross_entropy_loss
        else:
            raise ValueError(f"Loss function {loss} not supported")
            
        self.loss_name = loss
        self.previous_features = None
        
    def create_train_state(self, params: Dict) -> TrainState:
        """Creates initial `TrainState`."""
        return TrainState.create(
            apply_fn=self.net.apply,
            params=params,
            tx=self.tx,
        )
        
    @staticmethod
    def _mse_loss(logits: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
        return jnp.mean((logits - targets) ** 2)
        
    @staticmethod
    def _cross_entropy_loss(logits: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
        return optax.softmax_cross_entropy_with_integer_labels(logits, targets).mean()
        
    @partial(jax.jit, static_argnums=(0,))
    def _compute_loss(self, params: Dict, x: jnp.ndarray, y: jnp.ndarray) -> Tuple[jnp.ndarray, Tuple]:
        """Compute loss and return features."""
        outputs, features = self.net.apply(
            {'params': params},
            x,
            mutable=['intermediates'],
            capture_intermediates=True
        )
        loss = self.loss_func(outputs, y)
        return loss, (outputs, features)

    @partial(jax.jit, static_argnums=(0,))
    def _update_step(
        self,
        state: TrainState,
        x: jnp.ndarray,
        y: jnp.ndarray
    ) -> Tuple[TrainState, jnp.ndarray, Any]:
        """Compute gradients, update parameters, and return loss and features."""
        (loss, (outputs, features)), grads = jax.value_and_grad(
            self._compute_loss, has_aux=True)(state.params, x, y)
        state = state.apply_gradients(grads=grads)
        return state, loss, outputs, features
        
    def perturb(self, params: Dict, key: jax.random.PRNGKey) -> Dict:
        """Add Gaussian noise to parameters."""
        new_params = jax.tree_map(
            lambda x: x + self.perturb_scale * jax.random.normal(key, x.shape),
            params
        )
        return new_params
    def learn(
        self,
        state: TrainState,
        x: jnp.ndarray,
        target: jnp.ndarray
    ) -> Tuple[TrainState, jnp.ndarray, Any]:
        """
        Learn using one step of gradient-descent
        Args:
            state: Current training state
            x: input
            target: desired output
        Returns:
            Updated state, loss, and optional outputs
        """
        state, loss, outputs, features = self._update_step(state, x, target)
        self.previous_features = features
        
        if self.to_perturb:
            self.rng_key, subkey = jax.random.split(self.rng_key)
            state = state.replace(params=self.perturb(state.params, subkey))
            
        if self.loss_name == 'nll':
            return state, loss, outputs
        return state, loss

    def perturb(self, params: Dict, key: jax.random.PRNGKey) -> Dict: 
        """Add Gaussian noise to parameters."""
        new_params = jax.tree_map(
            lambda x: x + self.perturb_scale * jax.random.normal(key, x.shape),
            params
        )
        return new_params

'''
import torch
import torch.nn.functional as F
from torch import optim


class Backprop(object):
    def __init__(self, net, step_size=0.001, loss='mse', opt='sgd', beta_1=0.9, beta_2=0.999, weight_decay=0.0,
                 to_perturb=False, perturb_scale=0.1, device='cpu', momentum=0):
        self.net = net
        self.to_perturb = to_perturb
        self.perturb_scale = perturb_scale
        self.device = device

        # define the optimizer
        if opt == 'sgd':
            self.opt = optim.SGD(self.net.parameters(), lr=step_size, weight_decay=weight_decay, momentum=momentum)
        elif opt == 'adam':
            self.opt = optim.Adam(self.net.parameters(), lr=step_size, betas=(beta_1, beta_2),
                                  weight_decay=weight_decay)
        elif opt == 'adamW':
            self.opt = optim.AdamW(self.net.parameters(), lr=step_size, betas=(beta_1, beta_2),
                                   weight_decay=weight_decay)

        # define the loss function
        self.loss = loss
        self.loss_func = {'nll': F.cross_entropy, 'mse': F.mse_loss}[self.loss]

        # Placeholder
        self.previous_features = None

    def learn(self, x, target):
        """
        Learn using one step of gradient-descent
        :param x: input
        :param target: desired output
        :return: loss
        """
        self.opt.zero_grad()
        output, features = self.net.predict(x=x)
        loss = self.loss_func(output, target)
        self.previous_features = features

        loss.backward()
        self.opt.step()
        if self.to_perturb:
            self.perturb()
        if self.loss == 'nll':
            return loss.detach(), output.detach()
        return loss.detach()

    def perturb(self):
        with torch.no_grad():
            for i in range(int(len(self.net.layers)/2)+1):
                self.net.layers[i * 2].bias +=\
                    torch.empty(self.net.layers[i * 2].bias.shape, device=self.device).normal_(mean=0, std=self.perturb_scale)
                self.net.layers[i * 2].weight +=\
                    torch.empty(self.net.layers[i * 2].weight.shape, device=self.device).normal_(mean=0, std=self.perturb_scale)
'''
