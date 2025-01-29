import jax
import jax.numpy as jnp
import jax.random as random
import flax
import flax.linen as nn


class FFNN(nn.Module):
    """Simple Flax neural network"""
    
    @nn.compact
    def __call__(self, x):
        features = []
        
        # First linear layer
        x = nn.Dense(features=128, name='dense1')(x)
        features.append(x)
        
        # ReLU activation
        x = nn.relu(x)
        features.append(x)
        
        # Output layer
        x = nn.Dense(features=10, name='dense2')(x)
        features.append(x)
        
        return x, features

    def predict(self, params, x):
        return self.apply({'params': params}, x)


