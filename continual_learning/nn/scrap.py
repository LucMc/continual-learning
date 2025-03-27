import continual_learning.nn.online_norm as on
from continual_learning.nn.online_norm_np import OnlineNorm1d
import jax
import jax.numpy as jnp
import numpy as np

inps = jnp.arange(0,10)
n_feat = 1
eps = 1e-5
varstream = .99
mstream = jnp.zeros(n_feat)
var = jnp.ones(n_feat)
afwd = 0.999

print( on.norm_forward(inps, mstream, var, afwd, eps) ) 

N, C = 64, 128
norm = OnlineNorm1d(C, .999, .99)
inputs = np.random.randn(N, C)  # generate fake input
output = norm(inputs)
grad_out = np.random.randn(N, C)  # generate fake gradient
grad_in = norm.backward(grad_out)
