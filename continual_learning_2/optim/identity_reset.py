import optax
from chex import dataclass

@dataclass(frozen=True)
class IdentityState:
    logs: dict 

def identity_reset(*args, **kwargs):
    """ Identity reset method """
    
    def init_fn(params, *args, **kwargs):
        return IdentityState(logs={})
    
    def update_fn(updates, state, params, features, tx_state, *args, **extra_args):
        return params, state, tx_state
    
    return optax.GradientTransformationExtraArgs(init_fn, update_fn)
