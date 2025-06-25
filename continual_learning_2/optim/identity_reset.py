import optax

def identity_reset(*args, **kwargs):
    
    def init_fn(params, *args, **kwargs):
        return {}
    
    def update_fn(updates, state, params, features, tx_state, *args, **extra_args):
        return params, state, tx_state
    
    return optax.GradientTransformationExtraArgs(init_fn, update_fn)
