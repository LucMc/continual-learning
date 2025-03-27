import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Optional, Any, Callable, Tuple, Dict


def norm_forward_sequential(inputs, mstream, varstream, afwd, eps):
    """
    Implements the forward pass of the norm op in a sequential manner,
    processing each sample in the batch one by one.
    
    Args:
        inputs: Input tensor of shape [batch_size, features]
        mstream: Running mean of shape [features]
        varstream: Running variance of shape [features]
        afwd: Forward decay factor (typically close to 1, like 0.999)
        eps: Small constant for numerical stability
    
    Returns:
        out: Normalized output
        mstream_new: Updated running mean
        varstream_new: Updated running variance
        cache: Values needed for backward pass
    """
    # Define a function to process a single sample
    def process_sample(carry, x):
        m, v = carry
        
        # Get normalization params for this sample
        center = m
        scale = jnp.sqrt(v + eps)
        
        # Normalize sample
        out = (x - center) / scale
        
        # Update statistics for next sample
        v_new = afwd * v + (afwd * (1 - afwd) * (x - m) ** 2)
        m_new = m + ((1 - afwd) * (x - m))
        
        return (m_new, v_new), (out, center, scale)
    
    # Process all samples sequentially using scan
    (mstream_final, varstream_final), (outputs, centers, scales) = jax.lax.scan(
        process_sample,
        (mstream, varstream),
        inputs
    )
    
    cache = (outputs, scales)
    return outputs, mstream_final, varstream_final, cache


def norm_backward_sequential(grad_out, ustream, vstream, abkw, cache):
    """
    Implements the backward pass of the norm op in a sequential manner,
    processing each sample in the batch one by one.
    
    Args:
        grad_out: Gradient from the next layer
        ustream: Control statistic for mean gradient
        vstream: Control statistic for variance gradient
        abkw: Backward decay factor (typically close to 1, like 0.99)
        cache: Values from the forward pass
    
    Returns:
        grad_in: Gradient for input
        ustream_new: Updated u control variable
        vstream_new: Updated v control variable
        None: Placeholder to match numpy implementation
    """
    out, scale = cache
    
    # Process each sample sequentially
    def process_grad(carry, inputs):
        u, v = carry
        grad_out_i, out_i, scale_i = inputs
        
        # Compute gradient with control mechanism
        grad = grad_out_i - (1 - abkw) * v * out_i
        v_new = v + grad * out_i
        grad = grad / scale_i
        grad_in = grad - (1 - abkw) * u
        u_new = u + grad_in
        
        return (u_new, v_new), grad_in
    
    # Prepare inputs for scan - each row contains (grad_out, out, scale) for a sample
    scan_inputs = jnp.stack([
        jnp.reshape(grad_out, (-1, grad_out.shape[-1])),
        jnp.reshape(out, (-1, out.shape[-1])),
        jnp.reshape(scale, (-1, scale.shape[-1]))
    ], axis=-1)
    
    # Transpose to get samples as the first dimension
    scan_inputs = jnp.transpose(scan_inputs, (0, 2, 1))
    
    (ustream_new, vstream_new), grad_in = jax.lax.scan(
        process_grad,
        (ustream, vstream),
        scan_inputs
    )
    
    return grad_in, ustream_new, vstream_new, (None,)


def mult_scale_forward(inputs, weight):
    """Applies learnable scale to normalized activations"""
    return inputs * weight, (inputs, weight)


def mult_scale_backward(grad_out, cache):
    """Backwards pass for scale multiplication"""
    inputs, weight = cache
    grad_in = grad_out * weight
    grad_weight = jnp.sum(grad_out * inputs, axis=0)
    return grad_in, (grad_weight,)


def add_bias_forward(inputs, bias):
    """Adds learnable bias to normalized activations"""
    return inputs + bias, None


def add_bias_backward(grad_out, cache):
    """Backwards pass for bias addition"""
    grad_in = grad_out
    grad_bias = jnp.sum(grad_out, axis=0)
    return grad_in, (grad_bias,)


def layer_scaling_forward(inputs, eps, group_size=None):
    """
    Scales inputs by the root of the second moment for the entire layer or groups.
    
    Args:
        inputs: Input tensor of shape [batch_size, features]
        eps: Small constant for numerical stability
        group_size: Size of groups for groupwise normalization (default: use all features)
    
    Returns:
        out: Scaled output
        cache: Values needed for backward pass
    """
    shape = inputs.shape
    
    if group_size is None or group_size <= 0:
        group_size = shape[1]  # Use all features
    
    # Reshape to [batch_size, num_groups, group_size]
    tmp = jnp.reshape(inputs, (shape[0], shape[1] // group_size, group_size))
    
    # Calculate second moment per group
    moment2 = jnp.mean(tmp * tmp, axis=2, keepdims=True)
    
    # Normalize
    out = tmp / jnp.sqrt(moment2 + eps)
    
    # Reshape back to original shape
    out = jnp.reshape(out, shape)
    
    cache = (out, jnp.sqrt(moment2 + eps))
    return out, cache


def layer_scaling_backward(grad_out, cache):
    """Backwards pass for layer scaling"""
    out, scale = cache
    
    # Project out the component parallel to the output
    out_flat = jnp.reshape(out, (out.shape[0], -1))
    grad_out_flat = jnp.reshape(grad_out, (grad_out.shape[0], -1))
    
    # Calculate projection
    proj = jnp.mean(grad_out_flat * out_flat, axis=1, keepdims=True) * out_flat
    
    # Reshape back
    proj = jnp.reshape(proj, grad_out.shape)
    
    # Subtract projection and divide by scale
    grad_in = (grad_out - proj) / jnp.reshape(scale, (1, 1, 1))
    
    return grad_in, (None,)


def activation_clamping_forward(inputs, clamp_val):
    """Clips activations to control instability"""
    out = jnp.clip(inputs, -clamp_val, clamp_val)
    cache = (out, clamp_val)
    return out, cache


def activation_clamping_backward(grad_out, cache):
    """Backwards pass for activation clamping"""
    out, clamp_val = cache
    
    # Zero out gradients where activations were clipped
    mask = (out > -clamp_val) & (out < clamp_val)
    grad_in = grad_out * mask
    
    return grad_in, (None,)


class Norm1d(nn.Module):
    """
    Implements the per-feature normalization using exponential moving
    averages (forward) and control process (backward) part of Online Normalization.
    
    Attributes:
        num_features: Number of features to normalize
        alpha_fwd: Decay factor for forward statistics (default: 0.999)
        alpha_bkw: Decay factor for backward control (default: 0.99)
        eps: Small constant for numerical stability (default: 1e-5)
    """
    num_features: int
    alpha_fwd: float = 0.999
    alpha_bkw: float = 0.99
    eps: float = 1e-5
    
    def setup(self):
        # Initialize statistics with proper shapes
        self.mstream = self.variable('batch_stats', 'mstream', 
                                     jnp.zeros, (self.num_features,))
        self.varstream = self.variable('batch_stats', 'varstream', 
                                       jnp.ones, (self.num_features,))
        self.ustream = self.variable('batch_stats', 'ustream', 
                                     jnp.zeros, (self.num_features,))
        self.vstream = self.variable('batch_stats', 'vstream', 
                                     jnp.zeros, (self.num_features,))
        
        # We'll need to use custom VJP for correct gradient handling with control variables
        self.norm_vjp = jax.custom_vjp(self._norm_fwd)
        self.norm_vjp.defvjp(self._norm_fwd_vjp, self._norm_bwd_vjp)
    
    def _norm_fwd(self, x):
        """Forward pass function for the VJP definition"""
        outputs, m_new, v_new, cache = norm_forward_sequential(
            x, self.mstream.value, self.varstream.value, 
            self.alpha_fwd, self.eps
        )
        return outputs
    
    def _norm_fwd_vjp(self, x):
        """Forward-of-VJP for custom gradient"""
        outputs, m_new, v_new, cache = norm_forward_sequential(
            x, self.mstream.value, self.varstream.value, 
            self.alpha_fwd, self.eps
        )
        
        # Update statistics in forward pass
        self.mstream.value = m_new
        self.varstream.value = v_new
        
        return outputs, (cache, x.shape)
    
    def _norm_bwd_vjp(self, res, g):
        """Backward-of-VJP for custom gradient"""
        (cache, x_shape) = res
        
        grad_in, u_new, v_new, _ = norm_backward_sequential(
            g, self.ustream.value, self.vstream.value, 
            self.alpha_bkw, cache
        )
        
        # Update control variables in backward pass
        self.ustream.value = u_new
        self.vstream.value = v_new
        
        return (grad_in,)
    
    def __call__(self, x, training=True):
        if not training:
            # For inference, just use the current statistics
            mu = self.mstream.value
            var = self.varstream.value
            return (x - mu) / jnp.sqrt(var + self.eps)
            
        # For training, use the full online normalization
        return self.norm_vjp(x)


class LayerScaling(nn.Module):
    """
    Scales inputs by the root of the second moment for groups.
    
    Attributes:
        group_size: Size of groups (default: -1 meaning use all channels)
        eps: Small constant for numerical stability (default: 1e-5)
    """
    group_size: int = -1
    eps: float = 1e-5
    
    @nn.compact
    def __call__(self, x):
        out, _ = layer_scaling_forward(x, self.eps, self.group_size)
        return out


class ActivationClamp(nn.Module):
    """
    Clips the output to control instability.
    
    Attributes:
        clamp_value: Value to which activations are clipped (default: 5)
    """
    clamp_value: float = 5.0
    
    @nn.compact
    def __call__(self, x):
        return jnp.clip(x, -self.clamp_value, self.clamp_value)


class OnlineNorm1d(nn.Module):
    """
    Applies Online Normalization over inputs.
    
    Attributes:
        num_features: Number of features to normalize
        alpha_fwd: Decay factor for forward statistics (default: 0.999)
        alpha_bkw: Decay factor for backward control (default: 0.99)
        eps: Small constant for numerical stability (default: 1e-5)
        affine: Whether to use learnable affine parameters (default: True)
        ecm: Error compensation mechanism ('ac' or 'ls', default: 'ac')
        ls_eps: For LayerScaling ECM (default: 1e-5)
        clamp_val: For ActivationClamp ECM (default: 5)
    """
    num_features: int
    alpha_fwd: float = 0.999
    alpha_bkw: float = 0.99
    eps: float = 1e-5
    affine: bool = True
    ecm: str = 'ac'
    ls_eps: float = 1e-5
    clamp_val: float = 5.0
    
    def setup(self):
        # Create normalization component
        self.norm = Norm1d(
            num_features=self.num_features,
            alpha_fwd=self.alpha_fwd,
            alpha_bkw=self.alpha_bkw,
            eps=self.eps
        )
        
        # Create ECM (error compensation mechanism)
        if self.ecm.lower() == 'ls':
            self.error_comp = LayerScaling(eps=self.ls_eps)
        elif self.ecm.lower() == 'ac':
            self.error_comp = ActivationClamp(clamp_value=self.clamp_val)
        else:
            self.error_comp = None
        
        # Create affine parameters if needed
        if self.affine:
            self.weight = self.param('weight', nn.initializers.ones, (self.num_features,))
            self.bias = self.param('bias', nn.initializers.zeros, (self.num_features,))
    
    def __call__(self, x, training=True):
        # Apply normalization
        out = self.norm(x, training=training)
        
        # Apply affine transform if needed
        if self.affine:
            out = out * self.weight + self.bias
        
        # Apply error compensation if configured
        if self.error_comp is not None:
            out = self.error_comp(out)
        
        return out
