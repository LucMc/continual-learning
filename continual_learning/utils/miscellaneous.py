"""Reimplement these metrics in JAX, remove torch"""
import math
import itertools
import numpy as np
from torch import nn
from tqdm import tqdm
from math import sqrt
from torch.nn import Conv2d, Linear
import torch
import jax
import jax.numpy as jnp
from flax.traverse_util import ModelParamTraversal
from functools import partial

from scipy.linalg import svd


def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    """Alternative minibatching for mem efficiency"""
    assert inputs.shape[0] == targets.shape[0]
    if shuffle:
        indices = np.arange(inputs.shape[0])
        np.random.shuffle(indices)
    for start_idx in tqdm(range(0, inputs.shape[0], batchsize)):
        if shuffle:
            excerpt = indices[start_idx: start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


@partial(jax.jit, static_argnames=["learning_rate", "label"])
def compute_plasticity_metrics(old_params, new_params, learning_rate, label="net"):
    """Compute neural plasticity metrics, normalised by learning rate"""
    metrics = {}

    # Calculate weight changes for each layer
    total_abs_change = 0
    total_weights = 0
    kernel_traversal = ModelParamTraversal(lambda path_str, _: path_str.endswith('/kernel'))
    
    # Consider scan to reduce compile time for large networks
    for old_weights, new_weights in zip(kernel_traversal.iterate(old_params), kernel_traversal.iterate(new_params)):
        
        # Calculate changes
        abs_changes = jnp.abs(new_weights - old_weights)

        # Update totals
        total_abs_change += jnp.sum(abs_changes)
        total_weights += old_weights.size

    # Overall metrics
    normalised_change = total_abs_change / learning_rate

    return {f"{label}_plasticity": normalised_change / total_weights }

"""
    # def is_kernel(path):
    #     return path[-1].key == 'kernel'
    #
    # def per_layer_plasticity(path, layer_old, layer_new):
    #     abs_changes = jnp.abs(new_weights - old_weights)
    #
    # for layer_name, layer_params in old_params.items():

    # plasticity = jax.tree.map(lambda p1, p2: jnp.abs(p1 - p2).sum(), old_params, new_params)
    # total_plasticity = jax.tree.reduce(lambda acc, x: acc+x, plasticity)
    # total_weights = jax.tree.reduce(lambda acc, x: acc+x.size, old_params, initializer=0)

    # print("plasticity:\n", plasticity)
    # print("total_plasticity:\n", total_plasticity)
    # print("total_weights:\n", total_weights)


    ### OLD METHOD FOR CONTEXT

    def per_layer_plasticity(path, layer_old, layer_new):

        def kernel():
            abs_changes = jnp.abs(new_weights - old_weights)
            return {"changes": layer}

        return jax.lax.cond(path[-1]==jax.tree_util.DictKey("kernel"), kernel, lambda _: jnp.nan)
    
    plasticity = jax.tree.map_with_path(per_layer_plasticity, old_params, new_params)
    print("plasticity:\n", plasticity)
"""

def compute_matrix_rank_summaries(m: torch.Tensor, prop=0.99, use_scipy=False):
    """
    Computes the rank, effective rank, and approximate rank of a matrix
    Refer to the corresponding functions for their definitions
    :param m: (float np array) a rectangular matrix
    :param prop: (float) proportion used for computing the approximate rank
    :param use_scipy: (bool) indicates whether to compute the singular values in the cpu, only matters when using
                                  a gpu
    :return: (torch int32) rank, (torch float32) effective rank, (torch int32) approximate rank
    """
    if use_scipy:
        np_m = m.detach().numpy()
        sv = torch.tensor(svd(np_m, compute_uv=False, lapack_driver="gesvd"), device=m.device)
    else:
        sv = torch.linalg.svdvals(m)    # for large matrices, svdvals may fail to converge in gpu, but not cpu
    rank = torch.count_nonzero(sv).to(torch.int32)
    effective_rank = compute_effective_rank(sv)
    approximate_rank = compute_approximate_rank(sv, prop=prop)
    approximate_rank_abs = compute_abs_approximate_rank(sv, prop=prop)
    return rank, effective_rank, approximate_rank, approximate_rank_abs


def compute_effective_rank(sv: torch.Tensor):
    """
    Computes the effective rank as defined in this paper: https://ieeexplore.ieee.org/document/7098875/
    When computing the shannon entropy, 0 * log 0 is defined as 0
    :param sv: (float torch Tensor) an array of singular values
    :return: (float torch Tensor) the effective rank
    """
    norm_sv = sv / torch.sum(torch.abs(sv))
    entropy = torch.tensor(0.0, dtype=torch.float32, device=sv.device)
    for p in norm_sv:
        if p > 0.0:
            entropy -= p * torch.log(p)

    effective_rank = torch.tensor(np.e) ** entropy
    return effective_rank.to(torch.float32)


def compute_approximate_rank(sv: torch.Tensor, prop=0.99):
    """
    Computes the approximate rank as defined in this paper: https://arxiv.org/pdf/1909.12255.pdf
    :param sv: (float np array) an array of singular values
    :param prop: (float) proportion of the variance captured by the approximate rank
    :return: (torch int 32) approximate rank
    """
    sqrd_sv = sv ** 2
    normed_sqrd_sv = torch.flip(torch.sort(sqrd_sv / torch.sum(sqrd_sv))[0], dims=(0,))   # descending order
    cumulative_ns_sv_sum = 0.0
    approximate_rank = 0
    while cumulative_ns_sv_sum < prop:
        cumulative_ns_sv_sum += normed_sqrd_sv[approximate_rank]
        approximate_rank += 1
    return torch.tensor(approximate_rank, dtype=torch.int32)


def compute_abs_approximate_rank(sv: torch.Tensor, prop=0.99):
    """
    Computes the approximate rank as defined in this paper, just that we won't be squaring the singular values
    https://arxiv.org/pdf/1909.12255.pdf
    :param sv: (float np array) an array of singular values
    :param prop: (float) proportion of the variance captured by the approximate rank
    :return: (torch int 32) approximate rank
    """
    sqrd_sv = sv
    normed_sqrd_sv = torch.flip(torch.sort(sqrd_sv / torch.sum(sqrd_sv))[0], dims=(0,))   # descending order
    cumulative_ns_sv_sum = 0.0
    approximate_rank = 0
    while cumulative_ns_sv_sum < prop:
        cumulative_ns_sv_sum += normed_sqrd_sv[approximate_rank]
        approximate_rank += 1
    return torch.tensor(approximate_rank, dtype=torch.int32)

