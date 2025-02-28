import flax.linen as nn
import jax
import jax.numpy as jnp

def name_prefix(module: nn.Module) -> str:
    return module.name + "_" if module.name else ""


def compute_plasticity_metrics(old_params, new_params):
    """Compute metrics related to neural plasticity."""
    metrics = {}

    # Calculate weight changes for each layer
    total_abs_change = 0
    total_weights = 0
    layer_metrics = {}

    for layer_name, layer_params in old_params.items():
        if isinstance(layer_params, dict) and "kernel" in layer_params:
            old_weights = layer_params["kernel"]
            new_weights = new_params[layer_name]["kernel"]

            # Calculate changes
            abs_changes = jnp.abs(new_weights - old_weights)

            # Per-layer metrics
            layer_metrics[layer_name] = {
                "mean_change": float(abs_changes.mean()),
                "max_change": float(abs_changes.max()),
                "positive_changes": float(
                    (new_weights > old_weights).sum() / old_weights.size
                ),
                "negative_changes": float(
                    (new_weights < old_weights).sum() / old_weights.size
                ),
            }

            # Update totals
            total_abs_change += jnp.sum(abs_changes)
            total_weights += old_weights.size

    # Overall metrics
    metrics["total_plasticity"] = float(total_abs_change / max(total_weights, 1))
    metrics["layer_metrics"] = layer_metrics

    return metrics


def compute_forgetting_metrics(current_losses, best_losses):
    """
    Compute metrics related to catastrophic forgetting.

    Args:
        current_losses: Dictionary mapping task IDs to current losses
        best_losses: Dictionary mapping task IDs to best achieved losses
    """
    if not current_losses or not best_losses:
        return {"avg_forgetting": 0.0, "max_forgetting": 0.0}

    # Calculate forgetting for each task
    forgetting_by_task = {}
    total_forgetting = 0.0
    max_forgetting = 0.0

    for task_id, best_loss in best_losses.items():
        if task_id in current_losses:
            forgetting = max(
                0.0, current_losses[task_id] - best_loss
            )  # Only positive forgetting
            forgetting_by_task[task_id] = forgetting
            total_forgetting += forgetting
            max_forgetting = max(max_forgetting, forgetting)

    # Average forgetting
    avg_forgetting = total_forgetting / max(len(forgetting_by_task), 1)

    return {
        "avg_forgetting": float(avg_forgetting),
        "max_forgetting": float(max_forgetting),
        "forgetting_by_task": forgetting_by_task,
    }


def stability_plasticity_tradeoff(adaptation, forgetting):
    """Calculate the stability-plasticity tradeoff metric."""
    stability = 1.0 / (1.0 + forgetting)  # Higher forgetting means lower stability
    return (
        stability * adaptation
    )  # Good tradeoff means high adaptation with high stability

