import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import continual_learning.backprop as backprop
from continual_learning.utils.miscellaneous import (
    compute_matrix_rank_summaries,
    nll_accuracy,
)

# matplotlib.use("QtAgg")


class SimpleNet(nn.Module):
    def __init__(self, hidden_size=512):
        super(SimpleNet, self).__init__()
        self.act_type = "relu"
        self.layers = nn.ModuleList(
            [nn.Linear(784, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 10)]
        )
        self.layers_to_log = [0, 2]  # Track the linear layers

    def forward(self, x):
        features = []
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h)
            features.append(h)
        return h, features

    def predict(self, x):
        return self.forward(x)


# Device and hyperparameters setup remains the same...
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
hidden_size = 512
num_tasks = 500
num_epochs = 1
batch_size = 64
learning_rate = 0.001
img_size = 28 * 28

# Load MNIST dataset
train_dataset = torchvision.datasets.MNIST(
    root="./data", train=True, transform=transforms.ToTensor(), download=True
)

test_dataset = torchvision.datasets.MNIST(
    root="./data", train=False, transform=transforms.ToTensor()
)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

bp_net = SimpleNet(hidden_size=hidden_size).to(device)
backprop_trainer = backprop.Backprop(
# Initialize network and trainers
    bp_net, step_size=learning_rate, loss="nll", opt="adam", device=device
)

cbp_net = SimpleNet(hidden_size=hidden_size).to(device)
cont_backprop_trainer = backprop.ContinualBackprop(
    cbp_net, step_size=learning_rate, loss="nll", opt="adam", device=device
)

# Setup metrics tracking
total_examples = num_tasks * len(train_dataset)
RANK_MEASURE_PERIOD = len(train_dataset) / 20
num_hidden_layers = 1  # SimpleNet has 1 hidden layer

# Backprop metrics
bp_ranks = torch.zeros(
    (int(total_examples / RANK_MEASURE_PERIOD), num_hidden_layers), dtype=torch.float
)
bp_effective_ranks = torch.zeros(
    (int(total_examples / RANK_MEASURE_PERIOD), num_hidden_layers), dtype=torch.float
)
bp_approximate_ranks = torch.zeros(
    (int(total_examples / RANK_MEASURE_PERIOD), num_hidden_layers), dtype=torch.float
)
bp_approximate_ranks_abs = torch.zeros(
    (int(total_examples / RANK_MEASURE_PERIOD), num_hidden_layers), dtype=torch.float
)
bp_dead_neurons = torch.zeros(
    (int(total_examples / RANK_MEASURE_PERIOD), num_hidden_layers), dtype=torch.float
)
bp_weight_mag_sum = torch.zeros(
    (total_examples, num_hidden_layers + 1), dtype=torch.float
)
bp_losses_per_task = [[] for _ in range(num_tasks)]
bp_accuracies_per_task = [[] for _ in range(num_tasks)]


# Continual Backprop metrics
cbp_ranks = torch.zeros(
    (int(total_examples / RANK_MEASURE_PERIOD), num_hidden_layers), dtype=torch.float
)
cbp_effective_ranks = torch.zeros(
    (int(total_examples / RANK_MEASURE_PERIOD), num_hidden_layers), dtype=torch.float
)
cbp_approximate_ranks = torch.zeros(
    (int(total_examples / RANK_MEASURE_PERIOD), num_hidden_layers), dtype=torch.float
)
cbp_approximate_ranks_abs = torch.zeros(
    (int(total_examples / RANK_MEASURE_PERIOD), num_hidden_layers), dtype=torch.float
)
cbp_dead_neurons = torch.zeros(
    (int(total_examples / RANK_MEASURE_PERIOD), num_hidden_layers), dtype=torch.float
)
cbp_weight_mag_sum = torch.zeros(
    (total_examples, num_hidden_layers + 1), dtype=torch.float
)
cbp_losses_per_task = [[] for _ in range(num_tasks)]
cbp_accuracies_per_task = [[] for _ in range(num_tasks)]

# Create fixed test batch for rank measurements
test_examples = next(iter(train_loader))[0][:2000].reshape(-1, img_size).to(device)


def compute_metrics(
    net,
    outputs,
    test_examples_perm,
    dead_neurons,
    ranks,
    approximate_ranks,
    approximate_ranks_abs,
    effective_ranks,
    global_iter,
    idx,
    weight_mag_sum,
    losses_per_task,
    accuracies_per_task,
    task,
    loss
):  # Fix tracking based on global or idx
    # Calculate accuracy
    _, predicted = torch.max(outputs.data, 1)
    accuracy = (predicted == labels).sum().item() / labels.size(0)

    # Store the metrics in the appropriate lists
    losses_per_task[task].append(loss.item())
    accuracies_per_task[task].append(accuracy)

    _, features = net.predict(test_examples_perm)


    # Only measure the hidden layer (skip input and output)
    for layer_idx in range(num_hidden_layers):
        # Get post-activation features (after ReLU)
        feature_matrix = features[layer_idx * 2 + 1]  # Use post-ReLU activation

        (
            ranks[idx][layer_idx],
            effective_ranks[idx][layer_idx],
            approximate_ranks[idx][layer_idx],
            approximate_ranks_abs[idx][layer_idx],
        ) = compute_matrix_rank_summaries(m=feature_matrix, use_scipy=True)

        # A neuron is considered dead if it never activates (always outputs zero) across all batch samples
        dead_neurons[idx][layer_idx] = (feature_matrix == 0).all(dim=0).sum().item()

    # Log weight magnitudes
    for idx, layer_idx in enumerate(
        net.layers_to_log
    ):  # select which layers to log instead of all layers? or just hidden?
        if isinstance(net.layers[layer_idx], nn.Linear):
            weight_mag_sum[global_iter][idx] = (
                net.layers[layer_idx].weight.data.abs().sum()
            )

    return (
        dead_neurons,
        ranks,
        approximate_ranks,
        approximate_ranks_abs,
        effective_ranks,
        weight_mag_sum,
    )


# Training Loop
global_iter = 0
for task in range(num_tasks):
    print(f"\nTraining on Task {task + 1}")
    permutation = torch.randperm(img_size)

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.reshape(-1, img_size).to(device)
            images = images[:, permutation]
            labels = labels.to(device)

            bp_loss, bp_outputs = backprop_trainer.learn(images, labels)
            cbp_loss, cbp_outputs = cont_backprop_trainer.learn(images, labels)

            global_iter += 1
            idx = int(global_iter / RANK_MEASURE_PERIOD)

            if (global_iter + 1) % RANK_MEASURE_PERIOD == 0:
                # Compute metrics for backprop network
                (
                    bp_dead_neurons,
                    bp_ranks,
                    bp_approximate_rank,
                    bp_approximate_ranks_abs,
                    bp_effective_ranks,
                    bp_weight_mag_sum,
                ) = compute_metrics(
                    net=bp_net,
                    outputs=bp_outputs,
                    test_examples_perm=test_examples[:, permutation],
                    dead_neurons=bp_dead_neurons,
                    ranks=bp_ranks,
                    approximate_ranks=bp_approximate_ranks,
                    approximate_ranks_abs=bp_approximate_ranks_abs,
                    effective_ranks=bp_effective_ranks,
                    global_iter=global_iter,
                    idx=idx,
                    weight_mag_sum=bp_weight_mag_sum,
                    losses_per_task=bp_losses_per_task,
                    accuracies_per_task=bp_accuracies_per_task,
                    task=task,
                    loss=bp_loss,

                )

                # Compute metrics for continual backprop network
                (
                    cbp_dead_neurons,
                    cbp_ranks,
                    cbp_approximate_rank,
                    cbp_approximate_ranks_abs,
                    cbp_effective_ranks,
                    cbp_weight_mag_sum,
                ) = compute_metrics(
                    net=cbp_net,
                    outputs=cbp_outputs,
                    test_examples_perm=test_examples[:, permutation],
                    dead_neurons=cbp_dead_neurons,
                    ranks=cbp_ranks,
                    approximate_ranks=cbp_approximate_ranks,
                    approximate_ranks_abs=cbp_approximate_ranks_abs,
                    effective_ranks=cbp_effective_ranks,
                    global_iter=global_iter,
                    idx=idx,
                    weight_mag_sum=cbp_weight_mag_sum,
                    losses_per_task=cbp_losses_per_task,
                    accuracies_per_task=cbp_accuracies_per_task,
                    task=task,
                    loss=cbp_loss,
                )

                print(
                    "BP: approximate rank:",
                    bp_approximate_ranks[idx],
                    ", dead neurons:",
                    bp_dead_neurons[idx],
                )
                print(
                    "CBP: approximate rank:",
                    cbp_approximate_ranks[idx],
                    ", dead neurons:",
                    cbp_dead_neurons[idx],
                )

                print(
                    f"Task {task + 1} - Backprop Epoch [{epoch + 1}/{num_epochs}], "
                    f"Step [{i + 1}/{len(train_loader)}], "
                    f"Loss: {bp_loss.item()}, Accuracy: {bp_accuracies_per_task[task]}"
                )

                print(
                    f"Task {task + 1} - Continual Backprop Epoch [{epoch + 1}/{num_epochs}], "
                    f"Step [{i + 1}/{len(train_loader)}], "
                    f"Loss: {cbp_loss.item()}, Accuracy: {cbp_accuracies_per_task[task]}"
                )

# Save metrics
# data = {
#     'ranks': ranks.cpu(),
#     'effective_ranks': effective_ranks.cpu(),
#     'approximate_ranks': approximate_ranks.cpu(),
#     'abs_approximate_ranks': approximate_ranks_abs.cpu(),
#     'dead_neurons': dead_neurons.cpu(),
#     'weight_mag_sum': weight_mag_sum.cpu()
# }


# After training, create the plots
avg_bp_losses = np.array(bp_losses_per_task).mean(axis=1)
avg_cbp_losses = np.array(cbp_losses_per_task).mean(axis=1)
avg_bp_accuracies = np.array(bp_accuracies_per_task).mean(axis=1)
avg_cbp_accuracies = np.array(cbp_accuracies_per_task).mean(axis=1)


plt.plot(avg_bp_losses, label="avg_bp_losses")
plt.plot(avg_cbp_losses, label="avg_cbp_losses")
plt.show()

plt.plot(avg_bp_accuracies, label="avg_bp_accuracies")
plt.plot(avg_cbp_accuracies, label="avg_cbp_accuracies")
plt.show()

# Plot rank metrics if desired
if len(bp_ranks) > 0:
    plt.figure(figsize=(10, 6))
    for layer in range(num_hidden_layers):
        plt.plot(ranks[:, layer], label=f"Layer {layer + 1} Rank")
        plt.plot(
            dead_neurons[:, layer],
            label=f"Layer {layer + 1} Dead Neurons",
            linestyle="--",
        )
    plt.title("Network Rank and Dead Neurons Over Time")
    plt.xlabel("Measurement Period")
    plt.ylabel("Count")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()
