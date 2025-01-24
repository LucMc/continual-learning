import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
import continual_learning.backprop as backprop
from continual_learning.utils.miscellaneous import nll_accuracy, compute_matrix_rank_summaries

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.use("QtAgg")

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.act_type = 'relu'
        self.layers = nn.ModuleList([
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        ])
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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_tasks = 100
num_epochs = 2
batch_size = 64
learning_rate = 0.001
img_size = 28*28

# Load MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='./data',
                                         train=True,
                                         transform=transforms.ToTensor(),
                                         download=True)

test_dataset = torchvision.datasets.MNIST(root='./data',
                                        train=False,
                                        transform=transforms.ToTensor())

train_loader = DataLoader(dataset=train_dataset,
                         batch_size=batch_size,
                         shuffle=True)

test_loader = DataLoader(dataset=test_dataset,
                        batch_size=batch_size,
                        shuffle=False)

# Initialize network and trainers
bp_net = SimpleNet().to(device)
backprop_trainer = backprop.Backprop(bp_net, step_size=learning_rate, loss='nll', opt='adam', device=device)

cbp_net = SimpleNet().to(device)
cont_backprop_trainer = backprop.ContinualBackprop(cbp_net, step_size=learning_rate, loss='nll', opt='adam', device=device)

# Setup metrics tracking
total_examples = num_tasks * len(train_dataset)
RANK_MEASURE_PERIOD = len(train_dataset) / 20
num_hidden_layers = 1  # SimpleNet has 1 hidden layer

# Backprop metrics
bp_ranks = torch.zeros((int(total_examples/RANK_MEASURE_PERIOD), num_hidden_layers), dtype=torch.float)
bp_effective_ranks = torch.zeros((int(total_examples/RANK_MEASURE_PERIOD), num_hidden_layers), dtype=torch.float)
bp_approximate_ranks = torch.zeros((int(total_examples/RANK_MEASURE_PERIOD), num_hidden_layers), dtype=torch.float)
bp_approximate_ranks_abs = torch.zeros((int(total_examples/RANK_MEASURE_PERIOD), num_hidden_layers), dtype=torch.float)
bp_dead_neurons = torch.zeros((int(total_examples/RANK_MEASURE_PERIOD), num_hidden_layers), dtype=torch.float)
bp_weight_mag_sum = torch.zeros((total_examples, num_hidden_layers+1), dtype=torch.float)
bp_losses_per_task = [[] for _ in range(num_tasks)]
bp_accuracies_per_task = [[] for _ in range(num_tasks)]


# Continual Backprop metrics
cbp_ranks = torch.zeros((int(total_examples/RANK_MEASURE_PERIOD), num_hidden_layers), dtype=torch.float)
cbp_effective_ranks = torch.zeros((int(total_examples/RANK_MEASURE_PERIOD), num_hidden_layers), dtype=torch.float)
cbp_approximate_ranks = torch.zeros((int(total_examples/RANK_MEASURE_PERIOD), num_hidden_layers), dtype=torch.float)
cbp_approximate_ranks_abs = torch.zeros((int(total_examples/RANK_MEASURE_PERIOD), num_hidden_layers), dtype=torch.float)
cbp_dead_neurons = torch.zeros((int(total_examples/RANK_MEASURE_PERIOD), num_hidden_layers), dtype=torch.float)
cbp_weight_mag_sum = torch.zeros((total_examples, num_hidden_layers+1), dtype=torch.float)
cbp_losses_per_task = [[] for _ in range(num_tasks)]
cbp_accuracies_per_task = [[] for _ in range(num_tasks)]

# Create fixed test batch for rank measurements
test_examples = next(iter(train_loader))[0][:2000].reshape(-1, img_size).to(device)

def compute_metrics(net,
                    outputs,
                    test_examples_perm,
                    dead_neurons,
                    ranks,
                    approximate_ranks,
                    approximate_ranks_abs,
                    effective_ranks,
                    global_iter,
                    weight_mag_sum,
                    losses_per_task,
                    accuracies_per_task): # Fix tracking based on global or idx

    # Calculate accuracy
    _, predicted = torch.max(outputs.data, 1)
    accuracy = (predicted == labels).sum().item() / labels.size(0)

    _, predicted = torch.max(cbp_outputs.data, 1)
    cbp_accuracy = (predicted == labels).sum().item() / labels.size(0)
    
    losses_per_task[task].append(cbp_loss.item())
    accuracies_per_task[task].append(cbp_accuracy)

    _, features = net.predict(test_examples_perm)
    
    idx = int(global_iter / RANK_MEASURE_PERIOD)

    # Only measure the hidden layer (skip input and output)
    for layer_idx in range(num_hidden_layers):
        feature_matrix = features[layer_idx * 2]  # Skip ReLU layers

        (ranks[idx][layer_idx], 
        effective_ranks[idx][layer_idx], 
        approximate_ranks[idx][layer_idx], 
        approximate_ranks_abs[idx][layer_idx]) = compute_matrix_rank_summaries(m=feature_matrix, use_scipy=True)

        dead_neurons[new_idx][layer_idx] = (feature_matrix.abs().sum(dim=0) == 0).sum()

    # Log weight magnitudes
    for idx, layer_idx in enumerate(net.layers_to_log): # select which layers to log instead of all layers? or just hidden?
        if isinstance(net.layers[layer_idx], nn.Linear):
            weight_mag_sum[global_iter][idx] = net.layers[layer_idx].weight.data.abs().sum()

    return dead_neurons, ranks, approximate_ranks, approximate_ranks_abs, effective_ranks, weight_mag_sum

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
            
            if (global_iter+1) % RANK_MEASURE_PERIOD == 0:

                # Compute metrics
                bp_dead_neurons, bp_ranks, bp_approximate_rank, bp_approximate_ranks_abs, bp_effective_ranks, bp_weight_mag_sum = \
                        compute_metrics(bp_net,# TOD: Make a dataclass to store these
                                    bp_outputs,
                                    test_examples[:, permutation],
                                    bp_dead_neurons,
                                    bp_ranks,
                                    bp_approximate_ranks,
                                    bp_approximate_ranks_abs,
                                    bp_effective_ranks,
                                    global_iter,
                                    bp_weight_mag_sum,
                                    bp_losses_per_task,
                                    bp_accuracies_per_task)

                cbp_dead_neurons, cbp_ranks, cbp_approximate_rank, cbp_approximate_ranks_abs, cbp_effective_ranks, cbp_weight_mag_sum = \
                        compute_metrics(cbp_net,
                                        cbp_outputs,
                                        test_examples[:, permutation],
                                        cbp_dead_neurons,
                                        cbp_ranks,
                                        cbp_approximate_ranks,
                                        cbp_approximate_ranks_abs,
                                        cbp_effective_ranks,
                                        global_iter,
                                        cbp_weight_mag_sum,
                                        cbp_losses_per_task,
                                        cbp_accuracies_per_task)
                
                print('BP: approximate rank:', bp_approximate_ranks[idx], ', dead neurons:', bp_dead_neurons[new_idx])
                print('CBP: approximate rank:', cbp_approximate_ranks[idx], ', dead neurons:', cbp_dead_neurons[new_idx])

                print(f'Task {task+1} - Backprop Epoch [{epoch+1}/{num_epochs}], '
                      f'Step [{i+1}/{len(train_loader)}], '
                      f'Loss: {bp_loss.item():.4f}, Accuracy: {bp_accuracy:.4f}')

                print(f'Task {task+1} - Continual Backprop Epoch [{epoch+1}/{num_epochs}], '
                      f'Step [{i+1}/{len(train_loader)}], '
                      f'Loss: {cbp_loss.item():.4f}, Accuracy: {cbp_accuracy:.4f}')

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


plt.plot(avg_bp_losses, label='avg_bp_losses')
plt.plot(avg_cbp_losses, label='avg_cbp_losses')
plt.show()

plt.plot(avg_bp_accuracies, label='avg_bp_accuracies')
plt.plot(avg_cbp_accuracies, label='avg_cbp_accuracies')
plt.show()

# Plot rank metrics if desired
if len(bp_ranks) > 0:
    plt.figure(figsize=(10, 6))
    for layer in range(num_hidden_layers):
        plt.plot(ranks[:, layer], label=f'Layer {layer+1} Rank')
        plt.plot(dead_neurons[:, layer], label=f'Layer {layer+1} Dead Neurons', linestyle='--')
    plt.title('Network Rank and Dead Neurons Over Time')
    plt.xlabel('Measurement Period')
    plt.ylabel('Count')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()
