import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
import continual_learning.backprop as backprop


class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.layers = nn.ModuleList([
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        ])

    def forward(self, x):
        features = []
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h)
            features.append(h) # Store features after each layer
        return h, features  # Return output and features

    def predict(self, x):  # Predict method as required by Backprop
        return self.forward(x)



# ... (Import SimpleNet and Backprop classes from above) ...

# Device configuration (use GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
num_epochs = 5
batch_size = 64
learning_rate = 0.001

# MNIST Dataset
train_dataset = torchvision.datasets.MNIST(root='./data',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='./data',
                                          train=False,
                                          transform=transforms.ToTensor())

# Data Loaders
train_loader = DataLoader(dataset=train_dataset,
                            batch_size=batch_size,
                            shuffle=True)

test_loader = DataLoader(dataset=test_dataset,
                           batch_size=batch_size,
                           shuffle=False)

# Instantiate the Network and Backprop Trainer
net = SimpleNet().to(device)
backprop_trainer = backprop.Backprop(net, step_size=learning_rate, loss='nll', opt='adam', device=device)
# cont_backprop_trainer = backprop.ContinualBackprop(net, step_size=learning_rate, loss='nll', opt='adam', device=device)

# Training Loop
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # Move data to device
        images = images.reshape(-1, 28*28).to(device) # Flatten MNIST images
        labels = labels.to(device)

        # Train step using Backprop class
        loss, outputs = backprop_trainer.learn(images, labels) # 'learn' returns loss and outputs if loss is 'nll'

        if (i+1) % 100 == 0:
            _, predicted = torch.max(outputs.data, 1) # Get predictions from outputs
            accuracy = (predicted == labels).sum().item() / labels.size(0)
            print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}')

print("Training finished!")
