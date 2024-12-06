import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

import time
import math

from torch.utils.data import DataLoader

from ops.op_matmul import matmul
from ops.op_addbias import add_bias
from ops.op_relu import relu
from ops.op_sum import sum_dim
from ops.op_relu_backward import relu_backward

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_size = 784  # Example for MNIST dataset
hidden_size = 400
output_size = 10  # Number of classes in MNIST

torch.manual_seed(0)

weight_1 = torch.randn(input_size, hidden_size, device='cuda', requires_grad=True) * math.sqrt(2. / input_size)
bias_1 = torch.randn(hidden_size, device='cuda', requires_grad=True)
weight_2 = torch.randn(hidden_size, output_size, device='cuda', requires_grad=True) * math.sqrt(2. / hidden_size)
bias_2 = torch.randn(output_size, device='cuda', requires_grad=True)

optimizer = torch.optim.SGD([weight_1, bias_1, weight_2, bias_2], lr=0.001*batch_size)
# optimizer = torch.optim.SGD([weight_1, bias_1, weight_2, bias_2], lr=0.001)

criterion = nn.CrossEntropyLoss()

class MatMulFunction(Function):
    @staticmethod
    def forward(ctx, A, B):
        # TODO
        return C

    @staticmethod
    def backward(ctx, grad_output):
        # TODO
        return grad_A, grad_B
    
class AddBiasFunction(Function):
    @staticmethod
    def forward(ctx, input, bias):
        # TODO
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # TODO
        return grad_input, grad_bias
    
class ReluFunction(Function):
    @staticmethod
    def forward(ctx, input):
        # TODO
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # TODO
        return grad_input

# Forward pass
def forward_pass(x):
    # First layer
    # TODO

    # Second layer
    # TODO
    pass
    return None

import torchvision.transforms as transforms
import torchvision.datasets as datasets

# Hyperparameters
batch_size = 32
num_epochs = 10

# MNIST dataset
transform = transforms.ToTensor()
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
# Training the model
for epoch in range(num_epochs):
    print(f"Epoch: {epoch}")
    for i, (images, labels) in enumerate(train_loader):
        images = images.view(-1, 28*28).to(device)
        labels = labels.to(device)
        # Forward pass
        outputs = forward_pass(images)
        loss = criterion(outputs, labels)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.view(-1, 28*28).to(device)
        labels = labels.to(device)
        outputs = forward_pass(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f'Accuracy of the model on the 10000 test images: {100 * correct / total} %')
