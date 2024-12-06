import torch
from torch.utils.data import DataLoader

import torch.nn as nn
import torch.optim as optim
from torch.nn import init

import time
import math

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

        torch.manual_seed(0)

        init.normal_(self.fc1.weight, mean=0, std=math.sqrt(2. / input_size))
        init.zeros_(self.fc1.bias)
        init.normal_(self.fc2.weight, mean=0, std=math.sqrt(2. / hidden_size))
        init.zeros_(self.fc2.bias)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Example usage
if __name__ == "__main__":
    input_size = 784  # Example for MNIST dataset
    hidden_size = 400
    output_size = 10  # Number of classes in MNIST

    # Hyperparameters
    batch_size = 32
    num_epochs = 10

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MLP(input_size, hidden_size, output_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001 * batch_size)
    # optimizer = optim.SGD(model.parameters(), lr=0.001)

    import torchvision.transforms as transforms
    import torchvision.datasets as datasets

    # MNIST dataset
    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    # Training the model
    for epoch in range(num_epochs):
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            # Flatten the images
            images = images.view(-1, 28*28).to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 100 == 0:
                # print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
                pass

        # Testing the model
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_loader:
                images = images.view(-1, 28*28).to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            print(f'Accuracy of the model on the 10000 test images: {100 * correct / total} %')
