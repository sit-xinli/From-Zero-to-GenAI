# Custom Model Training with PyTorch

## Overview

This project demonstrates how to train and test a simple neural network using PyTorch on the MNIST dataset (handwritten digits 0-9). The process consists of two main steps:

1. **Training**: The neural network is trained on the MNIST dataset.
2. **Testing**: The trained model is evaluated on test data to measure accuracy.

## Target Audience

This guide is intended for beginners and intermediate developers who want to understand how to build, train, and evaluate a neural network using PyTorch.

## Prerequisites

Ensure you have the following installed:

- Python (>=3.8)
- PyTorch
- Torchvision
- Matplotlib

Install dependencies using:

```bash
pip install torch torchvision matplotlib
```

---

## Step 1: Training the Model

The training process consists of the following steps:

1. **Load the MNIST dataset** and apply transformations (convert to tensor, normalize values).
2. **Define a simple feedforward neural network** with one hidden layer.
3. **Select a loss function (CrossEntropyLoss)** and an optimizer (SGD with momentum).
4. **Train the model** using mini-batches over multiple epochs.
5. **Save the trained model** for later use.

### Code: `PyTorch-Training.py`

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# 1. Define transformation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 2. Load MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

# 3. Define a simple neural network
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.layer1 = nn.Linear(28 * 28, 128)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten input image
        x = self.relu(self.layer1(x))
        x = self.layer2(x)
        return x

# 4. Initialize model, loss function, and optimizer
model = NeuralNet()
loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# 5. Train the model
num_epochs = 5
for epoch in range(num_epochs):
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# 6. Save trained model
torch.save(model.state_dict(), 'mnist_model.pth')
print("Model saved as 'mnist_model.pth'")
```

---

## Step 2: Testing the Model

After training, we load the saved model and evaluate its performance on unseen data.

### Steps:

1. Load the saved model.
2. Use the MNIST test dataset.
3. Make predictions and calculate accuracy.
4. Display sample predictions.

### Demo

<a href="https://huggingface.co/spaces/Ganesh-Kunnamkumarath/simple-PyTorch-Handwritten-Digits-Reading" target="_blank">
Demo on Hugging Face Space
</a><br /><br />

### Code: `PyTorch-Testing.py`

```python
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt

# Define the same model structure
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.layer1 = nn.Linear(28 * 28, 128)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.relu(self.layer1(x))
        x = self.layer2(x)
        return x

# Load trained model
model = NeuralNet()
model.load_state_dict(torch.load("mnist_model.pth"))
model.eval()

# Load MNIST test dataset
transform = transforms.Compose([transforms.ToTensor()])
test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=100, shuffle=False)

# Evaluate model
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.view(-1, 28 * 28)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Model Accuracy on Test Data: {accuracy:.2f}%")

# Display sample predictions
def show_predictions():
    images, labels = next(iter(test_loader))
    images_flat = images.view(-1, 28 * 28)

    with torch.no_grad():
        outputs = model(images_flat)
        _, predicted = torch.max(outputs, 1)

    fig, axes = plt.subplots(2, 5, figsize=(10, 5))
    axes = axes.flatten()

    for i in range(10):
        img = images[i].squeeze()
        axes[i].imshow(img, cmap="gray")
        axes[i].set_title(f"Pred: {predicted[i].item()}")
        axes[i].axis("off")

    plt.show()

show_predictions()
```

---

## Real-World Use Cases

1. **Digit Recognition Systems** (e.g., postal services, banking check processing)
2. **Automated Handwritten Text Recognition**
3. **AI-Assisted Teaching Tools** (for recognizing students' handwritten work)
4. **Basic Introduction to Deep Learning and Neural Networks**

---

## Summary

- **Training**: A simple neural network is trained on the MNIST dataset.
- **Testing**: The trained model is evaluated for accuracy.
- **Visualization**: Predictions are displayed using Matplotlib.

This provides a foundation for custom model training using PyTorch, which can be extended to more complex models and datasets.
