import torch  # PyTorch library for deep learning
import torch.nn as nn  # Neural network module
import torch.optim as optim  # Optimization functions (like gradient descent)
import torchvision  # Computer vision utilities
import torchvision.transforms as transforms  # Image transformations
from torch.utils.data import DataLoader  # Manages loading data in batches


# 1. Define transformation: Converts image to tensor and normalizes pixel values
transform = transforms.Compose([
    transforms.ToTensor(),  # Converts image to PyTorch tensor
    # transforms.Normalize((0.5,), (0.5,))  # Normalize pixel values to range [-1,1]
    transforms.Normalize((0.1307,), (0.3081,))

])

# 2. Load the MNIST dataset (handwritten digits 0-9)
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)

# 3. Create DataLoaders (manages how data is fed into model in batches)
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

# 4. Define a simple neural network
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.layer1 = nn.Linear(28 * 28, 128)  # First hidden layer (input size: 28x28 pixels)
        self.relu = nn.ReLU()  # Activation function: Introduces non-linearity
        self.layer2 = nn.Linear(128, 10)  # Output layer (10 classes: digits 0-9)

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten 2D image (28x28) to a 1D array (784 values)
        x = self.relu(self.layer1(x))  # Pass through first layer + activation
        x = self.layer2(x)  # Pass through output layer
        return x  # Output raw scores (logits)

# 5. Initialize model, loss function, and optimizer
model = NeuralNet()  # Create instance of our network
loss_function = nn.CrossEntropyLoss()  # Measures how well the model is guessing

# optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer for adjusting weights
# Adam optimizer's internal moment estimates can sometimes be unstable.
# Solution: Try SGD optimizer with momentum instead of Adam

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# 6. Train the model
num_epochs = 5  # Number of times model sees the entire dataset
for epoch in range(num_epochs):
    for images, labels in train_loader:  # Loop through batches
        optimizer.zero_grad()  # Clear previous gradients
        outputs = model(images)  # Forward pass: Get predictions
        loss = loss_function(outputs, labels)  # Compute loss
        loss.backward()  # Backpropagation: Compute gradients
        optimizer.step()  # Update weights using gradients

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")  # Show progress

# 7. Save the trained model for future use
torch.save(model.state_dict(), 'mnist_model.pth')
print("Model training complete and saved as 'mnist_model.pth'")

"""

python 14.custom_model_training_PyTorch.py
Epoch [1/5], Loss: 0.0889
Epoch [2/5], Loss: 0.0744
Epoch [3/5], Loss: 0.0421
Epoch [4/5], Loss: 0.0182
Epoch [5/5], Loss: 0.0168
Model training complete and saved as 'mnist_model.pth'

Awesome! 🚀 Your training is now consistent and stable.

"""
