import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Define the same model structure (must match trained model)
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.layer1 = nn.Linear(28 * 28, 128)  # First hidden layer
        self.relu = nn.ReLU()  # Activation function
        self.layer2 = nn.Linear(128, 10)  # Output layer (10 classes)

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten image
        x = self.relu(self.layer1(x))  # First layer + activation
        x = self.layer2(x)  # Output layer
        return x

# Load the trained model
model = NeuralNet()
model.load_state_dict(torch.load("mnist_model.pth"))  # Load saved weights
model.eval()  # Set model to evaluation mode (no training)

# Load the MNIST test dataset
transform = transforms.Compose([transforms.ToTensor()])
test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=100, shuffle=False)

# Evaluate the model on test data
correct = 0
total = 0

with torch.no_grad():  # No gradient calculation needed
    for images, labels in test_loader:
        images = images.view(-1, 28 * 28)  # Flatten images
        outputs = model(images)  # Forward pass
        _, predicted = torch.max(outputs, 1)  # Get class with highest score
        total += labels.size(0)  # Count total samples
        correct += (predicted == labels).sum().item()  # Count correct predictions

accuracy = 100 * correct / total  # Compute accuracy percentage
print(f"Model Accuracy on Test Data: {accuracy:.2f}%")

# Display some predictions
def show_predictions():
    images, labels = next(iter(test_loader))  # Get a batch of test images
    images_flat = images.view(-1, 28 * 28)  # Flatten images for the model

    with torch.no_grad():
        outputs = model(images_flat)
        _, predicted = torch.max(outputs, 1)  # Get predicted class

    fig, axes = plt.subplots(2, 5, figsize=(10, 5))
    axes = axes.flatten()

    for i in range(10):
        img = images[i].squeeze()  # Remove extra dimensions
        axes[i].imshow(img, cmap="gray")
        axes[i].set_title(f"Pred: {predicted[i].item()}", fontsize=10)
        axes[i].axis("off")

    plt.show()

show_predictions()  # Call function to display images

"""
Steps for Model Testing
    Load the Trained Model - Load mnist_model.pth into a new model instance.
    Prepare the Test Dataset - Load the MNIST test dataset.
    Run the Model on Test Data - Pass test images through the model.
    Compute Accuracy - Compare predictions with actual labels.
    Display Some Predictions - Show a few images with predicted labels.

"""

"""

> python 15.custom_model_TESTING_PyTorch.py
> Model Accuracy on Test Data: 95.72%

Check the final accuracy of the model, the result is saved TrainedModel_TestingResult.png

"""
