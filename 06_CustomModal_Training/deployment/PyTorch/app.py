import torch
import gradio as gr
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import numpy as np

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
model.load_state_dict(torch.load("../../code/PyTorch/mnist_model.pth", map_location=torch.device('cpu')))
model.eval()  # Set model to evaluation mode

# Define image preprocessing function
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    image = transform(image)

    # Debug: Save the image to check if it's correctly processed
    from torchvision.utils import save_image
    save_image(image, "debug_image.png")

    return image.view(-1, 28 * 28)  # Flatten for model


# Define prediction function
def predict_digit(image):
    if isinstance(image, np.ndarray):  # Ensure input is a valid NumPy array
        image = Image.fromarray(image)  # Convert NumPy array to PIL Image
    image = preprocess_image(image)  # Preprocess image
    with torch.no_grad():
        outputs = model(image)  # Forward pass
        _, predicted = torch.max(outputs, 1)  # Get class with highest score
    return f"Predicted Digit: {predicted.item()}"

# Gradio interface
iface = gr.Interface(
    fn=predict_digit,
    inputs=gr.Image(type="pil", image_mode="L", height=280, width=280),  # Fixed input
    outputs="text",
    title="Digit Recognition",
    description="Draw a digit (0-9) and the model will predict it."
)

# Run the Gradio app
if __name__ == "__main__":
    iface.launch()


'''
Since the model is trained on the MNIST dataset, but it's misclassifying our handwritten digit

Possible Issues & Fixes
1. Model Architecture Issue
    Our current model has only two layers (128 hidden units), which may be too simple. MNIST models usually work better with convolutional layers (CNNs).

2. Our uploaded image might have different contrast, thickness, or noise compared to MNIST.
POssible Fix: Convert it to binary (black & white) before feeding into the model:
    - Apply a threshold to convert the image to black & white.
    - Resize the image to 28x28 pixels (MNIST image size).
    - Normalize the pixel values to match the MNIST dataset.
'''
