import torch
import torch.nn as nn #neural networks library
import torch.optim as optim #helps train model
from torchvision import datasets, transforms #MNIST dataset
from torch.utils.data import DataLoader #loads data in small batches

device = torch.device("cpu")

# prepare the data, convert image pixels into tensors and 
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Downloads the MNIST dataset
# 60,000 training images, 10,000 test images of handwritten digits
train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Break data up, shuffle to avoid bias
train_data = DataLoader(train_data, batch_size = 64, shuffle = True)
test_data = DataLoader(train_data, batch_size = 64, shuffle = False)

# Neural Network Time
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(), # Turns 28x28 images into 784 length vectors
            nn.Linear(28*28, 128), # First layer: 784 -> 128 neurons
            nn.ReLU(),  # Activation function
            nn.Linear(128, 64), # 128 -> 64
            nn.ReLU(),
            nn.Linear(64, 10) # 64 -> 10 (for digits 0-9)
        )

        def forward(self, x):
            return self.model(x)
        
# Training Tools
model = SimpleNN().to(device) # brain in training
criterion = nn.CrossEntropyLoss() # measures how bad the model is doing
optimizer = optim.Adam(model.parameters(), lr=0.001) # updates the model to do better next time

 # Train the model
epochs = 5
for epoch in range(epochs):
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()             # Clear previous gradients
        outputs = model(images)           # Make predictions
        loss = criterion(outputs, labels) # Compare predictions to actual labels
        loss.backward()                   # Calculate gradients
        optimizer.step()                  # Update model weights

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")

# Test the model
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

print(f"Test Accuracy: {100 * correct / total:.2f}%")







