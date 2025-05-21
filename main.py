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







