import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from pathlib import Path
from model import Net

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "mnist_model.pth"
DATA_DIR = BASE_DIR / "data"

def train():
    # 1. Setup Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(str(DATA_DIR), train=True, download=True, transform=transform),
        batch_size=64, shuffle=True)

    # 2. Initialize Model, Optimizer, and Loss
    model = Net()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # 3. Fast Training (1 Epoch)
    model.train()
    print("Starting training... this can take a couple of minutes.")
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f"Batch {batch_idx}/{len(train_loader)} - Loss: {loss.item():.4f}")

    # 4. Save the weights
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Training complete! Model weights saved to {MODEL_PATH}")

if __name__ == "__main__":
    train()