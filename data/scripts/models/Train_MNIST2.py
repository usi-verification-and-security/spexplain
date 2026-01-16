import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import os
from models import *

########################
# Hyperparameters
########################
batch_size = 128
epochs = 100
num_layers = 1
num_classes = 10
hidden_size = 200
learning_rate = 1e-3
weight_decay = 1e-3
data_root = "./data/mnist"
save_dir = "./data/models/mnist"
saving_file_name = f"mnist_{hidden_size}x{num_layers}.pth"

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


if __name__ == '__main__':
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ########################
    # Dataset & Dataloaders
    ########################
    # # CIFAR-10: 3 x 32 x 32 images  [web:2]
    # transform = transforms.Compose([
    #     transforms.ToTensor(),                          # HWC [0,255] -> CHW [0,1] [web:1]
    #     transforms.Normalize((0.5, 0.5, 0.5),           # simple normalization
    #                          (0.5, 0.5, 0.5))
    # ])
    
    
    transform = transforms.Compose([
        transforms.RandomCrop(28, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    """
    train_dataset = torchvision.datasets.MNSIST(
        root=data_root, train=True, download=True, transform=transform
    )

    test_dataset = torchvision.datasets.MNIST(
        root=data_root, train=False, download=True, transform=transform
    )
    """
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=0)

    ########################
    # Fully connected model
    ########################
    # Input dimension: 3 * 32 * 32 = 3072 for flattened CIFAR-10 images [web:2]
    input_dim = 28*28

    model = FCNet(input_dim, hidden_size, num_layers, num_classes).to(device)

    ########################
    # Loss and optimizer
    ########################
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    ########################
    # Training & evaluation
    ########################
    
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        print(
            f"Epoch [{epoch}/{epochs}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% "
            f"| Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%"
        )

    ########################
    # Save model
    ########################
    save_path = os.path.join(save_dir, saving_file_name)
    torch.save({
        "model_state_dict": model.state_dict(),
        "input_dim": input_dim,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "num_classes": num_classes,
    }, save_path)

    print(f"Model saved to: {save_path}")
