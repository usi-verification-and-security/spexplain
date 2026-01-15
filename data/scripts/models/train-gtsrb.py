import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

########################
# Hyperparameters
########################
batch_size = 256          # larger batch for speed (tune down if OOM)
epochs = 25               # more epochs; GTSRB is not huge [web:27][web:33]
learning_rate = 1e-3
hidden_size = 50
num_layers = 2
num_classes = 43          # GTSRB has 43 traffic sign classes [web:27][web:33]
data_root = "./data/datasets/gtsrb"
save_dir = "./data/models/gtsrb"
saving_file_name = f"gtsrb-{num_layers}x{hidden_size}.pth"
os.makedirs(save_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

########################
# Dataset & Dataloaders
########################
# GTSRB images vary in size; resize to 32x32 to match CIFAR-like setting. [web:22][web:27]
train_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))
])

test_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))
])

# Torchvision provides a built-in GTSRB dataset wrapper. [web:22][web:23]
train_dataset = torchvision.datasets.GTSRB(
    root=data_root, split="train", download=True, transform=train_transform
)
test_dataset = torchvision.datasets.GTSRB(
    root=data_root, split="test", download=True, transform=test_transform
)

train_loader = DataLoader(train_dataset, batch_size=batch_size,
                          shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size,
                         shuffle=False, num_workers=4, pin_memory=True)

########################
# Fully connected model
########################
# After resize to 32x32, input dim = 3 * 32 * 32 = 3072. [web:22]
input_dim = 3 * 32 * 32

class FCNet(nn.Module):
    def __init__(self, input_dim, hidden_size, num_layers, num_classes):
        super().__init__()
        layers = []
        in_features = input_dim

        for _ in range(num_layers):
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(nn.ReLU(inplace=True))
            in_features = hidden_size

        layers.append(nn.Linear(hidden_size, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # flatten
        return self.net(x)

model = FCNet(input_dim, hidden_size, num_layers, num_classes).to(device)

########################
# Loss, optimizer, scheduler
########################
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

########################
# Training & evaluation
########################
def train_one_epoch(model, loader, optimizer, criterion, device, scaler):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

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
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy

for epoch in range(1, epochs + 1):
    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    scheduler.step()

    print(
        f"Epoch [{epoch}/{epochs}] "
        f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% "
        f"| Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%"
    )

########################
# Save model
########################
save_path = os.path.join(save_dir, "fc_gtsrb_5x50.pth")
torch.save({
    "model_state_dict": model.state_dict(),
    "input_dim": input_dim,
    "hidden_size": hidden_size,
    "num_layers": num_layers,
    "num_classes": num_classes,
}, save_path)

print(f"Model saved to: {save_path}")
