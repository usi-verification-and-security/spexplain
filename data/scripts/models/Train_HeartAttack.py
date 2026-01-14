import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import pandas as pd
import os
from models import FCNet
from sklearn.preprocessing import MinMaxScaler

########################
# Hyperparameters
########################
batch_size = 16
epochs = 100
num_layers = 10
num_classes = 2
hidden_size = 50
learning_rate = 1e-3
weight_decay = 1e-4
data_path = "./data/datasets/heart_attack/heart_attack_full.csv"
save_dir = "./data/models/heart_attack"
saving_file_name = f"heart_attack_{hidden_size}x{num_layers}.pth"
train_split = 0.8

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * features.size(0)
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
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)

            outputs = model(features)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * features.size(0)
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
    # Load data
    ########################
    df = pd.read_csv(data_path)

    # Separate features and target
    X = df.iloc[:, :-1].values  # all columns except last
    y = df.iloc[:, -1].values   # last column

    # Normalize features to [0, 1]
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    # Convert to PyTorch tensors (no preprocessing)
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y)

    # Create dataset
    dataset = TensorDataset(X_tensor, y_tensor)

    # Split into train and test
    train_size = int(train_split * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(
        dataset, [train_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=0)

    ########################
    # Model
    ########################
    input_dim = X.shape[1]  # 14 features

    model = FCNet(input_dim, hidden_size, num_layers, num_classes).to(device)

    ########################
    # Loss and optimizer
    ########################
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    ########################
    # Training & evaluation
    ########################
    best_test_acc = 0.0

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        if test_acc > best_test_acc:
            best_test_acc = test_acc

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
        "num_classes": num_classes
    }, save_path)

    print(f"\nModel saved to: {save_path}")
    print(f"Best test accuracy: {best_test_acc:.2f}%")
