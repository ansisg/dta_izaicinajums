import argparse
import json
import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a simple CNN on MNIST and log metrics to JSON"
    )
    parser.add_argument(
        "--epochs", type=int, default=1,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Batch size for training and testing"
    )
    parser.add_argument(
        "--metrics", type=str, default="metrics.json",
        help="Path to output metrics JSON file"
    )
    return parser.parse_args()

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return nn.functional.log_softmax(x, dim=1)

def get_data_loaders(batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    train_size = len(train_dataset)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    test_dataset = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )
    return train_loader, test_loader

def train(model, device, train_loader, optimizer):
    model.train()
    total_loss = 0.0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = nn.functional.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    return avg_loss


def evaluate(model, device, test_loader):
    model.eval()
    test_loss = 0.0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += nn.functional.nll_loss(
                output, target, reduction='sum'
            ).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)
    return test_loss, accuracy

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_loader, test_loader = get_data_loaders(
        args.batch_size
    )

    model = SimpleCNN()
    optimizer = optim.Adadelta(model.parameters(), lr=1.0)
    for epoch in range(1, args.epochs + 1):
        train_loss = train(model, device, train_loader, optimizer)
        print(f"Epoch {epoch}: training loss = {train_loss:.4f}")
    test_loss, test_accuracy = evaluate(model, device, test_loader)
    print(f"Test set: average loss = {test_loss:.4f}, accuracy = {test_accuracy:.4f}")

    # Save metrics to JSON
    metrics = {
        "loss": test_loss,
        "accuracy": test_accuracy
    }
    with open(args.metrics, 'w') as f:
        json.dump(metrics, f)
    print(f"Metrics saved to {args.metrics}")


if __name__ == "__main__":
    main()

