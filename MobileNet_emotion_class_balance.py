import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from torchvision.datasets import ImageFolder
import wandb
from torchsummary import summary
import os
import numpy as np


def get_fer2013_dataloaders(data_dir, batch_size, num_workers, shuffle, val_split):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomRotation(degrees=10),  # ±10° rotation
        transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), scale=(0.8, 1.2)),  # ±20% scaling/shifting
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    full_dataset = ImageFolder(root=data_dir, transform=transform)
    total_size = len(full_dataset)
    val_size = int(val_split * total_size)
    train_size = total_size - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Get class counts
    class_counts = np.bincount([label for _, label in train_dataset])
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[label] for _, label in train_dataset]

    # Create a sampler for imbalanced dataset
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    # Use sampler only for the training set
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs, patience=5,
                model_dir='.', model_name='best_model.pth'):
    best_val_loss = float('inf')
    early_stop_counter = 0
    best_model_path = os.path.join(model_dir, model_name)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        train_acc = 100.0 * correct / total

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        val_acc = 100.0 * correct / total

        wandb.log({
            "Epoch": epoch + 1,
            "Train Loss": running_loss / len(train_loader),
            "Train Accuracy": train_acc,
            "Validation Loss": val_loss / len(val_loader),
            "Validation Accuracy": val_acc
        })

        print(
            f"Epoch {epoch + 1}/{epochs}, Train Loss: {running_loss / len(train_loader):.4f}, Train Acc: {train_acc:.2f}%, "
            f"Val Loss: {val_loss / len(val_loader):.4f}, Val Acc: {val_acc:.2f}%")

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved at {best_model_path}.")
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print("Early stopping triggered!")
                break


def test_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    test_acc = 100.0 * correct / total
    wandb.log({"Test Accuracy": test_acc})
    print(f"Test Accuracy: {test_acc:.2f}%")
    return test_acc


def save_run_details(file_path, params, final_test_acc, comment):
    with open(file_path, "w") as f:
        f.write("Training Parameters and Results\n")
        f.write("=" * 40 + "\n")
        for key, value in params.items():
            f.write(f"{key}: {value}\n")
        f.write("=" * 40 + "\n")
        f.write(f"Final Test Accuracy: {final_test_acc:.2f}%\n")
        f.write(comment)

def main():
    name = "MobileNetV2_Training_8bis"
    wandb.init(project="Emotion-recognition-FER2013-Training", name=name)
    train_dir = "C:\\Users\\floimb\\Documents\\data\\FER-2013\\train"
    test_dir = "C:\\Users\\floimb\\Documents\\data\\FER-2013\\test"
    model_dir = "C:\\Users\\floimb\\Documents\\Models\\Mobilenet"
    batch_size = 32
    num_workers = 4
    shuffle = True
    val_split = 0.1
    num_classes = 7
    learning_rate = 0.001
    epochs = 1000
    patience = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = get_fer2013_dataloaders(train_dir, batch_size, num_workers, shuffle, val_split)
    test_loader = get_fer2013_dataloaders(test_dir, batch_size, num_workers, shuffle, 0.0)[0]

    model = models.mobilenet_v2(pretrained=False)

    # Modify the first convolution layer to accept 1-channel input
    model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False)

    model.classifier[1] = torch.nn.Linear(model.last_channel, num_classes)
    model = model.to(device)

    summary(model, (1, 48, 48))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)
    train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs, patience, model_dir,
                name + "_best_model.pth")
    torch.save(model.state_dict(), os.path.join(model_dir, name + "_last_model.pth"))
    test_acc = test_model(model, test_loader, device)
    print("Training and testing complete.")

    # Log parameters to wandb
    wandb.config.update({
        "Model": "MobileNetV2",
        "Dataset": "FER2013",
        "Batch Size": batch_size,
        "Learning Rate": learning_rate,
        "Epochs": epochs,
        "Patience": patience,
        "Device": str(device),
        "Train Directory": train_dir,
        "Test Directory": test_dir
    })

    comment = "class balanced"

    # Save parameters to a file
    save_run_details(name + "_training_results.txt", wandb.config, test_acc, comment)

    print("Training parameters saved to training_results.txt")
    print("Training and testing complete.")
    wandb.finish()

if __name__ == "__main__":
    main()
