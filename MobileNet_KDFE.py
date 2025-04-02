import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
import wandb
from torchsummary import summary
import os


def get_split_dataloaders(data_dir, batch_size, num_workers, shuffle, val_split, test_split):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    full_dataset = ImageFolder(root=data_dir, transform=transform)
    total_size = len(full_dataset)
    test_size = int(test_split * total_size)
    val_size = int(val_split * total_size)
    train_size = total_size - val_size - test_size
    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader


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

        scheduler.step(val_loss)

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


def save_run_details(file_path, params, final_test_acc):
    with open(file_path, "w") as f:
        f.write("Training Parameters and Results\n")
        f.write("=" * 40 + "\n")
        for key, value in params.items():
            f.write(f"{key}: {value}\n")
        f.write("=" * 40 + "\n")
        f.write(f"Final Test Accuracy: {final_test_acc:.2f}%\n")


def main():
    wandb.init(project="Emotion-recognition-KDFE-Training", name="MobileNetV2-Training")
    data_dir = "C:\\Users\\floimb\\Documents\\data\\KDFE"
    model_dir = "C:\\Users\\floimb\\Documents\\Git\\Embedded_Rmotion_Recognition\\Mobilenet"
    os.makedirs(model_dir, exist_ok=True)
    batch_size = 32
    num_workers = 4
    shuffle = True
    val_split = 0.1
    test_split = 0.1
    num_classes = 7
    learning_rate = 0.001
    epochs = 1000
    patience = 10
    pretrained = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader = get_split_dataloaders(data_dir, batch_size, num_workers, shuffle, val_split,
                                                                  test_split)
    model = models.mobilenet_v2(pretrained=pretrained)
    model.classifier[1] = torch.nn.Linear(model.last_channel, num_classes)
    model = model.to(device)
    summary(model, (3, 256, 256))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs, patience, model_dir,
                "best_model.pth")
    torch.save(model.state_dict(), os.path.join(model_dir, "last_model.pth"))
    print(f"Final model saved at {os.path.join(model_dir, 'last_model.pth')}.")
    test_acc = test_model(model, test_loader, device)
    wandb.config.update({
        "Model": "MobileNetV2",
        "Dataset": "KDFE",
        "Batch Size": batch_size,
        "Learning Rate": learning_rate,
        "Epochs": epochs,
        "Patience": patience,
        "Device": str(device),
        "Data Directory": data_dir,
        "pretrained": pretrained
    })
    save_run_details(os.path.join(model_dir, "training_results.txt"), wandb.config, test_acc)
    print(f"Training parameters saved to {os.path.join(model_dir, 'training_results.txt')}.")
    wandb.finish()


if __name__ == "__main__":
    main()
