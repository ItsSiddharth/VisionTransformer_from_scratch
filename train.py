import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from model import ViTforImageClassification
from dataset import get_cifar10_dataloader

from config_json import get_config

config = get_config()

# Define training parameters
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu") # This is for apple silicon
num_epochs = config["num_epochs"]
learning_rate = config["lr"]
save_path = "best_vit_model.pth"

# Initialize model, loss function, and optimizer
model = ViTforImageClassification(config).to(device)
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

# Training and validation loops
def train(model, train_loader, val_loader, optimizer, num_epochs, save_path):
    best_val_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0
        
        loop = tqdm(train_loader, leave=True)
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            loss, outputs = model(images, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            loop.set_description(f"Epoch [{epoch+1}/{num_epochs}]")
            loop.set_postfix(loss=loss.item(), acc=100 * correct / total)
        
        train_acc = 100 * correct / total
        val_acc = validate(model, val_loader)
        
        print(f"Epoch {epoch+1}: Train Loss: {running_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"Best model saved at epoch: {epoch}")

def validate(model, val_loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs[0], 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

# Run training
train_data_loader, val_data_loader = get_cifar10_dataloader(batch_size=config["batch_size"], shuffle=True)
train(model, train_data_loader, val_data_loader, optimizer, num_epochs, save_path)