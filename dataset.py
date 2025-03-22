import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset

from PIL import Image

def get_cifar10_dataloader(batch_size, shuffle=True):
    cifar10 = load_dataset("cifar10")

    # Define a transformation to convert PIL images to tensors
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261]),
    ])

    # Select subset of data
    train_data = cifar10["train"].select(range(30000))
    test_data = cifar10["test"].select(range(10000))

    # Create PyTorch Dataset class
    class Cifar10Dataset(torch.utils.data.Dataset):
        def __init__(self, dataset, transform=None):
            self.dataset = dataset
            self.transform = transform
        
        def __len__(self):
            return len(self.dataset)
        
        def __getitem__(self, idx):
            sample = self.dataset[idx]
            image = sample["img"]
            label = sample["label"]

            if self.transform:
                image = self.transform(image.convert("RGB")) # Convert PIL image to Tensor

            return image, label

    # Create dataset objects with transformations
    train_dataset = Cifar10Dataset(train_data, transform=transform)
    test_dataset = Cifar10Dataset(test_data, transform=transform)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader