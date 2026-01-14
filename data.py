import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def get_mnist_dataloaders(batch_size=64, normalize=False):
    if normalize:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    else:
        transform = transforms.ToTensor()
    
    mnist_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    dataloader = DataLoader(mnist_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    mnist_dataset_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_dataloader = DataLoader(mnist_dataset_test, batch_size=batch_size, shuffle=False)

    return dataloader, test_dataloader
