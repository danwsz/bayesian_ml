import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class MNISTData:
    def __init__(self, batch_size=256, workers=4):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.view(-1))
        ])

        self.train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        self.test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

        train_size = int(0.8 * len(self.train_dataset))
        val_size = len(self.train_dataset) - train_size
        self.train_data, self.val_data = torch.utils.data.random_split(self.train_dataset, [train_size, val_size])

        self.train_loader = DataLoader(self.train_data, batch_size=batch_size, shuffle=True, num_workers=workers)
        self.val_loader = DataLoader(self.val_data, batch_size=batch_size, shuffle=False, num_workers=workers)
        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=workers)

        self.input_dim = 28 * 28
