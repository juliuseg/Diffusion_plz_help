import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Subset, DataLoader
import os

def load_dataset(config, small_sample=False, validation=False):
    validationSize = 0.1
    if config["dataset_name"] == "MNIST":
        transform = transforms.Compose([
            transforms.Pad((2, 2, 2, 2), fill=0),  # Adds 2 pixels to each side with black padding
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),  # Normalize to [-1, 1]
        ])
        dataset = datasets.MNIST("mnist", download=True, transform=transform)

    elif config["dataset_name"] == "CIFAR10":
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # RGB Normalize to [-1, 1]
        ])
        dataset = datasets.CIFAR10("cifar10", download=True, transform=transform)

    elif config["dataset_name"] == "CELEBA":
        # Path to the Cropped CelebA binary dataset
        single_file = "cropped_celeba_bin/data_batch_1"
        # Define transformation
        transform = transforms.Compose([
            transforms.Resize((config["image_shape"][1], config["image_shape"][2])),  
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),  # Convert PIL image to tensor
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalize to [-1, 1]
        ])

        # Load CelebA binary dataset
        dataset = CustomBinaryDataset(bin_file=single_file, img_size=(128, 128), num_channels=3, transform=transform)

    else:
        raise ValueError(f"Dataset {config['dataset_name']} is not supported!")
    
    if small_sample:
        # Use only the first 100 samples for quick testing
        dataset = Subset(dataset, range(10))
    
    if validation:
        dataset_size = len(dataset)
        val_size = int(validationSize * dataset_size)
        dataset = Subset(dataset, range(dataset_size - val_size, dataset_size))
        print ("Loading validation set")
    else:
        dataset_size = len(dataset)
        train_size = int((1-validationSize) * dataset_size)
        dataset = Subset(dataset, range(0, train_size))
        print ("Loading training set")


    return dataset


class CustomBinaryDataset(torch.utils.data.Dataset):
    def __init__(self, bin_file, img_size, num_channels=3, transform=None):
        self.bin_file = bin_file  # Path to a single binary file
        self.img_size = img_size
        self.num_channels = num_channels
        self.samples = []
        self.transform = transform

        # Read the batch into memory
        file_size = os.path.getsize(self.bin_file)
        sample_size = 1 + num_channels * img_size[0] * img_size[1]  # 1 byte label + pixel data
        num_samples = file_size // sample_size

        print(f"Loading {num_samples} samples from {self.bin_file}")

        with open(self.bin_file, "rb") as f:
            for _ in range(num_samples):
                raw = f.read(sample_size)
                self.samples.append(raw)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        raw = self.samples[idx]
        label = raw[0]  # Dummy label
        pixels = torch.tensor(
            list(raw[1:]), dtype=torch.float32
        ).reshape(self.num_channels, *self.img_size) / 255.0  # Normalize to [0, 1]

        if self.transform:
            # Convert to PIL Image for compatibility with transforms
            pixels = transforms.ToPILImage()(pixels)
            pixels = self.transform(pixels)

        return pixels, label
