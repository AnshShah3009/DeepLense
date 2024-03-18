import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import os
import torchvision.transforms as transforms


transform_test = transforms.Compose(
    [
        transforms.ToPILImage(),  # Convert numpy array to PIL Image
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),  # Resize to 224x224
        transforms.ToTensor(),  # Convert PIL Image to PyTorch tensor
    ]
)

transform = transforms.Compose(
    [
        transforms.ToPILImage(),  # Convert numpy array to PIL Image
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),  # Resize to 224x224
        transforms.RandomHorizontalFlip(
            p=0.5
        ),  # Flip the image horizontally with a 50% chance
        transforms.RandomVerticalFlip(
            p=0.5
        ),  # Flip the image vertically with a 50% chance
        transforms.RandomRotation(degrees=45),  # Rotate the image by 15 degrees
        transforms.GaussianBlur(
            kernel_size=3
        ),  # Blur the image with a kernel size of 3
        transforms.ToTensor(),  # Convert PIL Image to PyTorch tensor
    ]
)


class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {self.classes[i]: i for i in range(len(self.classes))}
        self.data = self.load_data()
        print(self.classes)

    def load_data(self):
        data = []
        for class_name in self.classes:
            class_dir = os.path.join(self.root_dir, class_name)
            for file in os.listdir(class_dir):
                if file.endswith(".npy"):
                    file_path = os.path.join(class_dir, file)
                    data.append((file_path, self.class_to_idx[class_name]))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_path, label = self.data[idx]
        sample = np.load(file_path)
        sample = sample.squeeze()
        sample = 256 * (sample - sample.min()) / (sample.max() - sample.min())
        sample = sample.astype(np.uint8)
        if self.transform:
            sample = self.transform(sample)
        return sample, label
