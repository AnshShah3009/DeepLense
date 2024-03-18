from logging import warn
import torch
from torch.nn import Softmax
import numpy as np
import matplotlib.pyplot as plt
import random
import os
from tqdm import tqdm

from constants import *


def get_device(device):
    """
    Check if selected device is available and fallback on other best device (and raise warning) if otherwise

    Parameters
    ----------
    device : str
        Device selected by user

    Returns
    -------
    str
        Best device available in machine
    """
    if (device == "tpu" or device == "best") and "COLAB_TPU_ADDR" in os.environ:
        import torch_xla.core.xla_model as xm

        return xm.xla_device()
    if (device == "cuda" or device == "best") and torch.cuda.is_available():
        return "cuda"
    if (device == "mps" or device == "best") and torch.has_mps:
        return "mps"
    if device == "cpu" or device == "best":
        return "cpu"
    warn(f"Requested device {device} not found, running on CPU")
    return "cpu"


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    import torch


import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import torchvision.transforms as transforms
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader


class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        transform = transforms.Compose(
            [
                transforms.ToPILImage(),  # Convert numpy array to PIL Image
                transforms.Grayscale(num_output_channels=3),
                transforms.Resize((224, 224)),  # Resize to 224x224
                transforms.ToTensor(),  # Convert PIL Image to PyTorch tensor
            ]
        )
        if transform is not None:
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




def train(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    log_interval: int,
):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )


def test(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
):
    model.eval()
    test_loss = 0
    correct = 0
    accuracy = 0
    f1_score = 0
    precision = 0
    recall = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            accuracy += accuracy_score(target.cpu(), pred.cpu())
            f1_score += f1_score(target.cpu(), pred.cpu())
            precision += precision_score(target.cpu(), pred.cpu())
            recall += recall_score(target.cpu(), pred.cpu())

    test_loss /= len(test_loader.dataset)
    accuracy /= len(test_loader.dataset)
    f1_score /= len(test_loader.dataset)
    precision /= len(test_loader.dataset)
    recall /= len(test_loader.dataset)
    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%), F1 Score: {:.4f}, Precision: {:.4f}, Recall: {:.4f}\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
            f1_score,
            precision,
            recall,
        )
    )

def count_parameters(model: nn.Module) -> int:
    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_parameters = num_parameters / 1e6
    print(f"The model has {num_parameters:.2f}M trainable parameters.")
    return num_parameters
