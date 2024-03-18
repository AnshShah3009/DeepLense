# %%
import timm
import torch
from utils import *
from data import CustomDataset, transform
from torch.utils.data import DataLoader
from tqdm import tqdm

# %%
train_dataset = CustomDataset(
    "/scratch/Ansh/DeepLense/specific_taskV/dataset/train", transform=transform
)
test_dataset = CustomDataset(
    "/scratch/Ansh/DeepLense/specific_taskV/dataset/val",
    transform=transform,
)

batch_size = 40

train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
)
test_loader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, num_workers=4
)

# %%
from timm import create_model

model_name = "efficientnet_b0"
pretrained = False

model = create_model(model_name, pretrained=pretrained)

# %%
import torchinfo

torchinfo.summary(model)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# %%
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

n_epochs = 1000

train_losses = []
test_losses = []
test_accuracies = []

with open("efficientnet.txt", "w") as f:
    f.write("")

for epoch in range(n_epochs):
    model.train()
    for i, (images, labels) in tqdm(enumerate(train_loader)):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print(
                f"Epoch [{epoch}/{n_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}"
            )
            train_losses.append(loss.item())
            with open("efficientnet.txt", "a") as f:
                f.write(
                    f"Epoch [{epoch}/{n_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}\n"
                )

    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in tqdm(test_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        test_losses.append(loss.item())
        test_accuracies.append(100 * correct / total)
        print(
            f"Test Accuracy of the model on the test images: {100 * correct / total}%"
        )
        with open("efficientnet.txt", "a") as f:
            f.write(
                f"Epoch [{epoch}/{n_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}\n"
            )
            f.write(
                f"Test Accuracy of the model on the test images: {100 * correct / total}%\n"
            )


torch.save(model.state_dict(), "efficient.pth")

# %%
