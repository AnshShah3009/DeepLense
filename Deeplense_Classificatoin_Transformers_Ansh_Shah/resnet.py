# %%
from vit_pytorch.cct import CCT
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

# %%
model = CCT(
    img_size=(224, 224),
    embedding_dim=384,
    n_conv_layers=3,
    kernel_size=7,
    stride=2,
    padding=3,
    pooling_kernel_size=5,
    pooling_stride=2,
    pooling_padding=1,
    num_layers=14,
    num_heads=6,
    mlp_ratio=3.0,
    num_classes=3,
    n_input_channels=3,
    positional_embedding="sine",  # ['sine', 'learnable', 'none']
)

model = model.to(device)

# %%
from utils import *

train_dataset = CustomDataset(
    "/scratch/Ansh/DeepLense/specific_taskV/dataset/train", transform=transform
)
test_dataset = CustomDataset(
    "/scratch/Ansh/DeepLense/specific_taskV/dataset/val", transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=300, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=300, shuffle=False, num_workers=4)

# %%
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
criterion = torch.nn.CrossEntropyLoss()

file = open("CvT.txt", "a")

for epoch in range(3000):
    accuracy = 0
    epoch_loss = 0
    model.train()
    for data, label in tqdm(train_loader):
        output = model(data.to(device))
        loss = criterion(output, label.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        accuracy += (output.argmax(1) == label.to(device)).sum().item() / len(label)

    model.eval()
    epoch_train = 0
    with torch.no_grad():
        for data, label in tqdm(test_loader):
            output = model(data.to(device))
            loss = criterion(output, label.to(device))
            epoch_train += loss.item()

    print(
        f"Epoch {epoch} Train Loss: {epoch_loss/len(train_loader)} Test Loss: {epoch_train/len(test_loader)}, Accuracy: {accuracy/len(train_loader)}"
    )

    file.write(
        f"Epoch {epoch} Train Loss: {epoch_loss/len(train_loader)} Test Loss: {epoch_train/len(test_loader)}, Accuracy: {accuracy/len(train_loader)}"
    )


# %%
# resnet = torch.hub.load("pytorch/vision:v0.9.0", "resnet18", pretrained=False)

# resnet.fc = torch.nn.Linear(512, 3)
# resnet = resnet.to(device)

# # %%
# for param in resnet.parameters():
#     param.requires_grad = True

# optimizer = torch.optim.Adam(resnet.parameters(), lr=1e-3)
# criterion = torch.nn.CrossEntropyLoss()

# file = open("resnet18.txt", "w")

# for epoch in range(3000):
#     epoch_loss = 0
#     accuracy = 0
#     resnet.train()
#     for data, label in tqdm(train_loader):
#         output = resnet(data.to(device))
#         loss = criterion(output, label.to(device))
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         epoch_loss += loss.item()
#         accuracy += (output.argmax(1) == label.to(device)).sum().item() / len(label)

#     resnet.eval()
#     epoch_train = 0
#     with torch.no_grad():
#         for data, label in tqdm(test_loader):
#             output = resnet(data.to(device))
#             loss = criterion(output, label.to(device))
#             epoch_train += loss.item()

#     print(
#         f"Epoch {epoch} Train Loss: {epoch_loss/len(train_loader)} Test Loss: {epoch_train/len(test_loader), accuracy/len(train_loader)}"
#     )

#     file.write(
#         f"Epoch {epoch} Train Loss: {epoch_loss/len(train_loader)} Test Loss: {epoch_train/len(test_loader), accuracy/len(train_loader)}"
#     )

# # %%
