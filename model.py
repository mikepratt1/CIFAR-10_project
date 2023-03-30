"""
CIFAR-10 classification using a CNN
"""

__date__ = "2023-03-27"
__author__ = "MikePratt"

# %% --------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import numpy as np

# Configure the device
device = "cuda" if torch.cuda.is_available() else "cpu"

# %% --------------------------------------------------------------------------
# Import the datasets 
# -----------------------------------------------------------------------------
train_data = datasets.CIFAR10(
    root='.\data',
    train=True,
    transform=ToTensor(),
    download=True
)

test_data = datasets.CIFAR10(
    root='.\data',
    train=False,
    transform=ToTensor(),
    download=True
)

class_labels = train_data.classes
label_map = {idx: label for idx, label in enumerate(class_labels)}
print(class_labels, label_map)

# %% --------------------------------------------------------------------------
# Create data loaders
# -----------------------------------------------------------------------------
BATCH_SIZE = 10
learning_rate = 0.01
num_epochs = 6

train_dataloader = DataLoader(
    dataset=train_data,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_dataloader = DataLoader(
    dataset=test_data,
    batch_size=BATCH_SIZE,
    shuffle=True
)

print(f"Train Data Loader {train_dataloader}, Test Data Loader: {test_dataloader}")
print(f"Length of train DL: {len(train_dataloader)} batches of {BATCH_SIZE}")
print(f"Length of train DL: {len(test_dataloader)} batches of {BATCH_SIZE}")

# Checkout what's inside the training dataloader
train_features_batch, train_labels_batch = next(iter(train_dataloader))
train_features_batch.shape, train_labels_batch.shape

# Visualise a random sample
fig, ax = plt.subplots()
rand_num = np.random.randint(0, len(train_features_batch))
image = train_features_batch[rand_num].permute(2,1,0)
ax.imshow(image)
ax.set_title(label_map[train_labels_batch[rand_num].item()])


# %% --------------------------------------------------------------------------
# Build a CNN
# -----------------------------------------------------------------------------
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3,    # in channels defined as 3 due to RGB dimension
                               out_channels=6, # user chooses, good practice to use a power of 2
                               kernel_size=5,   # determines receptive field of the convolutional layer
                               stride=1)        # determines the amount of shift of the kernel at each step of the convolution
        self.pool1 = nn.MaxPool2d(kernel_size=2,
                                  stride=2)
        self.conv2 = nn.Conv2d(in_channels=6,  # Must be the same as out_channels in conv1
                               out_channels=16, # Here we increase from in_channels to learn more complex features
                               kernel_size=5,
                               stride=1) 
        self.fc1 = nn.Linear(in_features=16*5*5,
                             out_features=120)
        self.fc2 = nn.Linear(in_features=120,
                             out_features=10)
                                 

    def forward(self, x):
        out = self.pool1(F.relu(self.conv1(x)))
        out = self.pool1(F.relu(self.conv2(out)))
        out = out.view(-1, 16*5*5)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        return out 

# Instantiate the model
model = ConvNet().to(device)

y_pred_test = model(train_data[0][0])
print(y_pred_test)

# %% --------------------------------------------------------------------------
# Optimizer and loss function 
# -----------------------------------------------------------------------------
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# %% --------------------------------------------------------------------------
# Functionize the training and testing loops
# -----------------------------------------------------------------------------
def train_step(model, data_loader, loss_fn, optimizer, device):

    train_loss = 0
    train_acc = 0
    for batch, (X,y) in enumerate(data_loader):
        
        # Send data to the correct device
        X, y = X.to(device), y.to(device)

        # Forward pass
        y_pred = model(X)

        # Calculate the loss
        loss = loss_fn(y_pred, y)
        train_loss += loss

        # Optimizer zero grad
        optimizer.zero_grad()

        # Back propagate the loss
        loss.backward()
        
        # Optimizer step
        optimizer.step()

        if batch % 1000 == 0:
            print(f"Batch: {batch}/{len(data_loader)}")
    
    train_loss /= len(data_loader)
    print(f"Train loss: {train_loss}")

def test_step(model, data_loader, loss_fn, device):

    n_correct = 0
    n_samples = 0
    test_loss = 0
    test_acc = 0

    model.eval()
    with torch.inference_mode():
        for X, y in data_loader:
            # Send to gpu
            X, y = X.to(device), y.to(device)

            # Forward pass
            test_pred = model(X)

            _, predicted = torch.max(test_pred, 1)
            n_samples += y.size(0)
            n_correct += (predicted == y).sum().item()
            acc = 100*n_correct/n_samples

            # Calculate loss and accuracy
            test_loss += loss_fn(test_pred,y)
            test_acc += acc

        test_loss /= len(data_loader)
        test_acc /= len(data_loader)
        print(f"Test loss: {test_loss} | Test accuracy: {test_acc}")


# %% --------------------------------------------------------------------------
# Train/ test loop
# -----------------------------------------------------------------------------

for epoch in range(num_epochs):
    print(f"Epoch: {epoch}\n ----------------")
    train_step(model, 
               train_dataloader, 
               loss_fn, 
               optimizer,               
               device)
    
    test_step(model, 
              test_dataloader, 
              loss_fn,
              device)

    