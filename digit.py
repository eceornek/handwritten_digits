import os

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from torchvision.io import read_image
from torchvision.transforms import ToTensor, Lambda
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd

class HandwrittenDigitsDataset(Dataset):
    def __init__(self, labels_file, images, transform=None, target_transform=None):
        with open(labels_file, 'r') as reader:
            self.labels = reader.readlines()
        self.images = images
        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        img_path = os.path.join(self.images, f"{index}.png")
        # image = read_image(img_path)
        image = Image.open(img_path)
        label = self.labels[index]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return (image, label)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),                 
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)

    model.train()

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    model.eval()

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


if __name__ == "__main__":

    train_data = HandwrittenDigitsDataset(
        labels_file="data/mnist_ece/train.txt",
        images="data/mnist_ece/train",
        transform = transforms.Compose([
            transforms.ToTensor(),
        ]),
        target_transform=int
    )
    
    lengths = [50000, 10000]
    train_split, valid_split = torch.utils.data.random_split(train_data, lengths)

    batch_size = 128
    epochs = 15
    learning_rate = 1e-1

    train_dataloader = DataLoader(
        train_split, batch_size=batch_size, shuffle=True,
        num_workers=2
    )
    valid_dataloader = DataLoader(
        valid_split, batch_size=batch_size,num_workers=2
    )

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model = NeuralNetwork().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer)
        test_loop(valid_dataloader, model, loss_fn)
    print("Done!")

