import argparse
import json
from PIL import Image
from pathlib import Path
import yaml
from utils.seed import set_seed
from matplotlib import pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim

import bentoml


Image.MAX_IMAGE_PIXELS = None

class PagePairDataset(Dataset):
    def __init__(self, json_path, image_folder, transform=None):
        with open(json_path, 'r') as f:
            self.pairs = json.load(f)

        self.image_folder = Path(image_folder)
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        img_a = Image.open(self.image_folder / pair["a"]).convert("RGB")
        img_b = Image.open(self.image_folder / pair["b"]).convert("RGB")
        label = torch.tensor(pair["label"], dtype=torch.float32)

        if self.transform:
            img_a = self.transform(img_a)
            img_b = self.transform(img_b)

        image_pair = torch.cat((img_a, img_b), dim=0)

        return image_pair, label
    

class SimpleClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 56 * 56, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.classifier(x)
        return x.squeeze(1)


def main():
    # parse command-line arguments
    parser = argparse.ArgumentParser(description="Train a classifier model on the given dataset.")
    parser.add_argument("train_dataset_folder", type=str, help="Input path to the training dataset folder.")
    parser.add_argument("model_folder", type=str, help="Ouptut path to the model folder.")
    args = parser.parse_args()
    
    datasetFolder = Path(args.train_dataset_folder)
    modelFolder = Path(args.model_folder)

    # Load parameters
    train_params = yaml.safe_load(open("params.yaml"))["train"]

    seed = train_params["seed"]
    lr = train_params["lr"]
    epochs = train_params["epochs"]

    # set seed for reproducibility
    set_seed(seed)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    dataset = PagePairDataset(
        json_path = datasetFolder / "pairs.json",
        image_folder = datasetFolder,
        transform = transform
    )

    def train_loop(model, dataloader: DataLoader, loss_fn, optimizer, epoch):
        # inspired from https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html

        # Set the model to training mode - important for batch normalization and dropout layers
        model.train()

        size = len(dataloader.dataset)
        train_loss = 0.0
        for batch, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss += loss.item() * len(images) / size
            if batch % 10 == 0:
                loss, current = loss.item(), batch * dataloader.batch_size + len(images)
                print(f"Train loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

        print(f"Train loss (epoch): {train_loss:>7f}")
        return train_loss

    def validation_loop(model, dataloader: DataLoader, loss_fn):
        # inspired from https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html

        # Set the model to evaluation mode - important for batch normalization and dropout layers
        model.eval()

        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        val_loss, correct = 0, 0

        # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
        # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_loss += loss_fn(outputs, labels).item()
                predicted = (outputs > 0.5).float()
                correct += (predicted == labels).type(torch.float).sum().item()
        
        val_loss /= num_batches
        correct /= size
        print(f"Validation Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {val_loss:>8f} \n")
        return val_loss

    # Split into train and val sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SimpleClassifier().to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_loss_history, val_loss_history = [], []
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        train_loss_history.append(train_loop(model, train_loader, criterion, optimizer, epoch))
        val_loss_history.append(validation_loop(model, val_loader, criterion))

    # save the model using BentoML to its model store
    bentoml.pytorch.save_model(
        "pdf_fragmentation_classifier",
        model,
        custom_objects = {
            # TODO create and implement the two methods
            #"preprocess": preprocess,
            #"postprocess": postprocess,
        }
    )

    # export the model from the model store to the local model folder
    modelFolder.mkdir(parents=True, exist_ok=True)
    bentoml.models.export_model(
        "pdf_fragmentation_classifier:latest",
        f"{modelFolder}/pdf_fragmentation_classifier.bentomodel",
    )

    print(f"\nModel saved at {modelFolder.absolute()}")

    # Save training history plot
    plotsFolder = (modelFolder / "plots")
    plotsFolder.mkdir(parents=True, exist_ok=True)

    fig = get_training_plot(train_loss_history, val_loss_history)
    fig.savefig(plotsFolder / "training_history.png")


def get_training_plot(train_loss_history: list, val_loss_history: list) -> plt.Figure:
    """Plot the training and validation loss"""
    epochs = range(1, len(train_loss_history) + 1)

    fig = plt.figure(figsize=(10, 4))
    plt.plot(epochs, train_loss_history, label="Training loss")
    plt.plot(epochs, val_loss_history, label="Validation loss")
    plt.xticks(epochs)
    plt.title("Training and validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    return fig

    
if __name__ == "__main__":
    main()
