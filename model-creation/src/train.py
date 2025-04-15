import argparse
import json
from PIL import Image
from pathlib import Path
import yaml
from utils.seed import set_seed

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
        image_folder = datasetFolder / "images",
        transform = transform
    )

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

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader):.4f}")
    
    # Set the model to evaluation mode 
    model.eval()
    correct = 0
    total = 0

    # from https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            predicted = (outputs > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    print(f"Validation Accuracy: {correct / total * 100:.2f}%")

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

    
if __name__ == "__main__":
    main()
