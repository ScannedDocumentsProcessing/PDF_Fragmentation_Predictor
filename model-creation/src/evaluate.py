import argparse
import json
from PIL import Image
from pathlib import Path
import os

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

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


def main():
    # parse command-line arguments
    parser = argparse.ArgumentParser(description="Evaluate a classifier model on the given dataset.")
    parser.add_argument("test_dataset_folder", type=str, help="Input path to the test dataset folder.")
    parser.add_argument("model_folder", type=str, help="Input path to the model folder.")
    args = parser.parse_args()
    
    datasetFolder = Path(args.test_dataset_folder)
    modelFolder = Path(args.model_folder)

    # TODO voir si on peut centraliser ça + PagePairDataset à un seul endroit, car là on a ce code dupliqué à trois endroits : train.py, ici et dans main.py (service)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    dataset = PagePairDataset(
        json_path = datasetFolder / "pairs.json",
        image_folder = datasetFolder,
        transform = transform
    )

    loader = DataLoader(dataset, batch_size=16, shuffle=False)

    # Import the model to the model store from a local model folder
    try:
        bentoml.models.import_model(f"{modelFolder}/pdf_fragmentation_classifier.bentomodel")
    except bentoml.exceptions.BentoMLException:
        print("Model already exists in the model store - skipping import.")

    # Load model
    modelName = "pdf_fragmentation_classifier:latest"
    # TODO discuter avec Jossef : j'ai dû ajouter weights_only=False, sinon j'obtenais une erreur. pourquoi ça marche sans cet argument dans le service ?
    model = bentoml.pytorch.load_model(modelName, weights_only=False)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # from https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    all_true_labels = []
    all_predictions = []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            predicted = (outputs > 0.5).float()

            # ".cpu()" is necessary if the code runs on GPU. See https://medium.com/@heyamit10/converting-pytorch-tensors-to-numpy-arrays-fa804b1fae1c
            all_true_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
    
    # Create folder and save performance report
    evaluation_folder = Path("evaluation")
    evaluation_folder.mkdir(exist_ok=True)

    cm = confusion_matrix(all_true_labels, all_predictions)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot().figure_.savefig(os.path.join(evaluation_folder, 'fragmentation_confusion_matrix.png'))
    
    with open(os.path.join(evaluation_folder, f"fragmentation_classification_report.json"), "w") as file:
        report = classification_report(all_true_labels, all_predictions, output_dict=True)
        json.dump(report, file)

    
if __name__ == "__main__":
    main()
