import sys
import os
import shutil
from pathlib import Path
import yaml
from models.rawpdfsprocessor import RawPdfsProcessor
from models.rawimagesprocessor import RawImagesProcessor
from services.pdfimagesaver import PdfImageSaver
from services.tobaccoimagesaver import TobaccoImageSaver
from services.pdfplumberloader import PDFPlumberLoader
from services.jsonlabelssaver import JSONLabelsSaver
from services.imageincrementor import ImageIncrementor
from utils.seed import set_seed
import random


def createListOfDocuments(folder) -> list[list[str]]:
    # TODO document this

    files = [entry.path for entry in os.scandir(folder)]
    files.sort()

    filesByDocNames = {}
    for file in files:
        filename = file.split("/")[-1]
        docName = filename.split("_")[0]
        files = filesByDocNames.get(docName, [])
        files.append(filename)
        filesByDocNames[docName] = files

    return [filesByDocNames.get(key) for key in filesByDocNames.keys()]


def splitTrainTest(files: list[str], test_size: float = None) -> dict:
    """
    Split the given files into two sets: train and test.
    - test_size: should be between 0.0 and 1.0 and represent the proportion of the dataset
    to include in the test split

    Return a dict that contains a 'train' and a 'test' key, and whose values are the list of
    files assigned to that subset.
    """

    nb_files = len(files)
    if test_size is None:
        nb_train_files = nb_files
    else:
        assert test_size < 1
        nb_train_files = round(nb_files * (1 - test_size))

    train_files = []
    test_files = []
    for idx, file in enumerate(files):
        if idx < nb_train_files:
            train_files.append(file)
        else:
            test_files.append(file)

    result = {}
    result['train'] = train_files
    if len(test_files) > 0:
        result['test'] = test_files

    print(f'found {nb_files} files. split: train = {len(train_files)}, test = {len(test_files)}')
    return result


def moveFilesToSubDirectory(parentFolder: str, datasets: dict):
    for dataset in datasets:
        path = Path(os.path.join(parentFolder, dataset))
        path.mkdir(parents=True, exist_ok=True)
        print(f'moving {len(datasets[dataset])} files to {path}')
        for file in datasets[dataset]:
            shutil.move(os.path.join(parentFolder, file), path)


def flatten(listOfList: list[list]):
    result = []
    for subList in listOfList:
        result.extend(subList)
    return result


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python src/prepare.py pdfs_source_directory images_source_directory destination_directory")
        sys.exit(1)

    pdfsSourceFolder = sys.argv[1]
    imagesSourceFolder = sys.argv[2]
    destinationFolder = sys.argv[3]

    params = yaml.safe_load(open("params.yaml"))["prepare"]
    split = params["split"]
    seed = params["seed"]

    # set seed for reproducibility
    set_seed(seed)

    incrementor = ImageIncrementor()

    # extract images from source pdf files and store them in the destination folder
    pdfsProcessor = RawPdfsProcessor(PDFPlumberLoader(), PdfImageSaver(incrementor))
    pdfsProcessor.prepare_images(pdfsSourceFolder, destinationFolder)

    # process the tobacco dataset by inverting the images colors and store them in the destination folder
    imagesProcessor = RawImagesProcessor(TobaccoImageSaver(incrementor))
    imagesProcessor.prepare_images(imagesSourceFolder, destinationFolder)

    # split to train and test
    docsAndFiles = createListOfDocuments(destinationFolder)
    random.shuffle(docsAndFiles)  # only shuffle the first level of the list (the documents), not the files (pages)
    preparedFiles = flatten(docsAndFiles)

    datasets = splitTrainTest(preparedFiles, split)
    moveFilesToSubDirectory(destinationFolder, datasets)

    # save labels
    labelsSaver = JSONLabelsSaver()
    for dataset in datasets:
        folder = os.path.join(destinationFolder, dataset)
        files = createListOfDocuments(folder)
        labelsSaver.process_and_save_labels(files, folder)
