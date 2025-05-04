import sys 
import os
from pathlib import Path
import yaml
from models.folder import Folder
from services.imagesaverincrementor import ImageSaverIncrementor
from services.pdfplumberloader import PDFPlumberLoader
from services.jsonlabelssaver import JSONLabelsSaver

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python src/prepare.py source_directory destination_directory")
        sys.exit(1)

    source_folder = sys.argv[1]
    destination_folder = sys.argv[2]

    prepare_params = yaml.safe_load(open("params.yaml"))["prepare"]
    split = prepare_params["split"]

    pdfLoader = PDFPlumberLoader()
    labelsSaver = JSONLabelsSaver()
    folders: dict[str, Folder] = Folder.of(source_folder, split)

    for datasetName, folder in folders.items():
        datasetFolder = os.path.join(destination_folder, datasetName)
        saver = ImageSaverIncrementor()
        folder.save_data_and_labels(pdfLoader, saver, datasetFolder, labelsSaver)
    
