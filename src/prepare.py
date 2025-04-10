import sys 
import os
from models.folder import Folder
from services.imagesaverincrementor import ImageSaverIncrementor
from services.pdfplumberloader import PDFPlumberLoader
from services.jsondatasaver import JSONDataSaver

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please provide a source directory path as an argument.")
        sys.exit(1)
    elif len(sys.argv) < 3:
        print("Please provide a destination directory path as an argument.")
        sys.exit(1)
    source_folder = sys.argv[1]
    destination_folder = sys.argv[2]
    pdfLoader = PDFPlumberLoader()
    folder = Folder.of(source_folder, pdfLoader)
    saver = ImageSaverIncrementor()
    dataSaver = JSONDataSaver()
    folder.save_images_and_data(saver, destination_folder, dataSaver)
    
