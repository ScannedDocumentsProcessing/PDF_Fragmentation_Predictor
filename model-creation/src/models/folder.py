from typing import List
from models.pdffile import PDFFile
from interfaces.imagesaver import ImageSaver
from interfaces.labelssaver import LabelsSaver
from interfaces.pdffileloader import PDFFileLoader

import os

class Folder:
    def __init__(self, pdffiles):
        self.__pdfs: List[str] = pdffiles    

    @classmethod
    def of(cls, folder_name: str, test_size: float = None) -> dict:
        """
        Split PDF files in the given folder into two sets: train and test.
        - test_size: should be between 0.0 and 1.0 and represent the proportion of the dataset
        to include in the test split

        Return a dict that contains a 'train' and optionally a 'test' value (if there's at least
        one file in the test dataset)
        """
        pdf_files_path = [os.path.join(folder_name, entry.name) for entry in os.scandir(folder_name) if entry.is_file() and entry.name.lower().endswith(".pdf")]
        pdf_files_path.sort()

        nb_files = len(pdf_files_path)
        if test_size is None:
            nb_train_files = nb_files
        else:
            assert test_size < 1
            nb_train_files = round(nb_files * (1 - test_size))

        train_files = []
        test_files = []
        for idx, file in enumerate(pdf_files_path):
            if idx < nb_train_files:
                train_files.append(file)
            else:
                test_files.append(file)

        result = {}
        result['train'] = Folder(train_files)
        if len(test_files) > 0:
            result['test'] = Folder(test_files)
        
        print(f'found {nb_files} files. split: train = {len(train_files)}, test = {len(test_files)}')
        return result
    

    def save_data_and_labels(self, loader: PDFFileLoader, saver: ImageSaver, destination: str, labelsSaver: LabelsSaver):
        images_filenames: list[list[str]] = []
        for file in self.__pdfs:
            pdf = PDFFile.of(file, loader)
            images_filenames.append(pdf.save_images(saver, destination))
        labelsSaver.process_and_save_labels(images_filenames, destination)
        