from typing import List
from models.page import Page
from models.image import Image
from models.pdffile import PDFFile
from interfaces.imagesaver import ImageSaver
from interfaces.datasaver import DataSaver
from interfaces.pdffileloader import PDFFileLoader

import json
import os

class Folder:
    def __init__(self, pdffiles):
        self.__pdfs: List[PDFFile] = pdffiles    

    @classmethod
    def of(cls, folder_name: str, loader: PDFFileLoader):
        pdf_files_path = [os.path.join(folder_name, entry.name)  for entry in os.scandir(folder_name) if entry.is_file() and entry.name.lower().endswith(".pdf")]
        print(pdf_files_path)
        pdffiles = []
        for file in pdf_files_path:
            pdffiles.append(PDFFile.of(file, loader))
            
        return Folder(pdffiles)
    
    def save_images_and_data(self, saver: ImageSaver, destination: str, dataSaver: DataSaver):
        incrementors = []
        for pdf in self.__pdfs:
            incrementor = pdf.save_images_and_data(saver, destination, dataSaver)
            if len(incrementor) > 0:
                incrementors = incrementors + incrementor
        dataSaver.save_data(incrementors, destination)
        