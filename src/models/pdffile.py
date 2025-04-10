from typing import List
from models.page import Page
from models.image import Image
from interfaces.imagesaver import ImageSaver
from interfaces.datasaver import DataSaver
from interfaces.pdffileloader import PDFFileLoader
import json

class PDFFile:
    def __init__(self, pages):
        self.__pages: List[Page] = pages
    

    @classmethod
    def of(cls, pdf_data: bytes, loader: PDFFileLoader):
        dict_pages = loader.process(pdf_data)  # Pass bytes to the loader
        pages = []
        for dpage in dict_pages:
            img = Image(dpage['image'])
            pages.append(Page(img))
        return PDFFile(pages)
    
    def save_images_and_data(self, saver: ImageSaver, destination: str, dataSaver: DataSaver):
        incrementors = []
        for page in self.__pages:
            incrementors.append(page.save_image(saver, destination))
        return dataSaver.process(incrementors)
        

