from typing import List
from models.page import Page
from models.image import Image
from interfaces.pdffileloader import PDFFileLoader
import json

class PDFFile:
    def __init__(self, pages):
        self.__pages: List[Page] = pages
    
    @property
    def pages(self):
        return self.__pages

    @classmethod
    def of(cls, pdf_data: bytes, loader: PDFFileLoader):
        dict_pages = loader.process(pdf_data)  # Pass bytes to the loader
        pages = []
        for dpage in dict_pages:
            img = Image(dpage['image'])
            pages.append(Page(img))
        return PDFFile(pages)
    
    @classmethod
    def ofBytes(cls, pdf_data: bytes, loader: PDFFileLoader):
        dict_pages = loader.processBytes(pdf_data)  # Pass bytes to the loader
        pages = []
        for dpage in dict_pages:
            img = Image(dpage['image'])
            pages.append(Page(img))
        return PDFFile(pages)
    
    def as_paired_dataset(self, transformer):
        return transformer.transform(self)

