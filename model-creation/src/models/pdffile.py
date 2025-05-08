from typing import List
from models.page import Page
from models.image import Image
from interfaces.imagesaver import ImageSaver
from interfaces.pdffileloader import PDFFileLoader
import json

class PDFFile:
    def __init__(self, pages):
        self.__pages: List[Page] = pages
    
    @property
    def pages(self):
        return self.__pages

    @classmethod
    def of(cls, filename: str, loader: PDFFileLoader):
        dict_pages = loader.process(filename)
        pages = []
        for dpage in dict_pages:
            img = Image(dpage['image'])
            pages.append(Page(img))
        print(f"{filename}: {len(pages)} page(s)")
        return PDFFile(pages)
    
    @classmethod
    def ofBytes(cls, pdf_data: bytes, loader: PDFFileLoader):
        dict_pages = loader.processBytes(pdf_data)  # Pass bytes to the loader
        pages = []
        for dpage in dict_pages:
            img = Image(dpage['image'])
            pages.append(Page(img))
        return PDFFile(pages)
    
    def save_images(self, saver: ImageSaver, destination: str) -> list[str]:
        """
        Save the images in the PDF and return the list of filenames
        """
        saver.begin_doc()
        images_filenames = []
        for page in self.__pages:
            images_filenames.append(page.save_image(saver, destination))
        return images_filenames
    
    def as_paired_dataset(self, transformer):
        return transformer.transform(self)

