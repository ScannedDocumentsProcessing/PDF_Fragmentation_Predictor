from typing import List

from models.image import Image
from interfaces.imagesaver import ImageSaver

class Page:
    def __init__(self, image):
        self.__image = image

    @property
    def image(self):
        return self.__image
    
    @property
    def image(self) -> Image:
        return self.__image
    
    def save_image(self, saver: ImageSaver, destination):
        return self.__image.save(saver, destination)

