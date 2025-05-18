from models.image import Image
from interfaces.imagesaver import ImageSaver


class Page:
    def __init__(self, image):
        self.__image: Image = image

    @property
    def image(self) -> Image:
        return self.__image

    def save_image(self, saver: ImageSaver, destination) -> str:
        """
        Save the image and return the file name.
        """
        return self.__image.save(saver, destination)
