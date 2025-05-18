from abc import ABC, abstractmethod
from services.imageincrementor import ImageIncrementor


class ImageSaver(ABC):

    def __init__(self, image_incrementor: ImageIncrementor):
        self.image_incrementor = image_incrementor

    def begin_doc(self):
        """
        Indicate that we start processing a new document.
        """
        self.image_incrementor.begin_doc()

    @abstractmethod
    def process_page_image(self, img, destination):
        """
        Save the image in the destination folder.
        """
        pass
