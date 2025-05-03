from abc import ABC, abstractmethod

class ImageSaver(ABC): 
    
    @abstractmethod
    def process(self, raw_img, destination) -> str:
        """
        Save the raw image and return the file name.
        Override this method to process a raw image correctly.
        """
        pass
