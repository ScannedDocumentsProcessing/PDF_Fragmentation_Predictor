from abc import ABC, abstractmethod

class ImageSaver(ABC): 
    
    @abstractmethod
    def process(self, raw_img, destination):
        """Overrides this method to process a raw image correctly."""
        pass
