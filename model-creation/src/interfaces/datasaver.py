from abc import ABC, abstractmethod

class DataSaver(ABC): 
    
    @abstractmethod
    def process(self, incrementors):
        """Overrides this method to process a raw image correctly."""
        pass
