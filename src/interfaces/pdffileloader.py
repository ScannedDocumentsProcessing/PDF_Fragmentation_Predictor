from abc import ABC, abstractmethod

class PDFFileLoader(ABC): 
    
    @abstractmethod
    def process(self, filename):
        """Overrides this method to process a raw image correctly."""
        pass
