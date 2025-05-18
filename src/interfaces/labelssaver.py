from abc import ABC, abstractmethod


class LabelsSaver(ABC):

    @abstractmethod
    def process_and_save_labels(self, pages_filenames: list[list[str]], destination_folder):
        """
        Process the pages filenames of multiples PDF files and save the pairs labels in the given destination folder.
        """
        pass
