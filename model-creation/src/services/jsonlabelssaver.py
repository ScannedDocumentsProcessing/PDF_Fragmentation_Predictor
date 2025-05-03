from interfaces.labelssaver import LabelsSaver
import os
from pathlib import Path
import json

class JSONLabelsSaver(LabelsSaver):
    def __init__(self):
        pass


    def __create_json(self, a, b, label):
        return {
            "a": a,
            "b": b,
            "label": label
        }
    

    def __process_filenames(self, pages_filenames: list[list[str]]) -> list[dict]:
        """
        Process the pages filenames of multiples PDF files and return the pairs labels.
        Example:
        - pages_filenames = [ ["pdf1_page1.png"], ["pdf2_page1.png", "pdf2_page2.png"] ]
        - output = [ {"a": "pdf1_page1.png", "b": "pdf2_page1.png", "label": 1},
          {"a": "pdf2_page1.png", "b": "pdf2_page2.png", "label": 0} ]
        """

        pairs_labels = []
        previous_filename = None
        label = None

        for pages_of_pdf_file in pages_filenames:

            for current_filename in pages_of_pdf_file:
                # on the very first iteration, we can't save a pair label:
                if previous_filename is not None:
                    pairs_labels.append(self.__create_json(previous_filename, current_filename, label))
                # data for the next iteration:
                previous_filename = current_filename
                label = 0 # while we treat pages of the same pdf file, the next pairs will indicate that we're in the same document
            
            # we got to the end of the current pdf file, so the next pair will indicate that we started a new document
            label = 1

        return pairs_labels


    def __save_data(self, json_data, destination):
        filename = f"pairs.json"
        full_path = os.path.join(destination, filename)

        path = Path(destination)
        path.mkdir(parents=True, exist_ok=True)

        print(f"saving labels to {full_path}")
        with open(full_path, "w") as f:
            json.dump(json_data, f, indent=2)
    
    
    def process_and_save_labels(self, pages_filenames: list[list[str]], destination_folder):
        """
        Process the pages filenames of multiples PDF files and save the pairs labels in the given destination folder.
        Example:
        - pages_filenames = [ ["pdf1_page1.png"], ["pdf2_page1.png", "pdf2_page2.png"] ]
        - output file content = [ {"a": "pdf1_page1.png", "b": "pdf2_page1.png", "label": 1},
          {"a": "pdf2_page1.png", "b": "pdf2_page2.png", "label": 0} ]
        """
        pairs_data = self.__process_filenames(pages_filenames)
        self.__save_data(pairs_data, destination_folder)

