from interfaces.datasaver import DataSaver
import os
from pathlib import Path
import json

class JSONDataSaver(DataSaver):
    def __init__(self):
        self.__incrementor = 0

    def __create_json(self, a, b, label):
        return {
            "a": f"{str(a).zfill(4)}.png",
            "b": f"{str(b).zfill(4)}.png",
            "label": label
        }
    
    def process(self, incrementors):
        pairs = []
        if len(incrementors) > 1:
            for i in range(len(incrementors) - 1):
                page_a = incrementors[i]
                page_b = incrementors[i + 1]
                label = 1

                pairs.append(self.__create_json(page_a, page_b, label))
        elif incrementors[0] > 0:
            page_a = incrementors[0] - 1
            page_b = incrementors[0]
            label = 0
            pairs.append(self.__create_json(page_a, page_b, label))

        return pairs

    def save_data(self, incrementors, destination):
        filename = f"pairs.json"
        full_path = os.path.join(destination, filename)

        path = Path(destination)
        path.mkdir(parents=True, exist_ok=True)

        with open(full_path, "w") as f:
            json.dump(incrementors, f, indent=2)
