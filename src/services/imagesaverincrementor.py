from interfaces.imagesaver import ImageSaver
from PIL import Image
import os
from pathlib import Path

class ImageSaverIncrementor(ImageSaver):
    def __init__(self):
        self.__incrementor = 0

    def process(self, raw_img, destination):
        self.__save_image(raw_img, destination)

        previousIncrementor = self.__incrementor
        self.__increment()
        return previousIncrementor
    
    def __increment(self):
        self.__incrementor = self.__incrementor + 1

    def __save_image(self, raw_img, destination):
        filename = f"{str(self.__incrementor).zfill(4)}.png"
        dir_path = os.path.join(destination, "images")
        full_path = os.path.join(dir_path, filename)

        path = Path(dir_path)
        path.mkdir(parents=True, exist_ok=True)

        img = Image.fromarray(raw_img)
        img.save(full_path)
        # save image to destination
        
