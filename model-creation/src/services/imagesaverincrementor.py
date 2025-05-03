from interfaces.imagesaver import ImageSaver
from PIL import Image
import os
from pathlib import Path

class ImageSaverIncrementor(ImageSaver):
    def __init__(self):
        self.__incrementor = 0


    def process(self, raw_img, destination) -> str:
        """
        Save the raw image and return the file name
        """

        image_number = self.__get_and_increment()
        filename = f"{str(image_number).zfill(4)}.png"
        dir_path = os.path.join(destination, "images")
        self.__save_image(raw_img, dir_path, filename)
        return filename
    

    def __get_and_increment(self):
        current = self.__incrementor
        self.__incrementor = self.__incrementor + 1
        return current


    def __save_image(self, raw_img, dir_path, filename):
        full_path = os.path.join(dir_path, filename)
        print(f'saving image to {full_path}')

        path = Path(dir_path)
        path.mkdir(parents=True, exist_ok=True)

        img = Image.fromarray(raw_img)
        img.save(full_path) # save image to destination
        
