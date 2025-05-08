from interfaces.imagesaver import ImageSaver
import PIL
from PIL import Image
import os
from pathlib import Path

class TobaccoImageSaver(ImageSaver):
        
    def process_page_image(self, img, destination):
        """
        Invert the colors of the source image and save the result in the destination folder.
        Image is the filename of an already existing image of the Tobacco dataset.
        """

        extension = "." + img.split(".")[-1]
        filename = self.image_incrementor.get_next_page_filename(False) + extension
        full_path = os.path.join(destination, filename)

        path = Path(destination)
        path.mkdir(parents=True, exist_ok=True)

        print(f'inverting image {img} and saving to {full_path}')
        image = Image.open(img)
        inverted_image = PIL.ImageOps.invert(image)
        inverted_image.save(full_path)
        
