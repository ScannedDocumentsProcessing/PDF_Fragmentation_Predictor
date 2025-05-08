from interfaces.imagesaver import ImageSaver
from PIL import Image
import os
from pathlib import Path

class PdfImageSaver(ImageSaver):

    def process_page_image(self, img, destination):
        """
        Save the raw image in the destination folder.
        """

        filename = self.image_incrementor.get_next_page_filename()
        full_path = os.path.join(destination, filename)

        path = Path(destination)
        path.mkdir(parents=True, exist_ok=True)

        print(f'saving image to {full_path}')
        img = Image.fromarray(img)
        img.save(full_path) # save image to destination
        
