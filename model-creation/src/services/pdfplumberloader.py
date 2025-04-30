import pdfplumber
import cv2
import numpy as np
import sys
from interfaces.pdffileloader import PDFFileLoader

class PDFPlumberLoader(PDFFileLoader):

    def process(self, filename: str):
        pages = []
        with pdfplumber.open(filename) as pdf:
            for page in pdf.pages:
                if len(page.images) == 1:
                    image_file_object = page.images[0]
                    nparr = np.fromstring(image_file_object["stream"].get_rawdata(), np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    pages.append({"page_number": page.page_number, "rotation": page.rotation, "image": img})
                else:
                    print("This PDF file is not a valid scanned PDF")
                    sys.exit(1)
            pdf.close()
        return pages

