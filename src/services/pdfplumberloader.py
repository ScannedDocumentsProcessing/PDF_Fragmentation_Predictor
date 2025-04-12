import pdfplumber
import cv2
import numpy as np
import sys
from interfaces.pdffileloader import PDFFileLoader
from io import BytesIO

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

    def processBytes(self, pdf_data: bytes):
        pages = []
        with pdfplumber.open(BytesIO(pdf_data)) as pdf:
            for page in pdf.pages:
                if len(page.images) > 0:
                    for image_file_object in page.images:
                        try:
                            nparr = np.fromstring(image_file_object["stream"].get_rawdata(), np.uint8)
                            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                            if img is None:
                                print(f"Warning: Failed to decode image on page {page.page_number}")
                                continue  # Skip invalid images
                            pages.append({"page_number": page.page_number, "image": img})
                        except Exception as e:
                            print(f"Error decoding image on page {page.page_number}: {str(e)}")
                            continue
                        
            pdf.close()
        if not pages:
            raise ValueError("The PDF file does not contain any valid images.")
        return pages
