import pdfplumber
import cv2
import numpy as np
import sys
from interfaces.pdffileloader import PDFFileLoader
from io import BytesIO
import zlib
import re


class PDFPlumberLoader(PDFFileLoader):

    def ascii85_pdf_decode(self, data):
        if isinstance(data, bytes):
            data = data.decode('latin1')

        data = data.strip().lstrip("<~").rstrip("~>")

        data = re.sub(r'z', '!!!!!', data)

        padding = (5 - len(data) % 5) % 5
        data += 'u' * padding

        output = bytearray()
        for i in range(0, len(data), 5):
            chunk = data[i:i+5]
            acc = 0
            for c in chunk:
                acc = acc * 85 + (ord(c) - 33)
            output += acc.to_bytes(4, 'big')

        if padding:
            output = output[:-padding]

        return bytes(output)

    def decode_stream(self, stream):
        raw = stream.get_rawdata()
        filters = stream.get_filters()

        for f in filters:
            filter_name = str(f[0]) if isinstance(f, tuple) else str(f)
            filter_name = filter_name.strip("/'")  # Clean up /'ASCII85Decode' → ASCII85Decode

            if filter_name == "ASCII85Decode":
                raw = self.ascii85_pdf_decode(raw)
            elif filter_name == "FlateDecode":
                raw = zlib.decompress(raw)
            elif filter_name == "DCTDecode":
                # JPEG
                break
            else:
                raise ValueError(f"Unsupported filter: {filter_name}")
        
        return raw

    def process(self, filename: str):
        pages = []
        with pdfplumber.open(filename) as pdf:
            for page in pdf.pages:
                if len(page.images) == 1:
                    image_file_object = page.images[0]
                    try:
                        raw_data = self.decode_stream(image_file_object["stream"])
                        nparr = np.frombuffer(raw_data, np.uint8)
                        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        if img is None:
                            raise ValueError("cv2.imdecode failed")
                        pages.append({
                            "page_number": page.page_number,
                            "rotation": page.rotation,
                            "image": img
                        })
                    except Exception as e:
                        print(f"Failed to decode image on page {page.page_number}: {e}")
                        sys.exit(1)
                else:
                    print("This PDF file is not a valid scanned PDF (multiple or no images per page)")
                    sys.exit(1)
        return pages


    def processBytes(self, pdf_data: bytes):
        return self.process(BytesIO(pdf_data))
