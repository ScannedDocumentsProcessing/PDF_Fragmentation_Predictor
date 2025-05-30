import sys 
from models.pdffile import PDFFile
from services.pdfplumberloader import PDFPlumberLoader
from pathlib import Path

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please provide a PDF file path as an argument.")
        sys.exit(1)
    filename = sys.argv[1]
    file_bytes = Path(filename).read_bytes()

    pdfLoader = PDFPlumberLoader()
    pdf = PDFFile.ofBytes(file_bytes, pdfLoader)
    print(pdf)