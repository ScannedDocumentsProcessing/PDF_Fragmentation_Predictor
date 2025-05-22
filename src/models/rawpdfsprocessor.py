from models.pdffile import PDFFile
from interfaces.imagesaver import ImageSaver
from interfaces.pdffileloader import PDFFileLoader
from interfaces.rawdataprocessor import RawDataProcessor


class RawPdfsProcessor(RawDataProcessor):

    def __init__(self, loader: PDFFileLoader, saver: ImageSaver):
        self.__loader = loader
        self.__saver = saver

    def prepare_images(self, src_folder: str, dst_folder: str):
        extensions = {'.pdf'}
        files = RawDataProcessor.find_files(src_folder, extensions)
        for file in files:
            pdf = PDFFile.of(file, self.__loader)
            pdf.save_images(self.__saver, dst_folder)
