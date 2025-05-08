from interfaces.imagesaver import ImageSaver
from interfaces.rawdataprocessor import RawDataProcessor

from pathlib import Path
from functools import cmp_to_key

class RawImagesProcessor(RawDataProcessor):

    def __init__(self, saver: ImageSaver):
        self.__saver = saver  

    
    def prepare_images(self, src_folder: str, dst_folder: str):
        """
        Process raw images in the source folder src_folder and store them in the destination folder dst_folder.
        Images reprensent pages of documents. A document can have a single page or multiple pages.
        - If a document has a single page, the page is reprensented by an image file with an arbitrary filename. The
        filename can't contain any underscore character (for instance jhz25e00.png)
        - For a multiple-pages document, there must be one file per page and they must have the following format:
        docName_pageNumber.ext, where docName is an arbitrary name that contains no underscore char, pageNumber is a counter,
        and ext is the file extension. docName must be the same for all the pages of the same document.
        Example of a document with two pages: juo75f00_1.png and juo75f00_2.png
        """

        extensions = {'.png', '.jpg', '.jpeg'}
        files = RawDataProcessor.find_files(src_folder, extensions)

        path = Path(dst_folder)
        path.mkdir(parents=True, exist_ok=True)

        previousDocName = None
        for fileDict in self.__class__.__get_filenames_dict(files):
            # multiple pages of the same document share the same docName. If we encouter a different docName, it means it's a new document:
            if previousDocName is None or previousDocName != fileDict["docName"]:
                self.__saver.begin_doc()
            
            # save image
            self.__saver.process_page_image(fileDict["filePath"], dst_folder)

            previousDocName = fileDict["docName"]


    @classmethod
    def __compare_files_dict(cls, dict_a: dict, dict_b: dict):
        if dict_a["docName"] > dict_b["docName"]:
            return 1
        if dict_a["docName"] < dict_b["docName"]:
            return -1
        if dict_a["pageNumber"] > dict_b["pageNumber"]:
            return 1
        if dict_a["pageNumber"] < dict_b["pageNumber"]:
            return -1
        return 0
    
    
    @classmethod
    def __get_filenames_dict(cls, filepaths):
        """
        Create a filenames dict that sorted by document name and page numbers.
        Handle sorting of files whose page number is not zero-padded. Regular string
        sorting would place juo75f00_10.png before juo75f00_2.png, for instance.
        """
        result = []
        for filepath in filepaths:
            filename = filepath.split('/')[-1]

            # we expect to obtain [docName, pageNumberWithExtension], for instance: ["juo75f00", "2.png"]
            # or simply [docNameWithExtension], for instance: ["jhz25e00.png"]
            docNameAndPageNumber = filename.split("_")

            if len(docNameAndPageNumber) > 2:
                raise RuntimeError(f"Unexpected file name: {filename}")

            docName = docNameAndPageNumber[0]
            if len(docNameAndPageNumber) == 1:
                pageNumber = 0
            else:
                pageNumber = int(docNameAndPageNumber[1].split(".")[0]) # ignore file extension

            result.append({
                "filePath": filepath,
                "docName": docName,
                "pageNumber": pageNumber
            })
        result.sort(key=cmp_to_key(cls.__compare_files_dict))
        return result

        
            

        
        