class ImageIncrementor:
    """
    Helper for naming the dataset images. The format is: docNumber_pageNumber.png,
    where "docNumber" is a counter of the documents, and "pageNumber" is a counter of the pages in the document.
    """

    def __init__(self):
        self.__doc_number = None
        self.__page_number = None

    def begin_doc(self):
        """
        Indicate that we start processing a new document.
        """

        if self.__doc_number is None:
            self.__doc_number = 1
        else:
            self.__doc_number = self.__doc_number + 1
        self.__page_number = 1

    def get_next_page_filename(self, with_png_extension=True) -> str:
        """
        Return the filename for the next page image. By default, adds a ".png" extension.
        """

        current_page_number = self.__page_number
        self.__page_number = self.__page_number + 1
        extension = ".png" if with_png_extension else ""
        return f"{str(self.__doc_number).zfill(4)}_{str(current_page_number).zfill(3)}{extension}"
