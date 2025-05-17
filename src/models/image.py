from interfaces.imagesaver import ImageSaver

class Image:
    def __init__(self, raw_data):
        self.__raw_data = raw_data
    
    @property
    def raw_data(self):
        return self.__raw_data
    
    def save(self, saver: ImageSaver, destination: str) -> str:
        """
        Save the image and return the file name.
        """
        return saver.process_page_image(self.__raw_data, destination)


