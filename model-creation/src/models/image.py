from interfaces.imagesaver import ImageSaver

class Image:
    def __init__(self, raw_data):
        self.__raw_data = raw_data
    
    @property
    def raw_data(self):
        return self.__raw_data
    
    def save(self, saver: ImageSaver, destination: str):
        return saver.process(self.__raw_data, destination)


