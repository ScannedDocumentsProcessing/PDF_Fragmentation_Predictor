from abc import ABC, abstractmethod
import os


class RawDataProcessor(ABC):

    @abstractmethod
    def prepare_images(self, src_folder: str, dst_folder: str):
        pass

    @classmethod
    def find_files(cls, folder_name: str, extensions: set[str]) -> list[str]:
        extensions = {ext.lower() for ext in extensions}
        files_path = []
        for entry in os.scandir(folder_name):
            if entry.is_file() and any(entry.name.lower().endswith(ext) for ext in extensions):
                files_path.append(os.path.join(folder_name, entry.name))
        files_path.sort()
        return files_path
