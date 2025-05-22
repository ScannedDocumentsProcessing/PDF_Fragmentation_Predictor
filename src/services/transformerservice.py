from PIL import Image as PILImage


class TransformerService:
    def __init__(self, transform, combine_fn):
        self.__transform = transform
        self.__combine_fn = combine_fn

    def transform(self, pdf_file):
        pages = pdf_file._PDFFile__pages
        results = []

        for i in range(len(pages) - 1):
            img_a = PILImage.fromarray(pages[i].image.raw_data)
            img_b = PILImage.fromarray(pages[i + 1].image.raw_data)

            img_a = self.__transform(img_a)
            img_b = self.__transform(img_b)

            image_pair = self.__combine_fn([img_a, img_b], dim=0)
            results.append((image_pair, i))

        return results
