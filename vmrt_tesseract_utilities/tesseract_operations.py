from abc import ABC, abstractmethod
from typing import Iterator

import pdf2image
import tesserocr
from PIL import Image

from vmrt_tesseract_utilities.logging import stdout_logger

"""
Provides Tesseract processing with different output scales (document, page, block).
"""

TESSDATA_PATH = '/usr/share/tessdata'


class TesseractOperationBase(ABC):
    """
    A base operation class for further use.
    """

    @abstractmethod
    def process_row(self, input_path: str) -> list:
        pass

    @staticmethod
    def _iter_pages(input_path: str) -> Iterator[tuple[int, Image.Image]]:
        """
        Renders a PDF one page at a time, so only a single page image is held in memory.

        Pages stay in colour: rendering them in greyscale with poppler gives slightly different OCR text and
        confidence scores on colour scans than letting Tesseract do the conversion.

        Parameters
        ----------
        input_path: str
          The path to the PDF.

        Yields
        ------
        page_index, image: tuple[int, Image.Image]
          The zero-based page index and the rendered page.
        """
        page_count = pdf2image.pdfinfo_from_path(input_path)['Pages']
        stdout_logger.info(f'Item {input_path} has {page_count} pages.')
        for pg in range(page_count):
            images = pdf2image.convert_from_path(input_path, first_page=pg + 1, last_page=pg + 1)
            image = images[0]
            del images
            yield pg, image
            image.close()

    @staticmethod
    def _open_api() -> tesserocr.PyTessBaseAPI:
        """
        Opens a Tesseract instance, to be reused for every page of one file.

        Returns
        -------
        api: tesserocr.PyTessBaseAPI
          The Tesseract instance; use it as a context manager so it is closed afterwards.
        """
        api = tesserocr.PyTessBaseAPI(TESSDATA_PATH)
        api.SetVariable('debug_file', '/dev/null')
        return api

    @staticmethod
    def _set_page(api: tesserocr.PyTessBaseAPI, image: Image.Image) -> None:
        """
        Loads a page into a reused Tesseract instance.

        Notes
        -----
        The adaptive classifier is cleared first, so each page is recognised as if by a fresh instance,
        as before the instance was shared between pages.
        """
        api.ClearAdaptiveClassifier()
        api.SetImage(image)


class TesseractOperationDoc(TesseractOperationBase):
    """
    Returns Tesseract results for entire documents.
    """
    def process_row(self, input_path: str) -> list:
        output = []
        try:
            confidence_scores = []
            document_content = ''
            with self._open_api() as api:
                for pg, image in self._iter_pages(input_path):
                    self._set_page(api, image)
                    ocr_result = api.GetUTF8Text()
                    confidence = api.MeanTextConf()
                    if len(ocr_result) > 0 and confidence > 0:
                        document_content += "\n" + ocr_result
                        confidence_scores.append(confidence)
            doc_confidence = sum(confidence_scores) / len(confidence_scores)
            stdout_logger.info(f'Item {input_path} had a confidence score of {doc_confidence}.')
            output.append({'content': document_content, 'confidence': doc_confidence, 'page': 1, 'block': 1})
        except Exception as e:
            stdout_logger.error(str(e))
        return output


class TesseractOperationPage(TesseractOperationBase):
    """
    Returns Tesseract results for entire pages.
    """
    def process_row(self, input_path: str) -> list:
        output = []
        try:
            with self._open_api() as api:
                for pg, image in self._iter_pages(input_path):
                    self._set_page(api, image)
                    ocr_result = api.GetUTF8Text()
                    page_confidence = api.MeanTextConf()
                    print(f'Item {input_path}:{pg} had a confidence score of {page_confidence}.')
                    output.append({'content': ocr_result, 'confidence': page_confidence, 'page': pg + 1, 'block': 0})
        except Exception as e:
            stdout_logger.error(str(e))
        return output


class TesseractOperationBlock(TesseractOperationBase):
    """
    Returns Tesseract results for individual blocks of text.

    Notes
    -----
    Each block is one line of text. Tesseract finds the lines with its full page layout analysis, then reads
    each one as a single line from a slightly padded box: boxes cut tight to the text often come back empty.
    """
    BOX_PADDING = 10

    def process_row(self, input_path: str) -> list:
        output = []
        try:
            with self._open_api() as api:
                for pg, image in self._iter_pages(input_path):
                    self._set_page(api, image)
                    api.SetPageSegMode(tesserocr.PSM.AUTO)
                    boxes = api.GetComponentImages(tesserocr.RIL.TEXTLINE, True)
                    api.SetPageSegMode(tesserocr.PSM.SINGLE_LINE)
                    for i, (_, box, _, _) in enumerate(boxes):
                        x = max(0, box['x'] - self.BOX_PADDING)
                        y = max(0, box['y'] - self.BOX_PADDING)
                        w = min(box['w'] + 2 * self.BOX_PADDING, image.width - x)
                        h = min(box['h'] + 2 * self.BOX_PADDING, image.height - y)
                        api.SetRectangle(x, y, w, h)
                        ocr_result = api.GetUTF8Text()
                        box_confidence = api.MeanTextConf()
                        print(f'Item {input_path}:{pg}:{i} had a confidence score of {box_confidence}.')
                        output.append({'content': ocr_result, 'confidence': box_confidence, 'page': pg + 1, 'block': i + 1})
        except Exception as e:
            stdout_logger.error(str(e))
        return output
