import os
import shutil
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
# The models Tesseract needs: English text, and orientation and script detection (osd) for rotated pages.
TESSDATA_MODELS = ('eng', 'osd')


class NoTextFoundError(Exception):
    """
    Raised when Tesseract finds no recognizable text in a document.
    """
    pass


class OcrSetupError(Exception):
    """
    Raised when the OCR tools aren't set up, so every file would fail.
    """
    pass


def check_ocr_setup(tessdata_path: str = TESSDATA_PATH) -> None:
    """
    Checks that poppler and Tesseract are usable before any file is processed.

    Parameters
    ----------
    tessdata_path: str
      The directory holding Tesseract's language data.

    Raises
    ------
    OcrSetupError
      If poppler isn't installed, a Tesseract model is missing, or Tesseract can't start with the given tessdata.
    """
    missing = [tool for tool in ('pdfinfo', 'pdftoppm') if shutil.which(tool) is None]
    if missing:
        raise OcrSetupError(f'Poppler is not installed or not on PATH (missing {", ".join(missing)}). '
                            'pdf2image needs it to read PDFs.')
    if not os.path.isdir(tessdata_path):
        raise OcrSetupError(f'Tesseract data directory {tessdata_path} does not exist.')
    # Without the osd model, Tesseract only warns and reads rotated pages as garbage, so check for it here.
    missing = [model for model in TESSDATA_MODELS
               if not os.path.isfile(os.path.join(tessdata_path, f'{model}.traineddata'))]
    if missing:
        raise OcrSetupError(f'Tesseract model(s) {", ".join(missing)} missing from {tessdata_path}.')
    try:
        with tesserocr.PyTessBaseAPI(tessdata_path):
            pass
    except RuntimeError as e:
        raise OcrSetupError(f'Tesseract could not start with the data in {tessdata_path}: {e}') from e


class TesseractOperationBase(ABC):
    """
    A base operation class for further use.

    Notes
    -----
    `process_row` raises on failure rather than returning an empty list, so the caller can record it. A file
    that fails part way through therefore saves nothing, rather than its first pages as if they were all of it.
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

        pdf2image returns each page as a PPM image, and tesserocr passes it on in that format, so Leptonica reads it
        with its own PPM reader. The older libpng, libtiff, libwebp and libjpeg bundled in the tesserocr wheel aren't
        used for our pages; keep it that way if this changes (e.g. converting the image first makes tesserocr pass a
        BMP, which Leptonica also reads itself).

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
            try:
                yield pg, image
            finally:
                # Also runs if the caller stops part way through, e.g. because OCR on this page failed.
                image.close()

    @staticmethod
    def _open_api() -> tesserocr.PyTessBaseAPI:
        """
        Opens a Tesseract instance, to be reused for every page of one file.

        Notes
        -----
        The page segmentation mode is AUTO_OSD, which detects each page's orientation first and turns it upright.
        In the default mode (AUTO), pages scanned upside down or turned 90 degrees counter-clockwise came out as
        garbage. tests/pytests/ocr_accuracy_test.py checks every orientation.

        Returns
        -------
        api: tesserocr.PyTessBaseAPI
          The Tesseract instance; use it as a context manager so it is closed afterwards.
        """
        api = tesserocr.PyTessBaseAPI(TESSDATA_PATH, psm=tesserocr.PSM.AUTO_OSD)
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
        confidence_scores = []
        document_content = ''
        with self._open_api() as api:
            for pg, image in self._iter_pages(input_path):
                self._set_page(api, image)
                ocr_result = api.GetUTF8Text()
                confidence = api.MeanTextConf()
                if ocr_result.strip() and confidence > 0:
                    document_content += "\n" + ocr_result
                    confidence_scores.append(confidence)
        if not confidence_scores:
            raise NoTextFoundError(f'No recognizable text found in {input_path}.')
        doc_confidence = sum(confidence_scores) / len(confidence_scores)
        stdout_logger.info(f'Item {input_path} had a confidence score of {doc_confidence}.')
        return [{'content': document_content, 'confidence': doc_confidence, 'page': 1, 'block': 1}]


class TesseractOperationPage(TesseractOperationBase):
    """
    Returns Tesseract results for entire pages.
    """
    def process_row(self, input_path: str) -> list:
        output = []
        with self._open_api() as api:
            for pg, image in self._iter_pages(input_path):
                self._set_page(api, image)
                ocr_result = api.GetUTF8Text()
                page_confidence = api.MeanTextConf()
                print(f'Item {input_path}:{pg} had a confidence score of {page_confidence}.')
                output.append({'content': ocr_result, 'confidence': page_confidence, 'page': pg + 1, 'block': 0})
        if not any(result['content'].strip() for result in output):
            raise NoTextFoundError(f'No recognizable text found in {input_path}.')
        return output


class TesseractOperationBlock(TesseractOperationBase):
    """
    Returns Tesseract results for individual blocks of text.

    Notes
    -----
    Each block is one line of text. Each page is recognised once, as in page mode, and the result is split into
    the lines Tesseract found, so block mode has the same words as page mode. Lines with no text (such as image
    regions) are left out.

    Reading each line again from a box cropped around it gave worse results on real records: boxes cut tight to
    the text often came back empty, padded boxes picked up neighbouring lines in densely typed records, and the
    padding that worked best changed between Tesseract versions.
    """

    def process_row(self, input_path: str) -> list:
        output = []
        level = tesserocr.RIL.TEXTLINE
        with self._open_api() as api:
            for pg, image in self._iter_pages(input_path):
                self._set_page(api, image)
                api.Recognize()
                block = 0
                for line in tesserocr.iterate_level(api.GetIterator(), level):
                    ocr_result = line.GetUTF8Text(level) or ''
                    if not ocr_result.strip():
                        continue
                    block += 1
                    line_confidence = line.Confidence(level)
                    print(f'Item {input_path}:{pg}:{block} had a confidence score of {line_confidence}.')
                    output.append({'content': ocr_result, 'confidence': line_confidence, 'page': pg + 1, 'block': block})
        if not any(result['content'].strip() for result in output):
            raise NoTextFoundError(f'No recognizable text found in {input_path}.')
        return output
