import math
import os
import re
import shutil
import subprocess
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
# Pages that read with a mean confidence below this are checked for being sideways or upside down. Pages scanned
# that way read at about 13 to 36; most upright pages read well above it.
ORIENTATION_CHECK_BELOW_CONFIDENCE = 60
# Tesseract's confidence in a detected orientation, below which it's treated as a guess and ignored. Turned pages,
# real and synthetic, measured 2.5 to 14; a guess that turned a readable real page upside down measured 0.1.
MIN_ORIENTATION_CONFIDENCE = 1.0
# The models Tesseract needs: English text, and orientation and script detection (osd) for rotated pages.
TESSDATA_MODELS = ('eng', 'osd')
# The resolution pages are rendered at, pdf2image's default.
RENDER_DPI = 200
# The most pixels a rendered page may have: Pillow refuses to open larger images (twice its Image.MAX_IMAGE_PIXELS,
# which is set here as a number since tests replace Pillow where it isn't installed). A page too large to render at
# RENDER_DPI within this is rendered at a lower resolution instead; that's how a scan with a page size of several
# feet failed. Pages over half of it render as before, with a DecompressionBombWarning. OCR of a page this size
# peaks at about 4 GB.
MAX_PAGE_PIXELS = 178_956_970
# Below this, text is too small to read, so a page that would need it fails as too large rather than as having no
# text.
MIN_RENDER_DPI = 50
POINTS_PER_INCH = 72


class NoTextFoundError(Exception):
    """
    Raised when Tesseract finds no recognizable text in a document.
    """
    pass


class PageTooLargeError(Exception):
    """
    Raised when a page is too large to render at a resolution its text could be read at.
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
    # The default start-up doesn't load the osd model, and OCR only warns if it can't, so load it on its own: a
    # truncated or corrupt osd.traineddata fails here instead of quietly turning rotated pages into garbage.
    try:
        with tesserocr.PyTessBaseAPI(tessdata_path, lang='osd', psm=tesserocr.PSM.OSD_ONLY):
            pass
    except RuntimeError as e:
        raise OcrSetupError(f'Tesseract could not load the osd model in {tessdata_path}: {e}') from e


def page_sizes(input_path: str, page_count: int) -> dict[int, tuple[float, float]]:
    """
    Gets the size of every page in a PDF, from one pdfinfo call.

    The size is the page's MediaBox, which is what pdf2image renders (pdfinfo's "Page size" is the CropBox, which
    can be smaller).

    Parameters
    ----------
    input_path: str
      The path to the PDF.
    page_count: int
      The number of pages.

    Returns
    -------
    dict[int, tuple[float, float]]
      Page number, from 1, to (width, height) in inches. Pages pdfinfo didn't report are left out.
    """
    result = subprocess.run(['pdfinfo', '-box', '-f', '1', '-l', str(page_count), input_path],
                            capture_output=True, text=True, errors='replace')
    return parse_media_boxes(result.stdout)


def parse_media_boxes(pdfinfo_output: str) -> dict[int, tuple[float, float]]:
    """
    Reads the page sizes from the output of `pdfinfo -box`.

    >>> parse_media_boxes('Page    1 MediaBox:     0.00     0.00   612.00   792.00\\n'
    ...                   'Page    1 CropBox:      0.00     0.00   306.00   396.00\\n'
    ...                   'Page    2 MediaBox:    -9.00     0.00   711.00  1152.00\\n')
    {1: (8.5, 11.0), 2: (10.0, 16.0)}

    Returns
    -------
    dict[int, tuple[float, float]]
      Page number to (width, height) in inches.
    """
    sizes = {}
    number = r'(-?\d+(?:\.\d+)?)'
    for match in re.finditer(rf'^Page\s+(\d+) MediaBox:\s+{number}\s+{number}\s+{number}\s+{number}\s*$',
                             pdfinfo_output, re.MULTILINE):
        x0, y0, x1, y1 = (float(value) for value in match.groups()[1:])
        sizes[int(match.group(1))] = (abs(x1 - x0) / POINTS_PER_INCH, abs(y1 - y0) / POINTS_PER_INCH)
    return sizes


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
        sizes = page_sizes(input_path, page_count)
        for pg in range(page_count):
            dpi = TesseractOperationBase._render_dpi(input_path, pg + 1, sizes.get(pg + 1))
            images = pdf2image.convert_from_path(input_path, dpi=dpi, first_page=pg + 1, last_page=pg + 1)
            image = images[0]
            del images
            try:
                yield pg, image
            finally:
                # Also runs if the caller stops part way through, e.g. because OCR on this page failed.
                image.close()

    @staticmethod
    def _render_dpi(input_path: str, page: int, size: tuple[float, float] | None) -> int:
        """
        Chooses the resolution to render a page at: RENDER_DPI, or lower if the page would then have more than
        MAX_PAGE_PIXELS pixels.

        A page that size is almost always a scan whose page size was set wrong, so it's several feet across, and
        its text is still large enough to read at the lower resolution.

        Parameters
        ----------
        input_path: str
          The path to the PDF, for the log.
        page: int
          The page number, from 1.
        size: tuple[float, float] | None
          The page's width and height in inches (see `page_sizes`), or None if they aren't known.

        Returns
        -------
        dpi: int
          The resolution, RENDER_DPI if the page's size isn't known or is zero.

        Raises
        ------
        PageTooLargeError
          If the page would have to be rendered below MIN_RENDER_DPI.
        """
        if not size or not all(size):
            stdout_logger.warning(f'Item {input_path}: the size of page {page} is unknown, so it is rendered at '
                                  f'{RENDER_DPI} dpi without checking it fits.')
            return RENDER_DPI
        width, height = size
        dpi = TesseractOperationBase._fit_dpi(width, height)
        if dpi < MIN_RENDER_DPI:
            raise PageTooLargeError(f'Page {page} is {width:.0f} x {height:.0f} inches, too large to read: it would '
                                    f'have to be rendered at {dpi} dpi, below the minimum of {MIN_RENDER_DPI}.')
        if dpi < RENDER_DPI:
            stdout_logger.info(f'Item {input_path}: page {page} is {width:.0f} x {height:.0f} inches, so it is '
                               f'rendered at {dpi} dpi instead of {RENDER_DPI}.')
        return dpi

    @staticmethod
    def _fit_dpi(width: float, height: float) -> int:
        """
        The highest resolution, up to RENDER_DPI, at which a page of the given size in inches has at most
        MAX_PAGE_PIXELS pixels.

        >>> TesseractOperationBase._fit_dpi(8.5, 11)
        200
        >>> TesseractOperationBase._fit_dpi(59, 80)
        194

        Poppler can round each side up by a pixel, so that's allowed for:

        >>> dpi = TesseractOperationBase._fit_dpi(59, 80)
        >>> (59 * dpi + 1) * (80 * dpi + 1) <= MAX_PAGE_PIXELS < (59 * (dpi + 1) + 1) * (80 * (dpi + 1) + 1)
        True
        """
        dpi = min(RENDER_DPI, math.floor(math.sqrt(MAX_PAGE_PIXELS / (width * height))))
        while dpi > 0 and (width * dpi + 1) * (height * dpi + 1) > MAX_PAGE_PIXELS:
            dpi -= 1
        return dpi

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
    def _open_osd_api() -> tesserocr.PyTessBaseAPI:
        """
        Opens a Tesseract instance that only detects page orientation, to be reused for every page of one file.

        Returns
        -------
        api: tesserocr.PyTessBaseAPI
          The Tesseract instance; use it as a context manager so it is closed afterwards.
        """
        api = tesserocr.PyTessBaseAPI(TESSDATA_PATH, lang='osd', psm=tesserocr.PSM.OSD_ONLY)
        api.SetVariable('debug_file', '/dev/null')
        return api

    @staticmethod
    def _detect_rotation(osd_api: tesserocr.PyTessBaseAPI, image: Image.Image) -> int:
        """
        Detects how far a page is turned.

        Returns
        -------
        int
          The degrees to turn the page counter-clockwise to make it upright: 90, 180 or 270, or 0 if it is upright
          or its orientation can't be told (for example, because it has too little text, or Tesseract is only
          guessing).
        """
        osd_api.SetImage(image)
        try:
            result = osd_api.DetectOrientationScript()
        except RuntimeError:
            return 0
        if not result or result.get('orient_conf', 0) < MIN_ORIENTATION_CONFIDENCE:
            return 0
        angle = result.get('orient_deg')
        return angle if angle in (90, 180, 270) else 0

    @staticmethod
    def _reading_score(text: str, confidence: int) -> int:
        """
        How much confidently read text a reading has: its mean confidence times its number of words.

        Mean confidence alone isn't enough to choose between two readings of a page: Tesseract can report a high
        confidence for a reading with no text at all.

        >>> TesseractOperationBase._reading_score('Weight 31.2 kg', 90)
        270
        >>> TesseractOperationBase._reading_score('', 95)
        0
        """
        return confidence * len(text.split())

    def _read_page(self, api: tesserocr.PyTessBaseAPI, osd_api: tesserocr.PyTessBaseAPI,
                   image: Image.Image) -> tuple[str, int]:
        """
        Reads one page, turning it upright first if it was scanned sideways or upside down.

        Notes
        -----
        A page that reads with low confidence is checked for its orientation. If it is turned, it is turned upright
        and read again, and the reading with more confidently read text (see `_reading_score`) is kept. Pages scanned
        upside down or turned 90 degrees counter-clockwise otherwise come out as garbage. Tesseract's AUTO_OSD mode,
        which checks every page, was tried first: on real records it also turned some upright pages that read well,
        ruining their text, and made OCR about 20% slower. tests/pytests/ocr_accuracy_test.py checks every
        orientation.

        Afterwards `api` holds the recognised version of the page that was kept, so its results can be iterated.

        Returns
        -------
        text, confidence: tuple[str, int]
          The page's text and mean confidence.
        """
        self._set_page(api, image)
        text, confidence = api.GetUTF8Text(), api.MeanTextConf()
        if confidence >= ORIENTATION_CHECK_BELOW_CONFIDENCE:
            return text, confidence
        angle = self._detect_rotation(osd_api, image)
        if not angle:
            return text, confidence
        upright = image.rotate(angle, expand=True)
        try:
            self._set_page(api, upright)
            upright_text, upright_confidence = api.GetUTF8Text(), api.MeanTextConf()
        finally:
            upright.close()
        if self._reading_score(upright_text, upright_confidence) > self._reading_score(text, confidence):
            return upright_text, upright_confidence
        # Turning didn't help, so go back to the page as scanned.
        self._set_page(api, image)
        api.Recognize()
        return text, confidence

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
        with self._open_api() as api, self._open_osd_api() as osd_api:
            for pg, image in self._iter_pages(input_path):
                ocr_result, confidence = self._read_page(api, osd_api, image)
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
        with self._open_api() as api, self._open_osd_api() as osd_api:
            for pg, image in self._iter_pages(input_path):
                ocr_result, page_confidence = self._read_page(api, osd_api, image)
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
        with self._open_api() as api, self._open_osd_api() as osd_api:
            for pg, image in self._iter_pages(input_path):
                # Recognises the page (turned upright if needed), so its lines can be iterated below.
                self._read_page(api, osd_api, image)
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
