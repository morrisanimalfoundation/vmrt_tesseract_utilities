import difflib
import os
import shutil

import pytest

"""
Checks how accurately OCR reads a generated PDF of known text, not only that it runs.

A Pillow, poppler or Tesseract update can make OCR slightly worse without anything crashing. Each page's OCR text is
compared with the text drawn on it, and the test fails if the similarity drops below MIN_SIMILARITY. The pages cover
paragraphs, a table, a smaller font, and every orientation a page can be scanned in.

Tesseract and poppler are only installed in the Docker image, so these are skipped elsewhere. In CI they run inside
the image, where skipped tests fail the build. All of the text is made up.
"""

tesserocr = pytest.importorskip('tesserocr')
pytest.importorskip('pdf2image')
if shutil.which('pdfinfo') is None or not os.path.isdir('/usr/share/tessdata'):
    pytest.skip('poppler or the Tesseract models are not installed', allow_module_level=True)

from tests.synthetic_pdfs import SyntheticPage, write_pdf  # noqa: E402
from vmrt_tesseract_utilities.tesseract_operations import (  # noqa: E402
    TesseractOperationDoc, TesseractOperationPage)

PROSE = [
    'History: Annual wellness exam for a nine year old spayed female.',
    'Owner reports good appetite, normal energy and no vomiting.',
    'Exam: Weight 31.2 kg. Temperature 101.4 F. Heart rate 96 bpm.',
    'Mild dental tartar noted on the upper premolars (grade 1/4).',
]
TABLE = [
    ('Vaccine', 'Date given', 'Next due'),
    ('Rabies', '03/14/2023', '03/14/2026'),
    ('DHPP', '03/14/2023', '03/14/2024'),
    ('Leptospirosis', '04/02/2023', '04/02/2024'),
]
SMALL_PRINT = [
    'Discharge instructions',
    'Give one tablet by mouth twice daily with food for 10 days.',
    'Call the clinic if lethargy, diarrhea or loss of appetite occurs.',
    'Recheck in six months; bring a fresh stool sample.',
]
ROTATED = [
    'Heartworm test negative. Fecal flotation negative.',
    'Plan: continue monthly preventives and recheck next year.',
]

# Each case is (name, page, whether to ignore word order). Tesseract reads a table's widely spaced columns one after
# another rather than row by row, so for the table only the words are compared, not their order. At 200 dpi, 31 pixel
# text is about 11 point, like typed records, and 20 pixel text is about 7 point fine print. Clean, larger text
# reads perfectly even from a badly degraded page, so it wouldn't show a drop in quality.
PAGES = [
    ('prose', SyntheticPage(PROSE, font_size=31), False),
    ('table', SyntheticPage(TABLE, font_size=31), True),
    ('small print', SyntheticPage(SMALL_PRINT, font_size=20), False),
    ('turned 90 degrees counter-clockwise', SyntheticPage(ROTATED, font_size=31, rotation=90), False),
    ('upside down', SyntheticPage(ROTATED, font_size=31, rotation=180), False),
    ('turned 90 degrees clockwise', SyntheticPage(ROTATED, font_size=31, rotation=270), False),
]

# Every page measured 1.0 (Tesseract 5.5.3 from tesserocr 2.12.0, Pillow 13.0.0, poppler 25.03). The margin allows
# for a few wrong characters on a page. Rendering the pages at a third of their resolution and scaling them back up,
# as a stand-in for worse OCR, cost the small print some characters and made the upside-down page read as garbage,
# and fails the test.
MIN_SIMILARITY = 0.98


def similarity(expected: str, actual: str, ignore_order: bool = False) -> float:
    """
    How similar OCR text is to the expected text, from 0 to 1, ignoring differences in whitespace.

    >>> similarity('Weight 31.2 kg', 'Weight  31.2 kg\\n')
    1.0
    >>> similarity('a b', 'b a', ignore_order=True)
    1.0
    >>> similarity('Weight 31.2 kg', 'Wcight 31.2 kg') < 1
    True
    """
    expected_words, actual_words = expected.split(), actual.split()
    if ignore_order:
        expected_words, actual_words = sorted(expected_words), sorted(actual_words)
    return difflib.SequenceMatcher(None, ' '.join(expected_words), ' '.join(actual_words), autojunk=False).ratio()


@pytest.fixture(scope='module')
def pdf_path(tmp_path_factory) -> str:
    """
    Writes every page in PAGES to one PDF.
    """
    return write_pdf(str(tmp_path_factory.mktemp('ocr_accuracy') / 'accuracy.pdf'), [page for _, page, _ in PAGES])


@pytest.fixture(scope='module')
def page_output(pdf_path) -> list:
    """
    The OCR results for each page, in page mode.
    """
    return TesseractOperationPage().process_row(pdf_path)


@pytest.mark.parametrize('index, name, page, ignore_order',
                         [(i, *case) for i, case in enumerate(PAGES)], ids=[name for name, _, _ in PAGES])
def test_page_is_read_accurately(page_output, index, name, page, ignore_order):
    """
    Test that each page's OCR text is close to the text drawn on it.
    """
    actual = page_output[index]['content']
    score = similarity(page.text, actual, ignore_order)
    assert score >= MIN_SIMILARITY, f'The {name} page scored {score:.3f}; OCR read:\n{actual}'


def test_document_is_read_accurately(pdf_path):
    """
    Test that document mode, which joins every page into one text, is as accurate as the pages on their own.
    """
    actual = TesseractOperationDoc().process_row(pdf_path)[0]['content']
    expected = '\n'.join(page.text for _, page, _ in PAGES)
    # Word order is ignored because of the table.
    score = similarity(expected, actual, ignore_order=True)
    assert score >= MIN_SIMILARITY, f'The document scored {score:.3f}; OCR read:\n{actual}'
