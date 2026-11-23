"""
Tests that the OCR operations take a born-digital page's text from the PDF and read every other page with OCR, in
each document type, on generated PDFs of made-up text.

Tesseract and poppler are only installed in the Docker image, so these are skipped elsewhere. In CI they run inside
the image, where skipped tests fail the build.
"""

import os
import re
import shutil

import pytest

tesserocr = pytest.importorskip('tesserocr')
pdf2image = pytest.importorskip('pdf2image')
if shutil.which('pdftotext') is None or not os.path.isdir('/usr/share/tessdata'):
    pytest.skip('poppler or the Tesseract models are not installed', allow_module_level=True)

from tests.synthetic_pdfs import (DigitalPage, SyntheticPage,  # noqa: E402
                                  write_pdf)
from vmrt_tesseract_utilities import tesseract_operations  # noqa: E402
from vmrt_tesseract_utilities.tesseract_operations import (  # noqa: E402
    SOURCE_MIXED, SOURCE_OCR, SOURCE_TEXT_LAYER, TesseractOperationBlock,
    TesseractOperationDoc, TesseractOperationPage)

VISIT = [
    'Annual wellness exam for a nine year old spayed female.',
    'Owner reports good appetite, normal energy and no vomiting.',
    'Exam: Weight 31.2 kg. Temperature 101.4 F. Heart rate 96 bpm.',
    ('Vaccine', 'Date given', 'Next due'),
    ('Rabies', '03/14/2023', '03/14/2026'),
]
SCANNED = ['Recheck exam, ears clean.', 'Weight 30.8 kg.']
GARBLED = ['Dental cleaning scheduled for next month.', 'Bloodwork before anesthesia is normal.'] * 2
PASTED = ['Lab report: glucose normal', 'Creatinine normal']


def words(text: str) -> list[str]:
    return re.findall(r'[A-Za-z0-9./:]+', text)


@pytest.fixture(scope='module')
def mixed_pdf(tmp_path_factory) -> str:
    """
    Four pages: born digital, a scan, born digital with a broken font encoding, and born digital with a scanned
    report pasted over a third of it.
    """
    path = str(tmp_path_factory.mktemp('text_layer') / 'mixed.pdf')
    return write_pdf(path, [
        DigitalPage(VISIT),
        SyntheticPage(SCANNED),
        DigitalPage(GARBLED, font_size=16, garbled=True),
        DigitalPage(VISIT[:3], image=SyntheticPage(PASTED, font_size=96), image_box=(36, 36, 540, 300)),
    ])


@pytest.fixture(scope='module')
def digital_pdf(tmp_path_factory) -> str:
    path = str(tmp_path_factory.mktemp('text_layer') / 'digital.pdf')
    return write_pdf(path, [DigitalPage(VISIT), DigitalPage(VISIT[:3])])


def test_page_mode_reads_each_page_the_right_way(mixed_pdf):
    output = TesseractOperationPage().process_row(mixed_pdf)
    assert [(o['page'], o['source']) for o in output] == [
        (1, SOURCE_TEXT_LAYER), (2, SOURCE_OCR), (3, SOURCE_OCR), (4, SOURCE_OCR)]
    # The born-digital page's text is exact, and has no OCR confidence.
    assert words(output[0]['content']) == words(DigitalPage(VISIT).text)
    assert output[0]['confidence'] is None
    assert all(o['confidence'] > 50 for o in output[1:])
    assert 'Recheck exam' in output[1]['content']
    # The garbled page copies out as symbols, but OCR reads what's printed.
    assert 'Dental cleaning scheduled' in output[2]['content']
    # The page with a pasted-in report gets the report's text too, which its text layer doesn't have.
    assert 'glucose' in output[3]['content'] and 'Annual wellness exam' in output[3]['content']


def test_text_layer_pages_are_not_rendered(mixed_pdf, monkeypatch):
    rendered = []
    convert_from_path = pdf2image.convert_from_path

    def tracking(path, **kwargs):
        rendered.append(kwargs['first_page'])
        return convert_from_path(path, **kwargs)

    monkeypatch.setattr(tesseract_operations.pdf2image, 'convert_from_path', tracking)
    TesseractOperationPage().process_row(mixed_pdf)
    assert rendered == [2, 3, 4]


def test_document_mode_read_both_ways_is_mixed(mixed_pdf):
    [output] = TesseractOperationDoc().process_row(mixed_pdf)
    assert output['source'] == SOURCE_MIXED
    assert 'Annual wellness exam' in output['content'] and 'Recheck exam' in output['content']
    # The mean confidence of the three pages read with OCR.
    assert 50 < output['confidence'] <= 100


def test_document_mode_all_from_the_text_layer_has_no_confidence(digital_pdf):
    [output] = TesseractOperationDoc().process_row(digital_pdf)
    assert output['source'] == SOURCE_TEXT_LAYER
    assert output['confidence'] is None
    assert words(output['content']) == words(DigitalPage(VISIT).text + '\n' + DigitalPage(VISIT[:3]).text)


def test_block_mode_splits_text_layer_pages_into_lines(digital_pdf):
    output = TesseractOperationBlock().process_row(digital_pdf)
    assert [(o['page'], o['block']) for o in output] == [(1, b) for b in range(1, 6)] + [(2, b) for b in range(1, 4)]
    assert all(o['source'] == SOURCE_TEXT_LAYER and o['confidence'] is None for o in output)
    # A table row stays one block, its cells apart.
    assert re.split(r'\s{2,}', output[4]['content'].strip()) == ['Rabies', '03/14/2023', '03/14/2026']


@pytest.mark.parametrize('operation', [TesseractOperationDoc, TesseractOperationPage, TesseractOperationBlock])
def test_ocr_only_reads_every_page_with_ocr(digital_pdf, operation):
    output = operation(ocr_only=True).process_row(digital_pdf)
    assert {o['source'] for o in output} == {SOURCE_OCR}
    assert all(o['confidence'] > 50 for o in output)
