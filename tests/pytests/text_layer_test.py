"""
Tests for reading a PDF's text layer and deciding which pages still need OCR.

The tests that read PDFs need poppler and Pillow, which are only installed in the Docker image, so they're skipped
elsewhere. In CI they run inside the image, where skipped tests fail the build. All of the text is made up.
"""

import re
import shutil
import subprocess
from unittest import mock

import PIL
import pytest

from tests.synthetic_pdfs import DigitalPage, SyntheticPage, write_pdf
from vmrt_tesseract_utilities import text_layer
from vmrt_tesseract_utilities.text_layer import (GARBLED, IMAGE, NO_TEXT, SCAN,
                                                 TextLayerError, read_pages)

needs_poppler = pytest.mark.skipif(shutil.which('pdftotext') is None or isinstance(PIL, mock.MagicMock),
                                   reason='poppler or Pillow is not installed')

VISIT = [
    'Annual wellness exam for a nine year old spayed female.',
    'Owner reports good appetite, normal energy and no vomiting.',
    'Exam: Weight 31.2 kg. Temperature 101.4 F. Heart rate 96 bpm.',
    ('Vaccine', 'Date given', 'Next due'),
    ('Rabies', '03/14/2023', '03/14/2026'),
    ('DHPP', '03/14/2023', '03/14/2024'),
]
SCANNED_LINES = ['Recheck exam, ears clean.', 'Weight 30.8 kg.']


def words(text: str) -> list[str]:
    return text.split()


@pytest.fixture
def pdf(tmp_path):
    """
    Writes synthetic pages to a PDF and returns its path.
    """

    def make(*pages):
        return write_pdf(str(tmp_path / 'record.pdf'), pages)
    return make


@needs_poppler
def test_born_digital_page_uses_its_text_layer(pdf):
    page = DigitalPage(VISIT)
    [result] = read_pages(pdf(page))
    assert result.page == 1
    assert result.use_text_layer
    assert words(result.text) == words(page.text)


@needs_poppler
def test_table_rows_stay_on_one_line(pdf):
    [result] = read_pages(pdf(DigitalPage(VISIT)))
    rows = [re.split(r'\s{2,}', line.strip()) for line in result.text.splitlines()]
    assert ['Rabies', '03/14/2023', '03/14/2026'] in rows


@needs_poppler
def test_scan_needs_ocr(pdf):
    [result] = read_pages(pdf(SyntheticPage(SCANNED_LINES)))
    assert result.ocr_reason == NO_TEXT
    assert result.text.strip() == ''


@needs_poppler
def test_scan_with_its_own_text_layer_still_needs_ocr(pdf):
    page = DigitalPage(VISIT, image=SyntheticPage(SCANNED_LINES), invisible=True)
    [result] = read_pages(pdf(page))
    assert result.ocr_reason == SCAN
    # The invisible text is still there; it's just not trusted.
    assert words(result.text) == words(page.text)


@needs_poppler
def test_scan_stored_as_an_inline_image_still_needs_ocr(pdf):
    # pdfimages lists an inline image with one fewer column, which once hid it.
    page = DigitalPage(VISIT, image=SyntheticPage(SCANNED_LINES), inline_image=True, invisible=True)
    [result] = read_pages(pdf(page))
    assert result.ocr_reason == SCAN


@needs_poppler
def test_scan_stored_as_strips_still_needs_ocr(pdf):
    # 16 strips, each about 6% of the page: none is large on its own, but together they're the whole page.
    page = DigitalPage(VISIT, image=SyntheticPage(SCANNED_LINES), image_strips=16, invisible=True)
    [result] = read_pages(pdf(page))
    assert result.ocr_reason == SCAN


@needs_poppler
def test_large_image_on_a_digital_page_needs_ocr(pdf):
    # A scanned report pasted into the lower part of the page: 4 x 5.2 inches, 22% of the page.
    page = DigitalPage(VISIT, image=SyntheticPage(SCANNED_LINES), image_box=(72, 72, 288, 374))
    [result] = read_pages(pdf(page))
    assert result.ocr_reason == IMAGE


@needs_poppler
def test_logo_on_a_digital_page_keeps_its_text_layer(pdf):
    # A logo of 1.5 x 2 inches: 3% of the page.
    page = DigitalPage(VISIT, image=SyntheticPage(['Logo']), image_box=(450, 650, 108, 144))
    [result] = read_pages(pdf(page))
    assert result.use_text_layer


@needs_poppler
def test_several_small_images_on_a_digital_page_keep_its_text_layer(pdf):
    # Four images of about 5% of the page each: 20% together, but none large enough to hold much text.
    page = DigitalPage(VISIT, image=SyntheticPage(['Logo']), image_box=(72, 72, 288, 260), image_strips=4)
    [result] = read_pages(pdf(page))
    assert result.use_text_layer


@needs_poppler
def test_garbled_text_layer_needs_ocr(pdf):
    [result] = read_pages(pdf(DigitalPage(VISIT, garbled=True)))
    assert result.ocr_reason == f'{GARBLED}: symbols'
    # Its numbers copy out fine; its letters don't.
    assert '03/14/2023' in result.text
    assert 'Rabies' not in result.text


@needs_poppler
def test_page_with_little_text_needs_ocr(pdf):
    [result] = read_pages(pdf(DigitalPage(['Page 2 of 3', 'Fax from clinic'])))
    assert result.ocr_reason == NO_TEXT


@needs_poppler
def test_mixed_pdf_is_decided_page_by_page(pdf):
    results = read_pages(pdf(DigitalPage(VISIT), SyntheticPage(SCANNED_LINES), DigitalPage(VISIT, garbled=True),
                             SyntheticPage(SCANNED_LINES, rotation=90), DigitalPage(VISIT[:3])))
    assert [r.page for r in results] == [1, 2, 3, 4, 5]
    assert [r.ocr_reason.split(':')[0] for r in results] == ['', NO_TEXT, GARBLED, NO_TEXT, '']
    assert words(results[4].text) == words('\n'.join(VISIT[:3]))


@needs_poppler
def test_damaged_pdf_raises(tmp_path):
    path = tmp_path / 'broken.pdf'
    path.write_bytes(b'%PDF-1.4\nthis is not a PDF\n')
    with pytest.raises(TextLayerError, match='broken.pdf'):
        read_pages(str(path))


def test_poppler_failure_raises_with_its_message():
    failed = subprocess.CompletedProcess(['pdfinfo'], 1, stdout=b'', stderr=b'Syntax Error: Couldn\'t find trailer')
    with mock.patch('subprocess.run', return_value=failed):
        with pytest.raises(TextLayerError, match="pdfinfo exited with code 1 on record.pdf: Syntax Error"):
            read_pages('record.pdf')


def test_poppler_timeout_raises():
    with mock.patch('subprocess.run', side_effect=subprocess.TimeoutExpired('pdftotext', 300)):
        with pytest.raises(TextLayerError, match='took longer than 300 seconds on record.pdf'):
            read_pages('record.pdf')


def test_read_pages_puts_the_tools_together():
    outputs = {
        'pdfinfo': 'Pages:          3\nPage    1 size: 612 x 792 pts\nPage    2 size: 612 x 792 pts\n'
                   'Page    3 size: 612 x 792 pts\n',
        'pdfimages': 'page   num  type   width height color comp bpc  enc interp  object ID x-ppi y-ppi size ratio\n'
                     '-----------------------------------------------------------------------------------------\n'
                     '   2     0 image    1700  2200  rgb     3   8  jpeg   no         7  0   200   200  263K 2.4%\n',
        'pdftotext': ('Annual exam. Weight 32 kg. Vaccines current. ' * 3 + '\n\f') * 2 + 'Page 3\n\f',
    }

    def run(args, **kwargs):
        return subprocess.CompletedProcess(args, 0, stdout=outputs[args[0]].encode(), stderr=b'')
    with mock.patch('subprocess.run', side_effect=run):
        results = read_pages('record.pdf')
    assert [(r.page, r.ocr_reason) for r in results] == [(1, ''), (2, SCAN), (3, NO_TEXT)]


@pytest.mark.parametrize('text, expected', [
    # A real page's text: words, numbers, a table, a dashed line and form lines.
    ('Exam: Weight 31.2 kg.\nRabies   03/14/2023   03/14/2026\n' + '-' * 60 + '\nSigned: ________\n' * 3, ''),
    # Typographic punctuation and Latin-1 symbols are usual.
    ('Temp 101.4 °F – “normal” • dose 5 µg ± 1 ' * 10, ''),
    ('一二三 ' * 40, 'odd characters'),
    ('Clinic \x01\x02 visit ' * 20, 'control characters'),
    ('&%$# visit ' * 20, 'symbols'),
    ('qzxv brrk tssp ' * 10, 'not words'),
    # Too few tokens to judge whether they're words.
    ('qzxv brrk ' * 9, ''),
])
def test_garbled_reason(text, expected):
    assert text_layer.garbled_reason(text) == expected
