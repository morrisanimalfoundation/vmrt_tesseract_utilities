from unittest import mock

import pytest

from vmrt_tesseract_utilities import tesseract_operations
from vmrt_tesseract_utilities.tesseract_operations import (
    ORIENTATION_CHECK_BELOW_CONFIDENCE, TesseractOperationPage)

"""
Tests for when a page is turned upright before OCR, with Tesseract replaced by fakes, so they also run where it isn't
installed. tests/pytests/ocr_accuracy_test.py checks the real thing on rotated pages inside the image.
"""


def _fakes(readings, angle=180):
    """
    Fake Tesseract instances: `readings` is the (text, confidence) of each successive read, and the orientation
    check reports `angle`, or raises if it's an exception.
    """
    api = mock.MagicMock()
    api.GetUTF8Text.side_effect = [text for text, _ in readings]
    api.MeanTextConf.side_effect = [confidence for _, confidence in readings]
    osd_api = mock.MagicMock()
    if isinstance(angle, Exception):
        osd_api.DetectOrientationScript.side_effect = angle
    else:
        osd_api.DetectOrientationScript.return_value = {'orient_deg': angle, 'orient_conf': 10.0}
    return api, osd_api, mock.MagicMock(name='page')


def test_confident_page_is_not_checked():
    """
    Test that a page that reads well is kept as it is, without the orientation check (which costs time and, in
    Tesseract's AUTO_OSD mode, turned some good upright pages on real records).
    """
    api, osd_api, image = _fakes([('Weight 31.2 kg', ORIENTATION_CHECK_BELOW_CONFIDENCE)])
    assert TesseractOperationPage()._read_page(api, osd_api, image) == ('Weight 31.2 kg', 60)
    osd_api.DetectOrientationScript.assert_not_called()
    image.rotate.assert_not_called()


def test_turned_page_is_read_upright():
    """
    Test that a page that reads poorly and is turned is read again upright, and that reading is kept.
    """
    api, osd_api, image = _fakes([('gArb1e', 30), ('Weight 31.2 kg', 88)], angle=270)
    assert TesseractOperationPage()._read_page(api, osd_api, image) == ('Weight 31.2 kg', 88)
    image.rotate.assert_called_once_with(270, expand=True)
    api.SetImage.assert_called_with(image.rotate.return_value)
    image.rotate.return_value.close.assert_called_once()


def test_turning_that_does_not_help_is_dropped():
    """
    Test that if the upright reading isn't more confident, the page as scanned is kept, and is loaded back into
    Tesseract so block mode iterates over the reading that was kept.
    """
    api, osd_api, image = _fakes([('Poor scan', 50), ('worse', 49)], angle=90)
    assert TesseractOperationPage()._read_page(api, osd_api, image) == ('Poor scan', 50)
    api.SetImage.assert_called_with(image)
    api.Recognize.assert_called_once()


@pytest.mark.parametrize('angle', [0, RuntimeError('too few characters')])
def test_upright_or_unknown_orientation_is_left_alone(angle):
    """
    Test that a poorly read page is kept as it is if it's upright, or if its orientation can't be told.
    """
    api, osd_api, image = _fakes([('Page 2', 20)], angle=angle)
    assert TesseractOperationPage()._read_page(api, osd_api, image) == ('Page 2', 20)
    image.rotate.assert_not_called()


def test_every_mode_reads_pages_through_the_orientation_check(monkeypatch):
    """
    Test that document, page and block mode all read pages with _read_page, so none of them skips the check.
    """
    reads = []
    monkeypatch.setattr(tesseract_operations.TesseractOperationBase, '_read_page',
                        lambda self, api, osd_api, image: reads.append(type(self).__name__) or ('Text', 90))
    monkeypatch.setattr(tesseract_operations.TesseractOperationBase, '_iter_pages',
                        staticmethod(lambda path: iter([(0, mock.MagicMock())])))
    monkeypatch.setattr(tesseract_operations.tesserocr, 'PyTessBaseAPI', mock.MagicMock())
    monkeypatch.setattr(tesseract_operations.tesserocr, 'iterate_level', lambda iterator, level: [])
    for operation in (tesseract_operations.TesseractOperationDoc(), TesseractOperationPage(),
                      tesseract_operations.TesseractOperationBlock()):
        try:
            operation.process_row('record.pdf')
        except tesseract_operations.NoTextFoundError:
            pass  # Block mode finds no lines in the fake iterator.
    assert reads == ['TesseractOperationDoc', 'TesseractOperationPage', 'TesseractOperationBlock']
