"""
Tests for `grls_id`, the shared definition of a GRLS subject ID.
"""

import re

import pytest

from vmrt_tesseract_utilities.grls_id import (find_subject_id,
                                              ocr_subject_id_regex, parse_id)


@pytest.mark.parametrize('path, expected', [
    ('/data/094-000131', '094-000131'),
    ('/data/094-000131/2019 visits', '094-000131'),
    # A different ID-shaped folder earlier in the path doesn't hide the GRLS one.
    ('/data/123-456789/094-000131', '094-000131'),
    ('/data/scan 2020-094-000131', '094-000131'),
    # Other site prefixes aren't GRLS dogs.
    ('/data/095-000131', None),
    # Part of a longer number isn't an ID: these used to be read as dog 094-000131.
    ('/data/094-0001312', None),
    ('/data/1094-000131', None),
    ('/data/scan_2020-123094-000131', None),
    # Only ASCII digits count.
    ('/data/094-０００１３１', None),
    ('/data/no id here', None),
])
def test_find_subject_id(path, expected):
    """
    Tests which folder paths a subject ID is read from.

    Parameters:
        path : str
            The folder path.
        expected : str or None
            The subject ID that should be found, or None.

    Asserts:
        The expected ID is found, or none is.
    """
    assert find_subject_id(path) == expected


@pytest.mark.parametrize('text, expected', [
    ('094-000520', ('094', '000520')),
    ('123-456789', ('123', '456789')),
    # Whitespace around an ID, such as a line ending left on a value read from a file, is ignored.
    ('094-000520\n', ('094', '000520')),
    ('  094-000520 ', ('094', '000520')),
    # Anything else alongside the ID means the string isn't an ID.
    ('094-0005201x', None),
    ('x094-000520', None),
    ('094-00052', None),
    ('094000520', None),
    ('094-０００５２０', None),
])
def test_parse_id(text, expected):
    """
    Tests which strings count as exactly one subject ID.

    Parameters:
        text : str
            The candidate ID.
        expected : tuple or None
            The expected prefix and suffix, or None.

    Asserts:
        The string is split into prefix and suffix, or rejected.
    """
    assert parse_id(text) == expected


def test_ocr_subject_id_regex_is_unchanged():
    """
    Tests that building the PII scrubber's regex from the shared constants produces the regex it used before.

    Asserts:
        The regex is exactly the one the scrubber was written and tested with.
    """
    assert ocr_subject_id_regex() == r'(?<!\d)[0O]94[-– ]?0?\d{6}(?!\d)'


@pytest.mark.parametrize('text, expected', [
    ('ID 094-012345.', ['094-012345']),
    ('ID 094 012345 and O94–0012345', ['094 012345', 'O94–0012345']),
    ('ID 0940123456', ['0940123456']),
    ('fax 10940123456', []),
])
def test_ocr_subject_id_regex_matches(text, expected):
    """
    Tests the OCR forms of a subject ID the PII scrubber's regex finds.

    Parameters:
        text : str
            The OCR'd text.
        expected : list of str
            The IDs that should be found.

    Asserts:
        Exactly the expected IDs are found.
    """
    assert re.findall(ocr_subject_id_regex(), text) == expected
