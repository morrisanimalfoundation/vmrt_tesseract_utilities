"""Tests for the shared CLI utilities."""

import sys
from unittest import mock

import pytest

from scripts.metadata_miners import visit_date_miner
from vmrt_tesseract_utilities.cli import normalize_flag_spelling


def test_normalize_flag_spelling():
    """normalize_flag_spelling rewrites underscore flags to hyphens."""
    argv = [
        'out',
        '--chunk_size',
        '5',
        '--visit_date_tsv=data/a_b.tsv',
        '-h',
        '--',
        '--keep_me',
    ]
    assert normalize_flag_spelling(argv) == [
        'out',
        '--chunk-size',
        '5',
        '--visit-date-tsv=data/a_b.tsv',
        '-h',
        '--',
        '--keep_me',
    ]


@pytest.mark.parametrize(
    'flags',
    [
        ['--chunk-size', '5', '--debug-sql'],
        ['--chunk_size', '5', '--debug_sql'],
        ['--chunk', '5', '--debug'],
    ],
)
def test_visit_date_miner_flag_spellings(flags):
    """visit_date_miner accepts hyphen, underscore, and shortened flag spellings."""
    argv = [
        'visit_date_miner.py',
        'out',
        '--visit-date-tsv=v.tsv',
        '--dog-profile-tsv=d.tsv',
        *flags,
    ]
    with mock.patch.object(sys, 'argv', argv):
        args = visit_date_miner.parse_args()
    assert args.chunk_size == 5
    assert args.debug_sql is True
