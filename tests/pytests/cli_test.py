"""Tests for the shared CLI utilities."""

import argparse
import logging
import sys
from unittest import mock

import pytest

from scripts.metadata_miners import visit_date_miner
from vmrt_tesseract_utilities.cli import (EXIT_TOO_MANY_FAILURES,
                                          add_max_failure_rate_argument,
                                          normalize_flag_spelling,
                                          parse_failure_rate, run_exit_code)


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


@pytest.mark.parametrize(
    'count, failure_count, max_failure_rate, expected',
    [
        (0, 0, 1.0, 0),
        (5, 0, 1.0, 0),
        (5, 2, 1.0, 0),
        (5, 5, 1.0, EXIT_TOO_MANY_FAILURES),
        (1, 1, 1.0, EXIT_TOO_MANY_FAILURES),
        (5, 2, 0.3, EXIT_TOO_MANY_FAILURES),
        (5, 1, 0.3, 0),
        (10, 1, 0.1, 0),
        (5, 5, 0.0, EXIT_TOO_MANY_FAILURES),
        (5, 1, 0.0, EXIT_TOO_MANY_FAILURES),
    ],
)
def test_run_exit_code(count, failure_count, max_failure_rate, expected):
    """run_exit_code fails a run where every file, or more than max_failure_rate of them, failed."""
    assert run_exit_code(count, failure_count, max_failure_rate) == expected


def test_run_exit_code_logs_why_the_run_failed(caplog):
    """run_exit_code logs why a run is treated as failed."""
    with caplog.at_level(logging.ERROR, logger='import_utility'):
        run_exit_code(5, 5, 1.0)
        run_exit_code(5, 2, 0.3)
    assert 'All 5 files failed' in caplog.text
    assert '2 of 5 files failed, more than --max-failure-rate 0.3 allows' in caplog.text


@pytest.mark.parametrize('value, expected', [('0', 0.0), ('0.25', 0.25), ('1', 1.0)])
def test_parse_failure_rate(value, expected):
    """parse_failure_rate accepts a number from 0 to 1."""
    assert parse_failure_rate(value) == expected


@pytest.mark.parametrize('value', ['-0.1', '1.5', 'abc', 'nan'])
def test_parse_failure_rate_rejects_other_values(value):
    """parse_failure_rate rejects anything that isn't a number from 0 to 1."""
    with pytest.raises(argparse.ArgumentTypeError, match='is not a number from 0 to 1'):
        parse_failure_rate(value)


def test_max_failure_rate_argument():
    """--max-failure-rate defaults to 1, and the parser rejects a value over 1."""
    parser = argparse.ArgumentParser()
    add_max_failure_rate_argument(parser)
    assert parser.parse_args([]).max_failure_rate == 1.0
    assert parser.parse_args(['--max-failure-rate', '0.1']).max_failure_rate == 0.1
    with pytest.raises(SystemExit):
        parser.parse_args(['--max-failure-rate', '1.5'])
