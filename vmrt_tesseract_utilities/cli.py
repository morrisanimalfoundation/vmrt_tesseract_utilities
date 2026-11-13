"""
Shared CLI utilities for consistent argument parsing across scripts.
"""

import argparse

from vmrt_tesseract_utilities.logging import stdout_logger

# Exit code when a run finished but too many of its files failed, the same code replace_strings.py uses for a run
# that finished with failures, so CI (e.g. Jenkins) can tell it apart from the script crashing (exit code 1).
EXIT_TOO_MANY_FAILURES = 3


def normalize_flag_spelling(argv: list[str]) -> list[str]:
    """Rewrite '--foo_bar' (and '--foo_bar=value') to '--foo-bar' so either spelling works.

    argparse accepts any unambiguous prefix of a flag, so when both '--foo_bar' and
    '--foo-bar' are defined, shortened forms like '--foo' become ambiguous. This
    normalizer rewrites underscore spellings to hyphens before parsing, letting us
    define only the hyphen form while keeping backwards compatibility.
    """
    normalized = []
    for i, arg in enumerate(argv):
        if arg == '--':
            return normalized + argv[i:]
        if arg.startswith('--'):
            name, sep, value = arg.partition('=')
            arg = name.replace('_', '-') + sep + value
        normalized.append(arg)
    return normalized


def parse_failure_rate(value: str) -> float:
    """Parse a --max-failure-rate value, a share of files from 0 to 1.

    Parameters
    ----------
    value : str
        The value given on the command line.

    Returns
    -------
    float
        The share of files.

    Raises
    ------
    argparse.ArgumentTypeError
        If the value isn't a number from 0 to 1.
    """
    try:
        rate = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f'{value!r} is not a number from 0 to 1')
    if not 0 <= rate <= 1:
        raise argparse.ArgumentTypeError(f'{value!r} is not a number from 0 to 1')
    return rate


def add_max_failure_rate_argument(parser: argparse.ArgumentParser) -> None:
    """Add --max-failure-rate to a script that records failed files and carries on.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The script's parser.
    """
    parser.add_argument('--max-failure-rate', type=parse_failure_rate, default=1.0, metavar='RATE',
                        help=f'Exit with code {EXIT_TOO_MANY_FAILURES} when more than this share of the files tried '
                             f'(0 to 1) failed. A run where every file failed always exits with '
                             f'{EXIT_TOO_MANY_FAILURES}. Default: 1, so only that run does.')


def run_exit_code(count: int, failure_count: int, max_failure_rate: float) -> int:
    """Work out the exit code of a run that records failed files and carries on.

    A run that tried no files succeeded. One where every file failed, or more than max_failure_rate of them, is
    treated as failed, since it usually has one cause (a broken model, a bad config, a full disk) rather than a few
    bad files.

    Parameters
    ----------
    count : int
        The number of files the run tried.
    failure_count : int
        The number of those that failed.
    max_failure_rate : float
        The largest share of files, from 0 to 1, that can fail with the run still exiting with 0.

    Returns
    -------
    int
        0, or EXIT_TOO_MANY_FAILURES if too many files failed.
    """
    if not count:
        return 0
    if failure_count == count:
        stdout_logger.error(f'All {count} files failed, so the run is treated as failed '
                            f'(exit code {EXIT_TOO_MANY_FAILURES}).')
        return EXIT_TOO_MANY_FAILURES
    if failure_count / count > max_failure_rate:
        stdout_logger.error(f'{failure_count} of {count} files failed, more than --max-failure-rate '
                            f'{max_failure_rate} allows, so the run is treated as failed '
                            f'(exit code {EXIT_TOO_MANY_FAILURES}).')
        return EXIT_TOO_MANY_FAILURES
    return 0
