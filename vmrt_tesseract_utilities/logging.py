import logging

"""
A simple logger configured to write output to stdout.
"""

stdout_logger = logging.getLogger('import_utility')
formatter = logging.Formatter('%(asctime)s: %(message)s')
stdout_logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)
stdout_logger.addHandler(ch)


def log_run_summary(verb: str, count: int, failures: list[tuple[str, str]], note: str = '') -> None:
    """
    Logs how many of a run's files succeeded, and lists each failure.

    Parameters
    ----------
    verb : str
        What was done to the files, in the past tense, e.g. 'Transcribed'.
    count : int
        The number of files the run tried.
    failures : list[tuple[str, str]]
        (file, error) for each file that failed.
    note : str
        Added to the line before the list of failures, e.g. ' and were skipped'.
    """
    stdout_logger.info(f'All done! {verb} {count - len(failures)} of {count} files.')
    if failures:
        stdout_logger.warning(f'{len(failures)} file(s) failed{note}:')
        for path, error in failures:
            stdout_logger.warning(f'  {path}: {error}')
