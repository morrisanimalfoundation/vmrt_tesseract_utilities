import logging
import os
from argparse import Namespace

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from scripts import replace_strings
from vmrt_tesseract_utilities.database import (Base,
                                               StringReplacementCompletion,
                                               TranscriptionInput,
                                               TranscriptionOutput)

"""
Tests for what `replace_strings` does when files fail, go missing or the run stops part way through.
"""


@pytest.fixture
def db(monkeypatch):
    """
    An isolated in-memory database, used in place of MySQL.
    """
    engine = create_engine('sqlite:///:memory:')
    Base.metadata.create_all(engine)
    test_sessionmaker = sessionmaker(bind=engine)
    monkeypatch.setattr(replace_strings, 'get_database_session', lambda **kwargs: test_sessionmaker)
    return test_sessionmaker


def _add_outputs(db, tmp_path, names: list, create=True) -> list:
    """
    Adds a transcription output for each name, with an OCR text file unless create is False.
    """
    paths = []
    with db.begin() as session:
        for name in names:
            path = tmp_path / 'ocr' / f'{name}.txt'
            if create:
                path.parent.mkdir(exist_ok=True)
                path.write_text(f'{name} saw Max today.')
            session.add(TranscriptionOutput(transcription_input=TranscriptionInput(input_file=f'{name}.pdf',
                                                                                   document_type='document'),
                                            ocr_output_file=str(path)))
            paths.append(path)
    return paths


def _args(tmp_path, **overrides) -> Namespace:
    args = Namespace(data_file='names.tsv', key_column='name', replacement_string='<NAME>',
                     output_dir=str(tmp_path / 'out'), document_type='document', chunk_size=100, debug_sql=False,
                     max_workers=1, no_multiprocessing=True, until_done=False, retry_failed=False)
    vars(args).update(overrides)
    pass_hash, targets = replace_strings.make_pass_hash(['Max'], args.replacement_string)
    args.replacement_pass_id = replace_strings.get_or_create_pass(args, pass_hash, len(targets))
    return args


def _statuses(db) -> dict:
    with db() as session:
        return {os.path.basename(c_output.ocr_output_file): (c.status, c.error)
                for c, c_output in session.query(StringReplacementCompletion, TranscriptionOutput)
                .join(TranscriptionOutput, StringReplacementCompletion.output_id == TranscriptionOutput.id)}


def test_failed_file_is_recorded_skipped_and_retried(db, tmp_path):
    """
    Test that a failing file doesn't stop the run, is recorded as failed and skipped on the next run, and is
    processed again with --retry-failed once the problem is fixed.
    """
    good, broken = _add_outputs(db, tmp_path, ['good', 'broken'])
    broken.unlink()

    assert replace_strings.run_pass(_args(tmp_path), ['Max']) == replace_strings.EXIT_INCOMPLETE
    statuses = _statuses(db)
    assert statuses['good.txt'] == ('success', None)
    assert statuses['broken.txt'][0] == 'failed'
    assert statuses['broken.txt'][1].startswith('FileNotFoundError')

    # Still failed, so the pass isn't reported as done, but the file isn't tried again.
    assert replace_strings.get_files_to_process(_args(tmp_path)) == []
    assert replace_strings.run_pass(_args(tmp_path), ['Max']) == replace_strings.EXIT_INCOMPLETE

    broken.write_text('broken saw Max today.')
    assert replace_strings.run_pass(_args(tmp_path, retry_failed=True), ['Max']) == 0
    assert _statuses(db)['broken.txt'] == ('success', None)


def test_retrying_a_file_that_keeps_failing_ends(db, tmp_path, caplog):
    """
    Test that with --until-done and --retry-failed, a file that fails again is only tried once in the run.
    """
    (broken,) = _add_outputs(db, tmp_path, ['broken'], create=False)
    replace_strings.run_pass(_args(tmp_path), ['Max'])

    with caplog.at_level(logging.INFO, logger='import_utility'):
        result = replace_strings.run_pass(_args(tmp_path, retry_failed=True, until_done=True, chunk_size=1), ['Max'])
    assert result == replace_strings.EXIT_INCOMPLETE
    assert 'Replaced strings in 0 of 1 files.' in caplog.text


def test_until_done_processes_every_chunk(db, tmp_path):
    """
    Test that one chunk is processed by default, and every chunk with --until-done.
    """
    _add_outputs(db, tmp_path, ['a', 'b', 'c', 'd', 'e'])
    replace_strings.run_pass(_args(tmp_path, chunk_size=2), ['Max'])
    assert len(_statuses(db)) == 2

    assert replace_strings.run_pass(_args(tmp_path, chunk_size=2, until_done=True), ['Max']) == 0
    assert {status for status, _ in _statuses(db).values()} == {'success'}
    assert len(_statuses(db)) == 5


def test_missing_output_is_reported_and_others_still_processed(db, tmp_path, caplog):
    """
    Test that a completed output whose file was deleted is reported, written to the missing-file report and fails
    the run, while new files are still processed.
    """
    _add_outputs(db, tmp_path, ['first'])
    replace_strings.run_pass(_args(tmp_path), ['Max'])
    with db() as session:
        first_output = session.query(TranscriptionOutput).one().list_replacement_output_file
    os.remove(first_output)
    _add_outputs(db, tmp_path, ['second'])

    args = _args(tmp_path)
    with caplog.at_level(logging.INFO, logger='import_utility'):
        assert replace_strings.run_pass(args, ['Max']) == replace_strings.EXIT_INCOMPLETE

    assert _statuses(db)['second.txt'] == ('success', None)
    assert 'Replaced strings in 1 of 1 files.' in caplog.text
    assert '1 file(s) completed this pass, but their output file is missing' in caplog.text
    report = os.path.join(args.output_dir, f'missing_replacement_outputs_pass_{args.replacement_pass_id}.txt')
    with open(report) as f:
        assert f.read().split() == [first_output]


def test_failed_write_keeps_the_old_file(tmp_path):
    """
    Test that if writing the new text fails part way, the file keeps its old text and no temporary file is left.
    """
    output = tmp_path / 'record.txt'
    output.write_text('text from the earlier passes')

    with pytest.raises(TypeError):
        replace_strings._write_atomically(str(output), 12345)  # Writing a non-string fails.

    assert output.read_text() == 'text from the earlier passes'
    assert os.listdir(tmp_path) == ['record.txt']
