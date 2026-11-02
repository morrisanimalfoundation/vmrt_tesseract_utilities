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
        first = session.query(TranscriptionOutput).one()
        first_id, first_output = first.id, first.list_replacement_output_file
    os.remove(first_output)
    _add_outputs(db, tmp_path, ['second'])

    args = _args(tmp_path)
    with caplog.at_level(logging.INFO, logger='import_utility'):
        assert replace_strings.run_pass(args, ['Max']) == replace_strings.EXIT_INCOMPLETE

    assert _statuses(db)['second.txt'] == ('success', None)
    assert 'Replaced strings in 1 of 1 files.' in caplog.text
    assert '1 file(s) completed this pass, but their output file is missing' in caplog.text
    assert "Don't start step 4" in caplog.text
    report = os.path.join(args.output_dir, f'missing_replacement_outputs_pass_{args.replacement_pass_id}.txt')
    with open(report) as f:
        assert f.read() == f'output {first_id}: {first_output}\n'

    # Once the file is back, the next run finds nothing missing and removes the old report.
    with open(first_output, 'w') as f:
        f.write('rebuilt')
    assert replace_strings.run_pass(args, ['Max']) == 0
    assert not os.path.exists(report)


def test_missing_check_waits_until_nothing_is_left(db, tmp_path, monkeypatch):
    """
    Test that completed outputs are only checked once no files are left to process, not on every chunk's run.
    """
    _add_outputs(db, tmp_path, ['a', 'b', 'c'])
    checks = []
    monkeypatch.setattr(replace_strings, 'find_missing_outputs', lambda args: checks.append(1) or [])

    replace_strings.run_pass(_args(tmp_path, chunk_size=2), ['Max'])
    assert checks == []  # A full chunk: more files may be waiting.
    replace_strings.run_pass(_args(tmp_path, chunk_size=2), ['Max'])
    assert checks == [1]


def test_written_files_keep_normal_permissions(tmp_path):
    """
    Test that a new file gets the permissions a plain write would give it, and a rewritten one keeps its own,
    rather than the owner-only permissions temporary files are created with.
    """
    umask = os.umask(0)
    os.umask(umask)
    new_file = tmp_path / 'new.txt'
    replace_strings._write_atomically(str(new_file), 'text')
    assert os.stat(new_file).st_mode & 0o777 == 0o666 & ~umask

    existing = tmp_path / 'existing.txt'
    existing.write_text('old')
    os.chmod(existing, 0o640)
    replace_strings._write_atomically(str(existing), 'new')
    assert os.stat(existing).st_mode & 0o777 == 0o640


def test_failed_sync_keeps_the_old_file_and_cleans_up(tmp_path, monkeypatch):
    """
    Test that an error saving the new text to disk (such as a full disk) keeps the old file and leaves no
    temporary file behind.
    """
    output = tmp_path / 'record.txt'
    output.write_text('text from the earlier passes')

    def disk_full(fd):
        raise OSError(28, 'No space left on device')

    monkeypatch.setattr(replace_strings.os, 'fsync', disk_full)
    with pytest.raises(OSError, match='No space left'):
        replace_strings._write_atomically(str(output), 'new text')

    assert output.read_text() == 'text from the earlier passes'
    assert os.listdir(tmp_path) == ['record.txt']


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
