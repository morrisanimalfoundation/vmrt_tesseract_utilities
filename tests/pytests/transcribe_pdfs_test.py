import argparse
import logging
import sys

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from scripts import transcribe_pdfs
from vmrt_tesseract_utilities.database import (Base, TranscriptionInput,
                                               TranscriptionOutput)
from vmrt_tesseract_utilities.tesseract_operations import (NoTextFoundError,
                                                           OcrSetupError)

"""
Tests for the transcribe_pdfs script.
"""


@pytest.fixture
def engine(monkeypatch, tmp_path):
    """
    A SQLite database with one good, one blank and one broken PDF waiting to be transcribed.

    The OCR setup check is skipped, since poppler and Tesseract aren't installed in CI.
    """
    monkeypatch.setattr(transcribe_pdfs, 'check_ocr_setup', lambda: None)
    db_url = f'sqlite:///{tmp_path}/test.db'
    monkeypatch.setenv('SQL_CONNECTION_STRING', db_url)
    engine = create_engine(db_url)
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        for name in ('good.pdf', 'blank.pdf', 'broken.pdf'):
            session.add(TranscriptionInput(document_type='document', input_file=name))
        session.commit()
    return engine


def _fake_process_row(self, input_path):
    if input_path == 'blank.pdf':
        raise NoTextFoundError('No recognizable text found in blank.pdf.')
    if input_path == 'broken.pdf':
        raise ValueError('Unable to get page count.')
    return [{'content': 'text', 'confidence': 90, 'page': 1, 'block': 1}]


def _run(tmp_path, retry_failed=False, shard=None, document_type='document', offset=0):
    args = argparse.Namespace(output_to=str(tmp_path), document_type=document_type, chunk_size=10,
                              offset=offset, retry_failed=retry_failed, shard=shard, debug_sql=False)
    transcribe_pdfs.run_tesseract(args)


def _statuses(engine) -> dict:
    with Session(engine) as session:
        return {i.input_file: (i.ocr_status, i.ocr_error) for i in session.query(TranscriptionInput)}


def test_run_tesseract_skips_failures_and_summarizes(monkeypatch, tmp_path, caplog, engine):
    """
    Test that one failing file doesn't stop the run, and the failures are listed at the end.

    Asserts:
        The good file gets an output row and the failed files don't.
        Each input's status and error are saved.
        The summary counts the successes and names each failed file with its error.
    """
    monkeypatch.setattr(transcribe_pdfs.TesseractOperationDoc, 'process_row', _fake_process_row)
    with caplog.at_level(logging.INFO, logger='import_utility'):
        _run(tmp_path)

    with Session(engine) as session:
        outputs = session.query(TranscriptionOutput).all()
        assert [o.transcription_input.input_file for o in outputs] == ['good.pdf']
    assert _statuses(engine) == {
        'good.pdf': ('success', None),
        'blank.pdf': ('failed', 'NoTextFoundError: No recognizable text found in blank.pdf.'),
        'broken.pdf': ('failed', 'ValueError: Unable to get page count.'),
    }
    assert 'Transcribed 1 of 3 files.' in caplog.text
    assert '2 file(s) failed:' in caplog.text
    assert 'blank.pdf: NoTextFoundError: No recognizable text found in blank.pdf.' in caplog.text
    assert 'broken.pdf: ValueError: Unable to get page count.' in caplog.text


def test_write_error_is_recorded_as_a_failure(monkeypatch, tmp_path, caplog, engine):
    """
    Test that a text file that can't be written is recorded as a failure, and the run carries on.

    Asserts:
        No file gets an output row, each is saved as failed with the write error, and the summary still runs.
    """
    def unwritable(output_path, data):
        raise PermissionError(f'Permission denied: {output_path}')

    monkeypatch.setattr(transcribe_pdfs.TesseractOperationDoc, 'process_row',
                        lambda self, path: [{'content': 'text', 'confidence': 90, 'page': 1, 'block': 1}])
    monkeypatch.setattr(transcribe_pdfs, '_write_file', unwritable)
    with caplog.at_level(logging.INFO, logger='import_utility'):
        _run(tmp_path)

    with Session(engine) as session:
        assert session.query(TranscriptionOutput).count() == 0
    statuses = _statuses(engine)
    assert {status for status, _ in statuses.values()} == {'failed'}
    assert all(error.startswith('PermissionError: Permission denied') for _, error in statuses.values())
    assert 'Transcribed 0 of 3 files.' in caplog.text


def test_failed_files_are_only_retried_when_asked(monkeypatch, tmp_path, engine):
    """
    Test that files that failed are skipped on the next run, and retried with `--retry-failed`.

    Asserts:
        A plain second run processes nothing.
        A retry run processes only the failed files, and a success clears the saved error.
    """
    monkeypatch.setattr(transcribe_pdfs.TesseractOperationDoc, 'process_row', _fake_process_row)
    _run(tmp_path)

    seen = []

    def recording_process_row(self, input_path):
        seen.append(input_path)
        return [{'content': 'text', 'confidence': 90, 'page': 1, 'block': 1}]

    monkeypatch.setattr(transcribe_pdfs.TesseractOperationDoc, 'process_row', recording_process_row)
    _run(tmp_path)
    assert seen == []

    _run(tmp_path, retry_failed=True)
    assert sorted(seen) == ['blank.pdf', 'broken.pdf']
    assert set(_statuses(engine).values()) == {('success', None)}


class _Killed(BaseException):
    """
    Stands in for the process being killed part way through a PDF: like a SIGKILL, it isn't caught by the script's
    `except Exception`, so nothing after it is recorded, but what was already committed stays.
    """


def _killed_on(*paths, calls=None):
    """
    A process_row that's "killed" on the given PDFs and reads the others, recording every PDF it's called for.
    """
    def process_row(self, input_path):
        if calls is not None:
            calls.append(input_path)
        if input_path in paths:
            raise _Killed()
        return [{'content': 'text', 'confidence': 90, 'page': 1, 'block': 1}]
    return process_row


def test_pdf_is_marked_started_before_ocr(monkeypatch, tmp_path, engine):
    """
    Test that a PDF is saved as 'started' before OCR begins, so a kill part way through leaves a record of it.

    Asserts:
        While each PDF is being read, the database (seen from another session) has it as 'started'.
    """
    seen = {}

    def checking_process_row(self, input_path):
        seen[input_path] = _statuses(engine)[input_path][0]
        return [{'content': 'text', 'confidence': 90, 'page': 1, 'block': 1}]

    monkeypatch.setattr(transcribe_pdfs.TesseractOperationDoc, 'process_row', checking_process_row)
    _run(tmp_path)
    assert seen == {'good.pdf': 'started', 'blank.pdf': 'started', 'broken.pdf': 'started'}


def test_pdf_cut_short_twice_is_marked_failed(monkeypatch, tmp_path, caplog, engine):
    """
    Test that a PDF whose OCR is cut short is tried once more, then marked failed, so it can't stop every run.

    Asserts:
        After the first kill it's 'started'; the next run retries it and leaves it 'retrying' when killed again;
        the run after that marks it failed with the interrupted error without reading it, and carries on with the
        other PDFs; after that it's skipped.
    """
    calls = []
    monkeypatch.setattr(transcribe_pdfs.TesseractOperationDoc, 'process_row', _killed_on('blank.pdf', calls=calls))
    with pytest.raises(_Killed):
        _run(tmp_path)
    assert _statuses(engine) == {'good.pdf': ('success', None), 'blank.pdf': ('started', None),
                                 'broken.pdf': (None, None)}

    with caplog.at_level(logging.INFO, logger='import_utility'), pytest.raises(_Killed):
        _run(tmp_path)
    assert 'blank.pdf: an earlier run was cut short while reading it, so it is tried once more.' in caplog.text
    assert _statuses(engine)['blank.pdf'] == ('retrying', None)
    assert calls == ['good.pdf', 'blank.pdf', 'blank.pdf']

    calls.clear()
    _run(tmp_path)
    assert calls == ['broken.pdf']
    assert _statuses(engine) == {'good.pdf': ('success', None),
                                 'blank.pdf': ('failed', transcribe_pdfs.INTERRUPTED_TWICE_ERROR),
                                 'broken.pdf': ('success', None)}

    calls.clear()
    _run(tmp_path)
    assert calls == []


def test_pdf_cut_short_once_succeeds_on_retry(monkeypatch, tmp_path, engine):
    """
    Test that a PDF cut short once, e.g. by a stopped build, is read normally on the next run.

    Asserts:
        It ends as 'success' with no error, and gets its output row.
    """
    monkeypatch.setattr(transcribe_pdfs.TesseractOperationDoc, 'process_row', _killed_on('good.pdf'))
    with pytest.raises(_Killed):
        _run(tmp_path)
    monkeypatch.setattr(transcribe_pdfs.TesseractOperationDoc, 'process_row', _killed_on())
    _run(tmp_path)
    assert set(_statuses(engine).values()) == {('success', None)}
    with Session(engine) as session:
        assert session.query(TranscriptionOutput).count() == 3


def test_retry_failed_tries_an_interrupted_pdf_again(monkeypatch, tmp_path, engine):
    """
    Test that --retry-failed tries a PDF that was marked failed for being cut short twice, from a fresh start.

    Asserts:
        It's read again and succeeds.
    """
    _set_status(engine, 'blank.pdf', 'failed', transcribe_pdfs.INTERRUPTED_TWICE_ERROR)
    calls = []
    monkeypatch.setattr(transcribe_pdfs.TesseractOperationDoc, 'process_row', _killed_on(calls=calls))
    _run(tmp_path, retry_failed=True)
    assert 'blank.pdf' in calls
    assert _statuses(engine)['blank.pdf'] == ('success', None)


def _set_status(engine, input_file: str, status: str, error: str = None) -> None:
    with Session(engine) as session:
        session.query(TranscriptionInput).filter_by(input_file=input_file).update(
            {'ocr_status': status, 'ocr_error': error})
        session.commit()


def test_retrying_a_failed_pdf_clears_its_old_error(monkeypatch, tmp_path, engine):
    """
    Test that a failed PDF retried with --retry-failed loses its old error while it's being read, so a run killed
    part way through doesn't leave 'started' next to an error from an earlier attempt.

    Asserts:
        While it's read, it's ('started', None).
    """
    _set_status(engine, 'broken.pdf', 'failed', 'ValueError: Unable to get page count.')
    seen = {}

    def checking_process_row(self, input_path):
        seen[input_path] = _statuses(engine)[input_path]
        return [{'content': 'text', 'confidence': 90, 'page': 1, 'block': 1}]

    monkeypatch.setattr(transcribe_pdfs.TesseractOperationDoc, 'process_row', checking_process_row)
    _run(tmp_path, retry_failed=True)
    assert seen['broken.pdf'] == ('started', None)


def test_retry_failed_reads_a_retrying_pdf(monkeypatch, tmp_path, engine):
    """
    Test that --retry-failed reads a PDF left 'retrying', rather than marking it failed unread like a plain run.

    Asserts:
        It's read and succeeds.
    """
    _set_status(engine, 'blank.pdf', 'retrying')
    calls = []
    monkeypatch.setattr(transcribe_pdfs.TesseractOperationDoc, 'process_row', _killed_on(calls=calls))
    _run(tmp_path, retry_failed=True)
    assert 'blank.pdf' in calls
    assert _statuses(engine)['blank.pdf'] == ('success', None)


def test_setup_error_stops_run_before_any_file(monkeypatch, tmp_path, engine):
    """
    Test that a setup problem stops the run instead of failing every file one by one.

    Asserts:
        OcrSetupError is raised, no file is processed, and no file is marked as failed.
    """
    def broken_setup():
        raise OcrSetupError('Poppler is not installed or not on PATH (missing pdfinfo).')

    def unexpected_process_row(self, input_path):
        raise AssertionError('No file should be processed when setup is broken.')

    monkeypatch.setattr(transcribe_pdfs, 'check_ocr_setup', broken_setup)
    monkeypatch.setattr(transcribe_pdfs.TesseractOperationDoc, 'process_row', unexpected_process_row)
    with pytest.raises(OcrSetupError, match='Poppler'):
        _run(tmp_path)
    assert set(_statuses(engine).values()) == {(None, None)}


def _recorder(seen):
    def recording_process_row(self, input_path):
        seen.append(input_path)
        return [{'content': 'text', 'confidence': 90, 'page': 1, 'block': 1}]
    return recording_process_row


def test_shards_split_the_inputs_without_overlap(monkeypatch, tmp_path, engine):
    """
    Test that runs with --shard 0/2 and 1/2 between them transcribe every input exactly once.

    The fixture's inputs have IDs 1 (good.pdf), 2 (blank.pdf) and 3 (broken.pdf). Shard 1/2 runs first, so shard
    0/2 runs after those files are done: with --offset that would move its window, with --shard it must not.

    Asserts:
        Shard 1/2 takes IDs 1 and 3, shard 0/2 takes ID 2, a second round takes nothing, and every input succeeded.
    """
    seen = []
    monkeypatch.setattr(transcribe_pdfs.TesseractOperationDoc, 'process_row', _recorder(seen))
    _run(tmp_path, shard=(1, 2))
    assert sorted(seen) == ['broken.pdf', 'good.pdf']
    seen.clear()
    _run(tmp_path, shard=(0, 2))
    assert seen == ['blank.pdf']
    seen.clear()
    _run(tmp_path, shard=(0, 2))
    _run(tmp_path, shard=(1, 2))
    assert seen == []
    assert set(_statuses(engine).values()) == {('success', None)}


def test_no_shard_takes_every_input(monkeypatch, tmp_path, engine):
    """
    Test that a run without --shard still transcribes every input.

    Asserts:
        All three inputs are processed.
    """
    seen = []
    monkeypatch.setattr(transcribe_pdfs.TesseractOperationDoc, 'process_row', _recorder(seen))
    _run(tmp_path)
    assert sorted(seen) == ['blank.pdf', 'broken.pdf', 'good.pdf']


@pytest.mark.parametrize('value, expected', [('0/1', (0, 1)), ('3/4', (3, 4)), ('10/12', (10, 12))])
def test_parse_shard_accepts_valid_values(value, expected):
    """
    Test that "i/N" values with 0 <= i < N are parsed.

    Asserts:
        The value is returned as (i, N).
    """
    assert transcribe_pdfs.parse_shard(value) == expected


@pytest.mark.parametrize('value, message', [
    ('2', 'expected i/N'),
    ('a/4', 'expected i/N'),
    ('1/2/3', 'expected i/N'),
    ('-1/4', 'expected i/N'),
    # int() accepts all of these; --shard shouldn't.
    ('+1/2', 'expected i/N'),
    (' 1 / 2 ', 'expected i/N'),
    ('1_0/2_0', 'expected i/N'),
    ('0/0', 'at least 1'),
    ('4/4', 'from 0 to 3'),
])
def test_parse_shard_rejects_invalid_values(value, message):
    """
    Test that a --shard value that isn't "i/N" with 0 <= i < N is rejected with a clear message.

    Asserts:
        argparse.ArgumentTypeError is raised, naming the problem.
    """
    with pytest.raises(argparse.ArgumentTypeError, match=message):
        transcribe_pdfs.parse_shard(value)


def test_shard_flag_is_parsed(monkeypatch):
    """
    Test that --shard reaches the parsed args as (i, N), and defaults to None.

    Asserts:
        `--shard 1/3` gives (1, 3); no flag gives None; a bad value exits with an argparse error.
    """
    monkeypatch.setattr(sys, 'argv', ['transcribe_pdfs.py', 'output', '--shard', '1/3'])
    assert transcribe_pdfs.parse_args().shard == (1, 3)
    monkeypatch.setattr(sys, 'argv', ['transcribe_pdfs.py', 'output'])
    assert transcribe_pdfs.parse_args().shard is None
    monkeypatch.setattr(sys, 'argv', ['transcribe_pdfs.py', 'output', '--shard', '3/3'])
    with pytest.raises(SystemExit):
        transcribe_pdfs.parse_args()


def _shard_database(monkeypatch, tmp_path, document_type, count):
    """
    A SQLite database with `count` PDFs of one document type waiting to be transcribed, IDs 1 to `count`.
    """
    monkeypatch.setattr(transcribe_pdfs, 'check_ocr_setup', lambda: None)
    db_url = f'sqlite:///{tmp_path}/shards.db'
    monkeypatch.setenv('SQL_CONNECTION_STRING', db_url)
    engine = create_engine(db_url)
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        session.add_all(TranscriptionInput(document_type=document_type, input_file=f'{n}.pdf')
                        for n in range(1, count + 1))
        session.commit()
    return engine


@pytest.mark.parametrize('shard_count', [1, 2, 3, 4, 5])
def test_shards_together_take_every_input_exactly_once(monkeypatch, tmp_path, shard_count):
    """
    Test that for each N, the shards 0/N to N-1/N between them transcribe all 23 inputs, each exactly once.

    A run takes at most 10 files (the chunk size), so each shard is rerun until a round finds nothing, as the
    README says to.

    Asserts:
        No input is processed twice, and every input is processed.
    """
    _shard_database(monkeypatch, tmp_path, 'document', 23)
    seen = []
    monkeypatch.setattr(transcribe_pdfs.TesseractOperationDoc, 'process_row', _recorder(seen))
    for shard_index in range(shard_count):
        while True:
            before = len(seen)
            _run(tmp_path, shard=(shard_index, shard_count))
            if len(seen) == before:
                break
    assert len(seen) == len(set(seen))
    assert sorted(seen) == sorted(f'{n}.pdf' for n in range(1, 24))


def test_page_inputs_keep_all_their_pages_in_one_shard(monkeypatch, tmp_path):
    """
    Test sharding in page mode, where one input becomes several output rows.

    The shard is chosen by the input's ID, so all of a PDF's pages are read by the same run.

    Asserts:
        Each input is read once, by the shard its ID belongs to, and ends up with one output row per page.
    """
    engine = _shard_database(monkeypatch, tmp_path, 'page', 4)
    seen = []

    def two_pages(self, input_path):
        seen.append(input_path)
        return [{'content': f'page {page}', 'confidence': 90, 'page': page, 'block': 1} for page in (1, 2)]

    monkeypatch.setattr(transcribe_pdfs.TesseractOperationPage, 'process_row', two_pages)
    _run(tmp_path, shard=(0, 2), document_type='page')
    assert sorted(seen) == ['2.pdf', '4.pdf']
    _run(tmp_path, shard=(1, 2), document_type='page')
    assert sorted(seen) == ['1.pdf', '2.pdf', '3.pdf', '4.pdf']
    with Session(engine) as session:
        pages = {i.input_file: len(i.assets) for i in session.query(TranscriptionInput)}
    assert pages == {'1.pdf': 2, '2.pdf': 2, '3.pdf': 2, '4.pdf': 2}


def test_offset_counts_within_the_shard(monkeypatch, tmp_path):
    """
    Test that --offset with --shard skips files of that shard only.

    Asserts:
        Shard 0/2 of IDs 1 to 6 is 2, 4 and 6; with --offset 1 the run takes 4 and 6.
    """
    _shard_database(monkeypatch, tmp_path, 'document', 6)
    seen = []
    monkeypatch.setattr(transcribe_pdfs.TesseractOperationDoc, 'process_row', _recorder(seen))
    _run(tmp_path, shard=(0, 2), offset=1)
    assert sorted(seen) == ['4.pdf', '6.pdf']


@pytest.mark.parametrize('mode, results', [
    ('page', [
        {'content': 'Page one', 'confidence': 91.0, 'page': 1, 'block': 0},
        {'content': 'Page two', 'confidence': 92.0, 'page': 2, 'block': 0},
    ]),
    ('block', [
        {'content': 'P1 block 1', 'confidence': 81.0, 'page': 1, 'block': 1},
        {'content': 'P1 block 2', 'confidence': 82.0, 'page': 1, 'block': 2},
        {'content': 'P1 block 3', 'confidence': 83.0, 'page': 1, 'block': 3},
        {'content': 'P2 block 1', 'confidence': 84.0, 'page': 2, 'block': 1},
        {'content': 'P2 block 2', 'confidence': 85.0, 'page': 2, 'block': 2},
        {'content': 'P2 block 3', 'confidence': 86.0, 'page': 2, 'block': 3},
    ]),
])
@pytest.mark.parametrize('has_existing_empty_row', [False, True])
def test_every_ocr_result_has_its_own_database_row(monkeypatch, tmp_path, mode, results, has_existing_empty_row):
    """
    Test that in page and block mode each OCR result is saved as its own output row (#49).

    Parameters:
        mode : str
            The document type, page or block.
        results : list of dict
            The OCR results the faked operation returns.
        has_existing_empty_row : bool
            Whether the input already has an empty output row, which the first result should fill.

    Asserts:
        There is one row per result, each pointing to its own file with that result's text and confidence, an
        existing empty row is reused, and the input is saved as transcribed successfully.
    """
    engine = _shard_database(monkeypatch, tmp_path, mode, 1)
    old_row_id = None
    with Session(engine) as session:
        input_row = session.query(TranscriptionInput).one()
        if has_existing_empty_row:
            old_row = TranscriptionOutput(transcription_input=input_row)
            session.add(old_row)
            session.commit()
            old_row_id = old_row.id
        input_id, input_file = input_row.id, input_row.input_file

    def fake_process_row(self, input_path):
        assert input_path == input_file
        return results

    operation = transcribe_pdfs.TesseractOperationPage if mode == 'page' else transcribe_pdfs.TesseractOperationBlock
    monkeypatch.setattr(operation, 'process_row', fake_process_row)
    _run(tmp_path, document_type=mode)

    assert _statuses(engine) == {input_file: ('success', None)}
    with Session(engine) as session:
        stored_rows = session.query(TranscriptionOutput).order_by(TranscriptionOutput.id).all()
        assert len(stored_rows) == len(results)
        assert len({row.ocr_output_file for row in stored_rows}) == len(results)
        assert {row.ocr_confidence for row in stored_rows} == {result['confidence'] for result in results}
        assert all(row.transcription_input.input_file == input_file for row in stored_rows)
        if has_existing_empty_row:
            assert old_row_id in {row.id for row in stored_rows}

        by_file = {row.ocr_output_file: row for row in stored_rows}
        for result in results:
            file_path = tmp_path.joinpath('unstructured_text', mode,
                                          f"{input_id}-1-{result['page']}-{result['block']}.txt")
            assert str(file_path) in by_file
            assert by_file[str(file_path)].ocr_confidence == result['confidence']
            assert file_path.read_text() == result['content']
