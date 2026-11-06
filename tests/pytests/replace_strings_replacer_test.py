import multiprocessing
import os
from argparse import Namespace
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from scripts import replace_strings
from vmrt_tesseract_utilities.database import (Base, TranscriptionInput,
                                               TranscriptionOutput)
from vmrt_tesseract_utilities.string_replacer import StringReplacer

"""
Tests that `replace_strings` builds its StringReplacer once per process instead of once per file, and that reusing
it doesn't change what gets replaced.
"""


@pytest.fixture
def builds(monkeypatch):
    """
    Counts the StringReplacers built, and starts each test with no replacer cached.
    """
    monkeypatch.setattr(replace_strings, '_replacer_cache', None)
    built = []

    class CountingReplacer(StringReplacer):
        def __init__(self, *args, **kwargs):
            built.append(args)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(replace_strings, 'StringReplacer', CountingReplacer)
    return built


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


def _output_log(path: Path) -> MagicMock:
    """
    A transcription output for an OCR text file that hasn't been through any replacement pass.
    """
    output_log = MagicMock()
    output_log.ocr_output_file = str(path)
    output_log.list_replacement_output_file = None
    return output_log


def test_get_replacer_builds_once_for_the_same_inputs(builds):
    """
    Test that asking for the same targets and replacement again returns the replacer already built, even when the
    list is a different object, as it is in a worker that was sent a copy.
    """
    first = replace_strings.get_replacer(['Max'], '<NAME>')
    second = replace_strings.get_replacer(['Max'], '<NAME>')

    assert second is first
    assert len(builds) == 1


def test_get_replacer_rebuilds_when_the_targets_or_replacement_change(builds):
    """
    Test that different targets or a different replacement string get a replacer of their own, so a cached one is
    never used for inputs it wasn't built from.
    """
    assert replace_strings.get_replacer(['Max'], '<NAME>').replace('Max met Rex') == '<NAME> met Rex'
    assert replace_strings.get_replacer(['Rex'], '<NAME>').replace('Max met Rex') == 'Max met <NAME>'
    assert replace_strings.get_replacer(['Rex'], '<ID>').replace('Max met Rex') == 'Max met <ID>'

    assert len(builds) == 3


@pytest.mark.parametrize('targets, text', [
    (['Max', 'Maxwell'], 'Maxwell and Max and max'),
    (['Maxwell', 'Max'], 'Maxwell and Max and max'),
    (['094-000520', 'Rex'], 'IDs 094-000520, 094000520 and 094-0000520, but not 1094-0005201, and rex'),
    (['', ' ', 'Rex'], 'Rex  rex'),
])
def test_reused_replacer_gives_the_same_text_as_a_fresh_one(builds, targets, text):
    """
    Test that a replacer taken from the cache replaces exactly what a newly built one does, including when the
    order of the targets matters, for ID variants and for blank targets, even with a different replacer cached first.
    """
    expected = StringReplacer(targets, '<NAME>').replace(text)
    assert expected != text
    replace_strings.get_replacer(['something else'], '<NAME>')

    for _ in range(2):
        assert replace_strings.get_replacer(targets, '<NAME>').replace(text) == expected


def test_process_file_builds_the_replacer_once_for_many_files(builds, tmp_path):
    """
    Test that processing several files with the same targets builds one replacer, and each file still has its
    targets replaced.
    """
    parsed_args = Namespace(replacement_string='<NAME>', output_dir=str(tmp_path / 'out'), document_type='document')
    results = {}
    for name in ['a', 'b', 'c']:
        source = tmp_path / f'{name}.txt'
        source.write_text(f'{name} saw Max today.')
        results[name] = replace_strings.process_file(_output_log(source), ['Max'], parsed_args)

    assert len(builds) == 1
    for name, result in results.items():
        assert Path(result.list_replacement_output_file).read_text() == f'{name} saw <NAME> today.'


def test_make_pass_hash_leaves_its_replacer_for_the_files_to_reuse(builds):
    """
    Test that the replacer built to work out the pass hash is the one the files use, so a run builds one in the
    parent instead of another for its files.
    """
    targets = ['Max', '', 'Maxwell']
    replace_strings.make_pass_hash(targets, '<NAME>')
    replace_strings.get_replacer(list(targets), '<NAME>')

    assert len(builds) == 1


def test_pass_hash_is_the_same_with_a_cached_replacer(builds):
    """
    Test that the pass hash and effective targets don't depend on whether the replacer was already cached.
    """
    first = replace_strings.make_pass_hash(['Max', '', 'Maxwell'], '<NAME>')
    second = replace_strings.make_pass_hash(['Max', '', 'Maxwell'], '<NAME>')

    assert first == second
    assert len(builds) == 1


def test_get_replacer_accepts_targets_that_can_only_be_read_once(builds):
    """
    Test that targets given as a one-shot iterable, such as a generator, still give a replacer with every target.
    """
    replacer = replace_strings.get_replacer((target for target in ['Max', 'Rex']), '<NAME>')

    assert replacer.replace('Max met Rex') == '<NAME> met <NAME>'


def test_a_failed_build_is_not_cached(monkeypatch):
    """
    Test that a replacer that fails to build isn't kept, so the next call tries again instead of failing or
    returning half of one.
    """
    monkeypatch.setattr(replace_strings, '_replacer_cache', None)
    attempts = []

    class FlakyReplacer(StringReplacer):
        def __init__(self, *args, **kwargs):
            attempts.append(args)
            if len(attempts) == 1:
                raise RuntimeError('the first build fails')
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(replace_strings, 'StringReplacer', FlakyReplacer)

    with pytest.raises(RuntimeError):
        replace_strings.get_replacer(['Max'], '<NAME>')

    assert replace_strings.get_replacer(['Max'], '<NAME>').replace('Max') == '<NAME>'
    assert len(attempts) == 2


def test_changing_the_targets_make_pass_hash_returns_does_not_change_the_cached_replacer(builds):
    """
    Test that the list make_pass_hash returns is a copy, so changing it can't change the replacer the files reuse.
    """
    _, targets = replace_strings.make_pass_hash(['Max', 'Maxwell'], '<NAME>')
    targets.append('Rex')

    assert replace_strings.get_replacer(['Max', 'Maxwell'], '<NAME>').target_strings == ['Max', 'Maxwell']
    assert len(builds) == 1


def test_a_run_builds_one_replacer_for_all_its_files(db, tmp_path, builds):
    """
    Test that a whole run over several files, the way the script starts one, builds a single replacer.
    """
    with db.begin() as session:
        for name in ['a', 'b', 'c', 'd', 'e']:
            path = tmp_path / f'{name}.txt'
            path.write_text(f'{name} saw Max today.')
            session.add(TranscriptionOutput(transcription_input=TranscriptionInput(input_file=f'{name}.pdf',
                                                                                   document_type='document'),
                                            ocr_output_file=str(path)))
    args = Namespace(data_file='names.tsv', key_column='name', replacement_string='<NAME>',
                     output_dir=str(tmp_path / 'out'), document_type='document', chunk_size=100, debug_sql=False,
                     max_workers=1, no_multiprocessing=True, until_done=False, retry_failed=False)
    pass_hash, targets = replace_strings.make_pass_hash(['Max'], args.replacement_string)
    args.replacement_pass_id = replace_strings.get_or_create_pass(args, pass_hash, len(targets))

    assert replace_strings.run_pass(args, ['Max']) == 0

    assert len(builds) == 1
    replaced = sorted((tmp_path / 'out' / 'list_replacement_output_file' / 'document').glob('*.txt'))
    assert [path.read_text() for path in replaced] == [f'{name} saw <NAME> today.' for name in 'abcde']


def _build_report(targets, replacement_string):
    """
    Runs in a worker: the worker's process id, and whether this call built a replacer instead of reusing one.
    """
    before = replace_strings._replacer_cache
    replace_strings.get_replacer(targets, replacement_string)
    return os.getpid(), replace_strings._replacer_cache is not before


def _builds_by_worker(start_method, max_workers=2, tasks=8):
    """
    Calls get_replacer from several workers started with the given method, once per task, with the same targets
    each time, as the files of a run are sent to the workers.

    Returns
    -------
    dict
        For each worker that took a task, how many replacers it built.
    """
    context = multiprocessing.get_context(start_method)
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=context) as executor:
        results = list(executor.map(_build_report, [['Max', 'Maxwell']] * tasks, ['<NAME>'] * tasks))
    builds = {}
    for pid, built in results:
        builds[pid] = builds.get(pid, 0) + built
    return builds


@pytest.mark.skipif('fork' not in multiprocessing.get_all_start_methods(), reason='needs the fork start method')
def test_forked_workers_build_nothing_when_the_parent_built_the_replacer(monkeypatch):
    """
    Test that workers forked after make_pass_hash, the way a run starts them, inherit its replacer and so build
    none of their own.
    """
    monkeypatch.setattr(replace_strings, '_replacer_cache', None)
    replace_strings.make_pass_hash(['Max', 'Maxwell'], '<NAME>')

    builds = _builds_by_worker('fork')

    assert builds
    assert set(builds.values()) == {0}


@pytest.mark.parametrize('start_method', [method for method in ('fork', 'spawn')
                                          if method in multiprocessing.get_all_start_methods()])
def test_each_worker_builds_the_replacer_once_when_the_parent_has_not(monkeypatch, start_method):
    """
    Test that when no replacer was built beforehand, each worker builds exactly one and reuses it for the rest of its
    files, whichever way the workers are started.
    """
    monkeypatch.setattr(replace_strings, '_replacer_cache', None)

    builds = _builds_by_worker(start_method)

    assert builds
    assert set(builds.values()) == {1}


def test_spawned_workers_build_their_own_replacer_even_when_the_parent_built_one(monkeypatch):
    """
    Test that workers started fresh, as on macOS, don't get the parent's replacer, so each builds one, once. The
    saving there comes from building once per worker instead of once per file.
    """
    monkeypatch.setattr(replace_strings, '_replacer_cache', None)
    replace_strings.make_pass_hash(['Max', 'Maxwell'], '<NAME>')

    builds = _builds_by_worker('spawn')

    assert builds
    assert set(builds.values()) == {1}
