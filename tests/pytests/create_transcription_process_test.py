import logging
import os
from argparse import Namespace

import pytest
from sqlalchemy import MetaData, String, create_engine, event
from sqlalchemy.exc import DataError, OperationalError
from sqlalchemy.orm import sessionmaker

from scripts import create_transcription_process
from vmrt_tesseract_utilities.database import (PATH_LENGTH, Base,
                                               TranscriptionInput,
                                               TranscriptionMetadata,
                                               TranscriptionOutput)


def test_registration_skips_already_registered_files(tmp_path, monkeypatch):
    """
    Tests that registering the same input files more than once does not create
    duplicate transcription input or metadata records.

    Parameters:
        tmp_path : pathlib.Path
            Temporary directory used to create the input file tree.
        monkeypatch : pytest.MonkeyPatch
            Fixture used to replace the database session with an isolated test database.

    Asserts:
        The second registration run does not add duplicate records.
    """
    # Build a small input tree matching the registration script's expected layout.
    dog_dir = tmp_path / "094-000131"
    dog_dir.mkdir()
    (dog_dir / "record.pdf").write_text("test")

    # Use an isolated database; this test won't touch the Docker database.
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    test_sessionmaker = sessionmaker(bind=engine)
    monkeypatch.setattr(
        create_transcription_process,
        "get_database_session",
        lambda **kwargs: test_sessionmaker,
    )

    args = Namespace(
        path_to_describe=str(tmp_path),
        document_type="document",
        debug_sql=False,
    )

    create_transcription_process.do_create_transcription_process(args)
    create_transcription_process.do_create_transcription_process(args)

    with test_sessionmaker() as session:
        inputs = session.query(TranscriptionInput).all()
        metadata = session.query(TranscriptionMetadata).all()

    assert len(inputs) == 1
    assert inputs[0].input_file == str(dog_dir / "record.pdf")
    assert inputs[0].document_type == "document"
    assert len(metadata) == 1
    assert metadata[0].subject_id == "094-000131"


def test_registration_treats_path_spellings_as_the_same_file(tmp_path, monkeypatch):
    """
    Tests that a file registered through different spellings of its path only
    creates one record.

    Parameters:
        tmp_path : pathlib.Path
            Temporary directory used to create the input file tree.
        monkeypatch : pytest.MonkeyPatch
            Fixture used to replace the database session and working directory.

    Asserts:
        Absolute, trailing-slash and relative spellings produce a single record.
    """
    dog_dir = tmp_path / "094-000131"
    dog_dir.mkdir()
    (dog_dir / "record.pdf").write_text("test")

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    test_sessionmaker = sessionmaker(bind=engine)
    monkeypatch.setattr(
        create_transcription_process,
        "get_database_session",
        lambda **kwargs: test_sessionmaker,
    )
    monkeypatch.chdir(tmp_path)

    for path in (str(dog_dir), f"{dog_dir}/", "094-000131"):
        args = Namespace(path_to_describe=path, document_type="document", debug_sql=False)
        create_transcription_process.do_create_transcription_process(args)

    with test_sessionmaker() as session:
        inputs = session.query(TranscriptionInput).all()

    assert [row.input_file for row in inputs] == [str(dog_dir / "record.pdf")]


def test_registration_reads_subject_id_from_path(tmp_path, monkeypatch, caplog):
    """
    Tests which folders are registered, and under which subject ID, when folder names contain other numbers.

    Parameters:
        tmp_path : pathlib.Path
            Temporary directory used to create the input file tree.
        monkeypatch : pytest.MonkeyPatch
            Fixture used to replace the database session with an isolated test database.
        caplog : pytest.LogCaptureFixture
            Fixture used to capture log messages.

    Asserts:
        Files under a GRLS subject ID folder are registered with that ID, and files in folders whose ID is part of
        a longer number, or has another site prefix, are skipped. Skipped folders whose name contains an ID-shaped
        number are named in a warning.
    """
    registered = {
        "123-456789/094-000131": "094-000131",
        "scan 2020-094-000132": "094-000132",
    }
    skipped = ["094-0001312", "1094-000133", "095-000134", "scan_2020-123094-000135"]
    for folder in [*registered, *skipped]:
        (tmp_path / folder).mkdir(parents=True)
        (tmp_path / folder / "record.pdf").write_text("test")

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    test_sessionmaker = sessionmaker(bind=engine)
    monkeypatch.setattr(
        create_transcription_process,
        "get_database_session",
        lambda **kwargs: test_sessionmaker,
    )

    args = Namespace(path_to_describe=str(tmp_path), document_type="document", debug_sql=False)
    create_transcription_process.do_create_transcription_process(args)

    with test_sessionmaker() as session:
        found = {
            metadata.transcription_input.input_file: metadata.subject_id
            for metadata in session.query(TranscriptionMetadata).all()
        }

    assert found == {
        str(tmp_path / folder / "record.pdf"): subject_id for folder, subject_id in registered.items()
    }
    warned = sorted(record.getMessage().rsplit(": ", 1)[1] for record in caplog.records
                    if record.levelname == "WARNING")
    assert warned == sorted(str(tmp_path / folder) for folder in skipped if folder != "095-000134")


def _sqlite_engine():
    """
    An in-memory SQLite database with the schema, set up so savepoints behave as they do in MySQL.

    Python's sqlite3 driver doesn't send BEGIN, so releasing a savepoint would commit it on its own; this is the
    fix from SQLAlchemy's SQLite documentation ("Serializable isolation / Savepoints / Transactional DDL").
    """
    engine = create_engine("sqlite:///:memory:")

    @event.listens_for(engine, "connect")
    def no_driver_transactions(dbapi_connection, connection_record):
        dbapi_connection.isolation_level = None

    @event.listens_for(engine, "begin")
    def emit_begin(connection):
        connection.exec_driver_sql("BEGIN")

    return engine


def _register(tmp_path, monkeypatch, engine=None):
    """
    Registers the files under tmp_path as documents, in engine or a new in-memory database, and returns the
    registered paths.
    """
    if engine is None:
        engine = _sqlite_engine()
        Base.metadata.create_all(engine)
    test_sessionmaker = sessionmaker(bind=engine)
    monkeypatch.setattr(create_transcription_process, "get_database_session", lambda **kwargs: test_sessionmaker)
    create_transcription_process.do_create_transcription_process(
        Namespace(path_to_describe=str(tmp_path), document_type="document", debug_sql=False))
    with test_sessionmaker() as session:
        return sorted(row.input_file for row in session.query(TranscriptionInput))


def test_registration_only_registers_pdfs(tmp_path, monkeypatch, caplog):
    """
    Test that only PDFs are registered, whatever the case of the extension, since they're all OCR reads.

    The EMR folders also hold image viewer exports, lock files and the like, five times as many files as PDFs.

    Asserts:
        record.pdf and SCAN.PDF are registered, LOCKFILE and image.jpg aren't, and the summary counts them.
    """
    dog_dir = tmp_path / "094-000131"
    dog_dir.mkdir()
    for name in ("record.pdf", "SCAN.PDF", "LOCKFILE", "image.jpg"):
        (dog_dir / name).write_text("test")

    with caplog.at_level(logging.INFO, logger="import_utility"):
        registered = _register(tmp_path, monkeypatch)

    assert registered == [str(dog_dir / "SCAN.PDF"), str(dog_dir / "record.pdf")]
    assert ("Registered 2 new PDFs. Skipped 0 already registered, 2 files that aren't PDFs, 0 with paths too long "
            "to store, and 0 the database rejected.") in caplog.text


def test_registration_skips_paths_too_long_for_the_live_column(tmp_path, monkeypatch, caplog):
    """
    Test that the length check uses the database's actual column widths, not the model's.

    A database created before the columns were widened keeps its old widths. Here the tables are the model's with
    input_file just wider than the short path, and ocr_output_file narrow too.

    Asserts:
        The short path is registered and the long one isn't; the warning names both narrow columns; the summary
        counts the skipped path.
    """
    dog_dir = tmp_path / "094-000131"
    dog_dir.mkdir()
    short = dog_dir / "a.pdf"
    short.write_text("test")
    # Sized from the temporary folder's path, which varies.
    width = len(str(short)) + 5
    long_name = dog_dir / ("x" * 10 + ".pdf")
    long_name.write_text("test")
    assert width < PATH_LENGTH

    # A copy of the model's tables with narrower path columns, as an old database would have.
    old_schema = MetaData()
    for table in Base.metadata.sorted_tables:
        table.to_metadata(old_schema)
    old_schema.tables["transcription_input"].c.input_file.type = String(width)
    old_schema.tables["transcription_output"].c.ocr_output_file.type = String(255)
    engine = _sqlite_engine()
    old_schema.create_all(engine)

    with caplog.at_level(logging.INFO, logger="import_utility"):
        registered = _register(tmp_path, monkeypatch, engine)

    assert registered == [str(short)]
    assert (f"These path columns hold fewer than {PATH_LENGTH} characters, as the database was created before they "
            f"were widened: transcription_input.input_file ({width}), transcription_output.ocr_output_file (255).") in caplog.text
    assert f"Not registered, the path is longer than {width} characters: {long_name}" in caplog.text
    assert "1 with paths too long to store" in caplog.text


def test_the_model_tables_are_left_alone_by_the_narrow_copy():
    """
    Test that narrowing the copied tables in the test above didn't change the model's own columns.

    Asserts:
        input_file is still PATH_LENGTH wide.
    """
    assert TranscriptionInput.__table__.c.input_file.type.length == PATH_LENGTH


def test_registration_needs_the_tables(tmp_path, monkeypatch, caplog):
    """
    Test that registering into a database with no tables says to install them, rather than looking like a failure
    part way through.

    Asserts:
        RuntimeError names database_setup.py install, and the log says no files were looked at.
    """
    (tmp_path / "094-000131").mkdir()
    with caplog.at_level(logging.INFO, logger="import_utility"), \
            pytest.raises(RuntimeError, match="database_setup.py install"):
        _register(tmp_path, monkeypatch, _sqlite_engine())
    assert "Registration failed before looking at any files" in caplog.text
    assert "got as far as" not in caplog.text


def test_a_root_folder_named_like_apple_double_still_registers(tmp_path, monkeypatch):
    """
    Test that the "._" check applies only below the folder being registered, not to the folder itself.

    Asserts:
        The PDF inside a root folder named "._copy" is registered.
    """
    root = tmp_path / "._copy"
    dog_dir = root / "094-000131"
    dog_dir.mkdir(parents=True)
    (dog_dir / "record.pdf").write_text("test")
    assert _register(root, monkeypatch) == [str(dog_dir / "record.pdf")]


def test_registration_skips_apple_double_files(tmp_path, monkeypatch):
    """
    Test that macOS "._" files, which end in .pdf but aren't PDFs, aren't registered.

    Asserts:
        Only the real PDF is registered.
    """
    dog_dir = tmp_path / "094-000131"
    dog_dir.mkdir()
    (dog_dir / "Medical Records.pdf").write_text("test")
    (dog_dir / "._Medical Records.pdf").write_text("test")

    assert _register(tmp_path, monkeypatch) == [str(dog_dir / "Medical Records.pdf")]


def test_registration_warns_about_non_pdfs_from_an_older_registration(tmp_path, monkeypatch, caplog):
    """
    Test that non-PDF rows left by a registration that took every file are counted and reported.

    Only rows the README's cleanup removes are counted: a non-PDF that was transcribed stays, so it isn't.

    Asserts:
        The warning counts the untranscribed non-PDF only, and the new PDF is still registered.
    """
    dog_dir = tmp_path / "094-000131"
    dog_dir.mkdir()
    (dog_dir / "record.pdf").write_text("test")
    engine = _sqlite_engine()
    Base.metadata.create_all(engine)
    with sessionmaker(bind=engine)() as session:
        session.add(TranscriptionInput(document_type="document", input_file=str(dog_dir / "LOCKFILE")))
        # Transcribed, so the README's cleanup leaves it and the warning mustn't count it.
        transcribed = TranscriptionInput(document_type="document", input_file=str(dog_dir / "._record.pdf"))
        session.add(TranscriptionOutput(transcription_input=transcribed, ocr_output_file="out.txt"))
        session.commit()

    with caplog.at_level(logging.INFO, logger="import_utility"):
        registered = _register(tmp_path, monkeypatch, engine)

    assert str(dog_dir / "record.pdf") in registered
    assert '1 files already registered as "document" aren\'t PDFs' in caplog.text


def test_a_rejected_file_doesnt_undo_the_others(tmp_path, monkeypatch, caplog):
    """
    Test that a file the database rejects is rolled back on its own, and the others are registered.

    The rejection is simulated with a before_insert hook raising the DataError MySQL gives for a value the column
    can't hold.

    Asserts:
        The other two PDFs and their metadata are registered, the rejected one isn't, a warning names it, and the
        summary counts it.
    """
    dog_dir = tmp_path / "094-000131"
    dog_dir.mkdir()
    for name in ("a.pdf", "bad.pdf", "c.pdf"):
        (dog_dir / name).write_text("test")

    def reject(mapper, connection, target):
        if target.input_file.endswith("bad.pdf"):
            raise DataError("INSERT", {}, Exception("Data too long for column 'input_file'"))

    event.listen(TranscriptionInput, "before_insert", reject)
    try:
        with caplog.at_level(logging.INFO, logger="import_utility"):
            engine = _sqlite_engine()
            Base.metadata.create_all(engine)
            registered = _register(tmp_path, monkeypatch, engine)
    finally:
        event.remove(TranscriptionInput, "before_insert", reject)

    assert registered == [str(dog_dir / "a.pdf"), str(dog_dir / "c.pdf")]
    with sessionmaker(bind=engine)() as session:
        assert session.query(TranscriptionMetadata).count() == 2
    assert f"Not registered, the database rejected it (Exception: Data too long for column 'input_file'): {dog_dir / 'bad.pdf'}" in caplog.text
    assert "and 1 the database rejected." in caplog.text


def test_a_failed_registration_logs_how_far_it_got(tmp_path, monkeypatch, caplog):
    """
    Test that if registration fails part way, the error is raised and what it had counted so far is logged.

    Asserts:
        The error propagates, the counts so far are logged, and nothing is saved.
    """
    dog_dir = tmp_path / "094-000131"
    dog_dir.mkdir()
    (dog_dir / "a.pdf").write_text("test")
    real_walk = os.walk

    def walk_then_fail(top):
        yield from real_walk(top)
        raise OSError("Dropbox folder went away")

    monkeypatch.setattr(create_transcription_process.os, "walk", walk_then_fail)
    engine = _sqlite_engine()
    Base.metadata.create_all(engine)
    with caplog.at_level(logging.INFO, logger="import_utility"), pytest.raises(OSError, match="went away"):
        _register(tmp_path, monkeypatch, engine)

    assert "Registration failed and nothing from this run was saved. It had got as far as 1 new PDFs" in caplog.text
    with sessionmaker(bind=engine)() as session:
        assert session.query(TranscriptionInput).count() == 0


def test_path_columns_hold_long_paths():
    """
    Test that every file path column is PATH_LENGTH wide, and that is longer than the longest EMR path seen (284).

    Asserts:
        input_file and the four output path columns are all PATH_LENGTH characters.
    """
    assert PATH_LENGTH > 284
    columns = [TranscriptionInput.__table__.c.input_file] + [
        TranscriptionOutput.__table__.c[name] for name in (
            "ocr_output_file", "list_replacement_output_file", "pii_scrubber_output_file", "pii_scrubber_confidence_file")]
    assert {column.type.length for column in columns} == {PATH_LENGTH}


def test_a_transaction_error_stops_registration(tmp_path, monkeypatch, caplog):
    """
    Test that an error that ends the transaction (a deadlock, a lost connection) stops the run, instead of being
    reported as one rejected file and carrying on with every remaining file failing the same way.

    Asserts:
        The OperationalError propagates, no file is reported as rejected, and nothing is saved.
    """
    dog_dir = tmp_path / "094-000131"
    dog_dir.mkdir()
    for name in ("a.pdf", "b.pdf", "c.pdf"):
        (dog_dir / name).write_text("test")

    def lose_connection(mapper, connection, target):
        if target.input_file.endswith("b.pdf"):
            raise OperationalError("INSERT", {}, Exception("Lost connection to MySQL server during query"))

    engine = _sqlite_engine()
    Base.metadata.create_all(engine)
    event.listen(TranscriptionInput, "before_insert", lose_connection)
    try:
        with caplog.at_level(logging.INFO, logger="import_utility"), pytest.raises(OperationalError):
            _register(tmp_path, monkeypatch, engine)
    finally:
        event.remove(TranscriptionInput, "before_insert", lose_connection)

    assert "the database rejected it" not in caplog.text
    assert "Registration failed and nothing from this run was saved" in caplog.text
    with sessionmaker(bind=engine)() as session:
        assert session.query(TranscriptionInput).count() == 0
