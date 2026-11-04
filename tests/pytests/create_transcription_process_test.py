import logging
from argparse import Namespace

from sqlalchemy import create_engine
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


def _register(tmp_path, monkeypatch):
    """
    Registers the files under tmp_path as documents in an in-memory database, and returns the registered paths.
    """
    engine = create_engine("sqlite:///:memory:")
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
    assert "Registered 2 new PDFs. Skipped 0 already registered, 2 files that aren't PDFs, and 0 with paths too long" in caplog.text


def test_registration_skips_paths_too_long_to_store(tmp_path, monkeypatch, caplog):
    """
    Test that a path longer than the database column is reported and skipped, and the other files still register.

    Before, one such path made the insert fail, which rolled back the whole registration.

    Asserts:
        The short path is registered, the long one isn't, a warning names it, and the summary counts it.
    """
    dog_dir = tmp_path / "094-000131"
    dog_dir.mkdir()
    short = dog_dir / "record.pdf"
    short.write_text("test")
    long_name = dog_dir / ("x" * 40 + ".pdf")
    long_name.write_text("test")
    monkeypatch.setattr(create_transcription_process, "PATH_LENGTH", len(str(short)) + 5)

    with caplog.at_level(logging.INFO, logger="import_utility"):
        registered = _register(tmp_path, monkeypatch)

    assert registered == [str(short)]
    assert f"Not registered, the path is longer than {len(str(short)) + 5} characters: {long_name}" in caplog.text
    assert "and 1 with paths too long to store." in caplog.text


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
