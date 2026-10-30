from argparse import Namespace

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from scripts import create_transcription_process
from vmrt_tesseract_utilities.database import (Base, TranscriptionInput,
                                               TranscriptionMetadata)


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