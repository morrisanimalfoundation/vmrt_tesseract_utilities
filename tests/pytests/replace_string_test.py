from unittest.mock import MagicMock, patch

from argparse import Namespace

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from scripts.replace_strings import (process_file, read_target_strings, get_files_to_process,
                                     make_pass_hash, scrub_and_write_files, get_or_create_pass)
from vmrt_tesseract_utilities.database import (
    Base,
    StringReplacementCompletion,
    StringReplacementPass,
    TranscriptionInput,
    TranscriptionOutput,
)

"""
Tests for the `replace_strings` script.
"""


def test_read_target_strings_csv():
    """
    Test the `read_target_strings` function with a CSV file.

    This test mocks the `open` function to simulate reading a CSV file and verifies
    that the `read_target_strings` function correctly extracts the target strings.

    Asserts:
        The result matches the expected list of strings.
    """
    with patch("builtins.open", new_callable=MagicMock) as mock_open:
        mock_open.return_value.__enter__.return_value = [
            "key_column\n",
            "string1\n",
            "string2\n"
        ]
        result = read_target_strings("test.csv", "key_column")
        assert result == ["string1", "string2"]


def test_read_target_strings_tsv():
    """
    Test the `read_target_strings` function with a TSV file.

    This test mocks the `open` function to simulate reading a TSV file and verifies
    that the `read_target_strings` function correctly extracts the target strings.

    Asserts:
        The result matches the expected list of strings.
    """
    with patch("builtins.open", new_callable=MagicMock) as mock_open:
        mock_open.return_value.__enter__.return_value = [
            "key_column\n",
            "string1\n",
            "string2\n"
        ]
        result = read_target_strings("test.tsv", "key_column")
        assert result == ["string1", "string2"]


def test_process_file():
    """
    Test the `process_file` function.

    This test mocks the `open` function to simulate reading and writing files,
    and verifies that the `process_file` function correctly processes the file
    and updates the `output_log` object.

    Asserts:
        The `list_replacement_output_file` attribute of the result matches the expected output file path.
    """
    output_log = MagicMock()
    output_log.ocr_output_file = "test.txt"
    output_log.list_replacement_output_file = None
    strings_to_replace = ["old_string"]
    parsed_args = MagicMock()
    parsed_args.replacement_string = "new_string"
    parsed_args.output_dir = "output"
    parsed_args.document_type = "document"

    with patch("builtins.open", new_callable=MagicMock) as mock_open:
        mock_open.return_value.__enter__.return_value.read.return_value = "This is a text blob with old_string."
        result = process_file(output_log, strings_to_replace, parsed_args)
        assert result.list_replacement_output_file == "output/list_replacement_output_file/document/test.txt"


def test_make_pass_hash_uses_effective_ordered_targets():
    """
    Tests that pass hashes ignore blank targets but preserve target order.

    Asserts:
        Blank targets do not change the hash, and reordering targets does.
    """
    first_hash, first_targets = make_pass_hash(
        ["Max", "", "Maxwell"],
        "<NAME>",
    )
    blank_removed_hash, blank_removed_targets = make_pass_hash(
        ["Max", "Maxwell"],
        "<NAME>",
    )
    reordered_hash, _ = make_pass_hash(
        ["Maxwell", "Max"],
        "<NAME>",
    )

    assert first_targets == ["Max", "Maxwell"]
    assert blank_removed_targets == ["Max", "Maxwell"]
    assert first_hash == blank_removed_hash
    assert first_hash != reordered_hash



def test_replacement_pass_skips_same_pass_and_processes_different_pass(tmp_path, monkeypatch):
    """
    Tests that completed outputs are skipped for the same pass and selected
    again for a different replacement pass.

    Parameters:
        tmp_path : pathlib.Path
            Temporary directory used for the OCR input and replacement output.
        monkeypatch : pytest.MonkeyPatch
            Fixture used to replace the database connection with an isolated database.

    Asserts:
        The same pass does not change a completed file, while a different pass
        processes that file again.
    """
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    test_sessionmaker = sessionmaker(bind=engine)

    monkeypatch.setattr("scripts.replace_strings.get_database_session", lambda **kwargs: test_sessionmaker)

    ocr_file = tmp_path / "ocr.txt"
    ocr_file.write_text("Max Maxwell")

    with test_sessionmaker.begin() as session:
        transcription_input = TranscriptionInput(input_file="record.pdf", document_type="document")
        output_log = TranscriptionOutput(transcription_input=transcription_input, ocr_output_file=str(ocr_file))
        session.add(output_log)

    args = Namespace(
        data_file="names.tsv",
        key_column="name",
        replacement_string="<NAME>",
        output_dir=str(tmp_path),
        document_type="document",
        chunk_size=100,
        debug_sql=False,
        max_workers=1,
        no_multiprocessing=True,
    )

    first_hash, first_targets = make_pass_hash(
        ["Max", "Maxwell"],
        args.replacement_string,
    )
    args.replacement_pass_id = get_or_create_pass(args, first_hash, len(first_targets))

    first_batch = get_files_to_process(args)
    assert len(first_batch) == 1

    scrub_and_write_files(first_batch, first_targets, args, use_multiprocessing=False)

    replacement_file = tmp_path / "list_replacement_output_file" / "document" / "ocr.txt"
    first_result = replacement_file.read_text()
    assert first_result == "<NAME> <NAME>well"

    # The completed first pass should no longer select this output.
    same_pass_batch = get_files_to_process(args)
    assert same_pass_batch == []

    scrub_and_write_files(same_pass_batch, first_targets, args, use_multiprocessing=False)
    assert replacement_file.read_text() == first_result

    second_hash, second_targets = make_pass_hash(
        ["<NAME>well"],
        args.replacement_string,
    )
    args.replacement_pass_id = get_or_create_pass(args, second_hash, len(second_targets))

    second_batch = get_files_to_process(args)
    assert len(second_batch) == 1

    scrub_and_write_files(second_batch, second_targets, args, use_multiprocessing=False)

    assert replacement_file.read_text() == "<NAME> <NAME>"