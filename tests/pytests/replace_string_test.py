from unittest.mock import MagicMock, patch

from scripts.replace_strings import process_file, read_target_strings

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
