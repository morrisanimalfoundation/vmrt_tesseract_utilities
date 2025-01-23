from vmrt_tesseract_utilities.logging import stdout_logger

"""
Utility functions for scrubbing actions.
"""


def write_scrubbed_txt(output_filename: str, anonymized_text: str) -> None:
    """
    Writes the anonymized text to an output file.

    Parameters
    ----------
    output_filename : str
        The path to the file.
    anonymized_text : str
        The anonymized text.
    """
    try:
        if anonymized_text:
            with open(output_filename, 'w') as f:
                f.write(anonymized_text)
    except Exception as e:
        stdout_logger.error(f'Error writing scrubbed output: {e}')
        raise
