from vmrt_tesseract_utilities.logging import stdout_logger

"""
Utility functions for scrubbing actions.
"""


def split_text(text: str, max_length: int) -> list[tuple[int, str]]:
    r"""
    Splits text into pieces of at most max_length characters, each with its offset in the text.

    Each piece ends at the last line break that fits, so a word (and usually a name or address) isn't cut in two.
    If that line break is in the first half of the piece, or there isn't one, the piece ends at the last space or
    tab instead, if that's later. Only a run of max_length characters with neither is cut
    mid-word. Joining the pieces gives back the text.

    Parameters
    ----------
    text : str
        The text to split.
    max_length : int
        The most characters a piece may have.

    Returns
    -------
    list[tuple[int, str]]
        (offset, piece) pairs, in order. Empty text gives no pieces.

    Examples
    --------
    >>> split_text('one two\nthree four', 10)
    [(0, 'one two\n'), (8, 'three four')]
    >>> split_text('two three four', 8)
    [(0, 'two '), (4, 'three '), (10, 'four')]
    >>> split_text('a\nb c d e f', 10)
    [(0, 'a\nb c d e '), (10, 'f')]
    >>> split_text('abcdefghij', 4)
    [(0, 'abcd'), (4, 'efgh'), (8, 'ij')]
    >>> split_text('short', 10)
    [(0, 'short')]
    """
    if max_length < 1:
        raise ValueError('max_length must be at least 1.')
    pieces = []
    start = 0
    while start < len(text):
        end = start + max_length
        if end < len(text):
            window = text[start:end]
            cut = window.rfind('\n')
            # A line break near the start would make a tiny piece, so a later space or tab is used instead.
            if cut < len(window) // 2:
                cut = max(cut, window.rfind(' '), window.rfind('\t'))
            if cut >= 0:
                end = start + cut + 1
        pieces.append((start, text[start:end]))
        start = end
    return pieces


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
