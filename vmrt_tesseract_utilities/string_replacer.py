from typing import List

"""
Provides string replacement functionality.
"""


class StringReplacer:
    """
    Replaces target strings within a text blob with a specified replacement string.

    Parameters
    ----------
    target_strings : list of str
        The strings to be replaced.
    replacement_string : str
        The string to replace the target strings with.

    Examples
    --------
    >>> replacer = StringReplacer(["old_string1", "old_string2"], "<new_string>")
    >>> text_blob = "This is a text blob with old_string1 and old_string2."
    >>> new_blob = replacer.replace(text_blob)
    >>> print(new_blob)
    This is a text blob with <new_string> and <new_string>.
    """
    def __init__(self, target_strings: List[str], replacement_string: str):
        """
        Initializes the StringReplacer.

        Parameters
        ----------
        target_strings : list of str
            The strings to be replaced.
        replacement_string : str
            The string to replace the target strings with.
        """
        self.target_strings = target_strings
        self.replacement_string = replacement_string

    def replace(self, text_blob: str) -> str:
        """
        Replaces the target strings within the text blob.

        Parameters
        ----------
        text_blob : str
            The input text blob.

        Returns
        -------
        str
            The modified text blob with replaced strings.
        """
        for target in self.target_strings:
            text_blob = text_blob.replace(target, self.replacement_string)
        return text_blob
