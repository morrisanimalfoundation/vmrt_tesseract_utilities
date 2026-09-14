import re
from typing import List

"""
Provides string replacement functionality.
"""

# Matches GRLS-style subject IDs (a numeric site prefix, a dash, and a numeric
# suffix, e.g. "094-000520") so that variant formats appearing in OCR'd text can
# also be matched.
GRLS_ID_RE = re.compile(r'^(\d+)-(\d+)$')


class StringReplacer:
    """
    Replaces target strings within a text blob with a specified replacement string.

    A target string that looks like a GRLS subject ID (e.g. "094-000520") is also
    matched in the variant forms it commonly appears as in OCR'd text: without the
    dash ("094000520"), and with an extra zero-padded digit in the suffix
    ("094-0000520", "0940000520").

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

    @staticmethod
    def _expand_id_variants(target: str) -> List[str]:
        """
        Expands a GRLS-style subject ID into the variant forms it may appear as in
        OCR'd text. Strings that don't look like a GRLS ID are returned unchanged.

        Parameters
        ----------
        target : str
            The candidate target string.

        Returns
        -------
        list of str
            The target string, plus any additional ID variants.
        """
        match = GRLS_ID_RE.match(target)
        if not match:
            return [target]
        prefix, suffix = match.groups()
        padded_suffix = f'0{suffix}'
        variants = [
            target,
            f'{prefix}{suffix}',
            f'{prefix}-{padded_suffix}',
            f'{prefix}{padded_suffix}',
        ]
        # Preserve order while removing duplicates.
        return list(dict.fromkeys(variants))

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
            for variant in self._expand_id_variants(target):
                text_blob = re.sub(re.escape(variant), self.replacement_string, text_blob, flags=re.IGNORECASE)
        return text_blob
