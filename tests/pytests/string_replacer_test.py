from vmrt_tesseract_utilities.string_replacer import StringReplacer

"""
Tests for the StringReplacer class.
"""


def test_replace_single_string():
    """
    Test the `replace` method for a single string replacement.

    This test verifies that the `replace` method correctly replaces a single
    target string with the replacement string in the given text blob.

    Asserts:
        The result matches the expected string with the target string replaced.
    """
    replacer = StringReplacer(["old_string"], "new_string")
    text_blob = "This is a text blob with old_string."
    expected = "This is a text blob with new_string."
    assert replacer.replace(text_blob) == expected


def test_replace_multiple_strings():
    """
    Test the `replace` method for multiple string replacements.

    This test verifies that the `replace` method correctly replaces multiple
    target strings with the replacement string in the given text blob.

    Asserts:
        The result matches the expected string with all target strings replaced.
    """
    replacer = StringReplacer(["old_string1", "old_string2"], "new_string")
    text_blob = "This is a text blob with old_string1 and old_string2."
    expected = "This is a text blob with new_string and new_string."
    assert replacer.replace(text_blob) == expected


def test_replace_no_match():
    """
    Test the `replace` method when there are no matches.

    This test verifies that the `replace` method returns the original text blob
    unchanged when none of the target strings are found.

    Asserts:
        The result matches the original text blob.
    """
    replacer = StringReplacer(["old_string"], "new_string")
    text_blob = "This is a text blob with no match."
    expected = "This is a text blob with no match."
    assert replacer.replace(text_blob) == expected


def test_replace_empty_target_strings():
    """
    Test the `replace` method with an empty list of target strings.

    This test verifies that the `replace` method returns the original text blob
    unchanged when the list of target strings is empty.

    Asserts:
        The result matches the original text blob.
    """
    replacer = StringReplacer([], "new_string")
    text_blob = "This is a text blob with old_string."
    expected = "This is a text blob with old_string."
    assert replacer.replace(text_blob) == expected


def test_replace_empty_text_blob():
    """
    Test the `replace` method with an empty text blob.

    This test verifies that the `replace` method returns an empty string when
    the input text blob is empty.

    Asserts:
        The result is an empty string.
    """
    replacer = StringReplacer(["old_string"], "new_string")
    text_blob = ""
    expected = ""
    assert replacer.replace(text_blob) == expected


def test_replace_special_characters():
    """
    Test the `replace` method with special characters in the target and replacement strings.

    This test verifies that the `replace` method correctly handles special characters
    in the target and replacement strings.

    Asserts:
        The result matches the expected string with the target string replaced.
    """
    replacer = StringReplacer(["old_string!"], "new_string?")
    text_blob = "This is a text blob with old_string!."
    expected = "This is a text blob with new_string?."
    assert replacer.replace(text_blob) == expected


def test_replace_id_variant_without_dash():
    """
    Test that a GRLS-style ID target also matches its no-dash form in the text.

    Asserts:
        The no-dash occurrence is replaced even though the target string has a dash.
    """
    replacer = StringReplacer(["094-000520"], "<ID>")
    text_blob = "Subject 094000520 was seen for a follow-up."
    expected = "Subject <ID> was seen for a follow-up."
    assert replacer.replace(text_blob) == expected


def test_replace_id_variant_zero_padded():
    """
    Test that a GRLS-style ID target also matches its zero-padded forms in the text.

    Asserts:
        Both the dashed and non-dashed zero-padded occurrences are replaced.
    """
    replacer = StringReplacer(["094-000520"], "<ID>")
    text_blob = "Records 094-0000520 and 0940000520 refer to the same dog."
    expected = "Records <ID> and <ID> refer to the same dog."
    assert replacer.replace(text_blob) == expected


def test_replace_id_variant_original_still_matches():
    """
    Test that the original dashed ID form is still matched as before.

    Asserts:
        The exact target string is still replaced.
    """
    replacer = StringReplacer(["094-000520"], "<ID>")
    text_blob = "Subject 094-000520 presented for a routine visit."
    expected = "Subject <ID> presented for a routine visit."
    assert replacer.replace(text_blob) == expected


def test_replace_non_id_string_unaffected_by_id_expansion():
    """
    Test that target strings which don't look like a GRLS ID aren't expanded.

    Asserts:
        A plain name-like target only matches itself, not an unrelated numeric string.
    """
    replacer = StringReplacer(["Jones-1"], "<ID>")
    text_blob = "Jones-1 and Jones1 should not both be treated as ID variants."
    expected = "<ID> and Jones1 should not both be treated as ID variants."
    assert replacer.replace(text_blob) == expected
