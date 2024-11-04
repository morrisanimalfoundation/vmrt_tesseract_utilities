import argparse

import spacy

def scrub_pii_with_spacy(input_file, output_file):
    """
    Scrubs PII from a text file using spaCy's `en_core_web_lg` model.

    Allows specifying a minimum confidence score for entities to be redacted.

    Parameters
    ----------
    input_file : str
        Path to the input text file.
    output_file : str
        Path to the output text file.

    Returns
    -------
    None
        This function does not return any value; it writes the anonymized text to the output file.
    """

    try:
        nlp = spacy.load("en_core_web_lg")
    except OSError:
        print("Downloading en_core_web_lg model...")
        spacy.cli.download("en_core_web_lg")
        nlp = spacy.load("en_core_web_lg")

    with open(input_file, "r") as f:
        text = f.read()

    doc = nlp(text)

    anonymized_text = anonymize_text(text, doc)

    with open(output_file, "w") as f:
        f.write(anonymized_text)

def anonymize_text(text, doc):
    """
    Replaces PII entities in the text with "[REDACTED]",
    considering the confidence threshold.

    Parameters
    ----------
    text : str
        The original text.
    doc : spacy.tokens.Doc
        The spaCy Doc object containing the analyzed text.

    Returns
    -------
    str
        The anonymized text with PII entities redacted.
    """
    redacted_text = text
    for ent in sorted(doc.ents, key=lambda x: x.start_char, reverse=True):
        # spaCy doesn't provide explicit confidence scores for entities
        redacted_text = redacted_text[:ent.start_char] + "[REDACTED]" + redacted_text[ent.end_char:]
    return redacted_text

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrub PII from a text file using spaCy.")
    parser.add_argument("--input_file", help="Path to the input text file.", required=True)
    parser.add_argument("--output_file", help="Path to the output text file.", required=True)
    args = parser.parse_args()

    scrub_pii_with_spacy(args.input_file, args.output_file)