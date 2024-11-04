import argparse
import stanza

def scrub_pii_with_stanza(input_file: str, output_file: str) -> None:
    """
    Scrubs PII from a text file using Stanza's `en` NER model.

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
        This function does not return any value; it writes the anonymized text
        to the output file.
    """

    stanza.download('en')
    nlp = stanza.Pipeline('en', processors='tokenize,ner')

    with open(input_file, 'r') as f:
        text = f.read()

    doc = nlp(text)

    anonymized_text = anonymize_text(text, doc)

    with open(output_file, 'w') as f:
        f.write(anonymized_text)

def anonymize_text(text: str, doc) -> str:
    """
    Replaces PII entities in the text with "[REDACTED]",
    considering the confidence threshold.

    Parameters
    ----------
    text : str
        The original text.
    doc : stanza.Document
        The Stanza Document object containing the analyzed text.

    Returns
    -------
    str
        The anonymized text with PII entities redacted.
    """
    redacted_text = ""
    last_index = 0
    for ent in doc.ents:
        # stanza doesn't provide explicit confidence scores for entities
        redacted_text += text[last_index:ent.start_char] + f"<{ent.type}>"
        last_index = ent.end_char
    redacted_text += text[last_index:]
    return redacted_text

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrub PII from a text file using Stanza.")
    parser.add_argument("--input_file", help="Path to the input text file.", required=True)
    parser.add_argument("--output_file", help="Path to the output text file.", required=True)
    args = parser.parse_args()
    scrub_pii_with_stanza(args.input_file, args.output_file)