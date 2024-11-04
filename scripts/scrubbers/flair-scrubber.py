import argparse

from flair.data import Sentence
from flair.models import SequenceTagger

def scrub_pii_with_flair(input_file: str, output_file: str, confidence_threshold: float = 0.5, model = "ner-large") -> None:
    """
    Scrubs PII from a text file using Flair's models.

    Parameters
    ----------
    input_file : str
        Path to the input text file.
    output_file : str
        Path to the output text file.
    confidence_threshold : float, optional
        The minimum confidence score (between 0 and 1) for an entity
        to be redacted (default is 0.5).
    model : str, optional
        The Flair NER model to use.

    Returns
    -------
    None
        This function does not return any value; it writes the anonymized text
        to the output file.
    """

    # Load the Flair NER model
    model = SequenceTagger.load(model)

    with open(input_file, "r") as f:
        text = f.read()

    # Anonymize the text
    anonymized_text = anonymize_text(text, model, confidence_threshold)

    with open(output_file, "w") as f:
        f.write(anonymized_text)

def anonymize_text(text: str, model: SequenceTagger, confidence_threshold: float) -> str:
    """
    Replaces PII entities in the text with "[REDACTED]",
    considering the confidence threshold.

    Parameters
    ----------
    text : str
        The original text.
    model : flair.models.SequenceTagger
        The loaded Flair NER model.
    confidence_threshold : float
        The minimum confidence score for redaction.

    Returns
    -------
    str
        The anonymized text with PII entities redacted.
    """

    sentence = Sentence(text)
    model.predict(sentence)

    redacted_text = ""
    last_index = 0
    for entity in sentence.get_spans('ner'):
        if entity.score >= confidence_threshold:
            redacted_text += text[last_index:entity.start_position] + f"<{entity.tag}>"
            last_index = entity.end_position
    redacted_text += text[last_index:]

    return redacted_text

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrub PII from a text file using Flair.")
    parser.add_argument("--input_file", help="Path to the input text file.", required=True)
    parser.add_argument("--output_file", help="Path to the output text file.", required=True)
    parser.add_argument("--confidence_threshold", type=float, default=0.5,
                        help="Minimum confidence score for redaction (default: 0.5)")
    parser.add_argument("--model", help="Flair NER model to use.", default="ner-large")
    args = parser.parse_args()

    scrub_pii_with_flair(args.input_file, args.output_file, args.confidence_threshold)