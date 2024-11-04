import argparse
import csv

from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

def is_pii(target_string):
    """
    Checks if a target string is present (even partially) within an array of strings.

    Args:
        target_string: The string to search for.

    Returns:
        True if the target string is found in any of the array elements, False otherwise.
    """
    string_array = [
        "Zoetis Reference Laboratories",
        "300 High Rise Drive Suite 300",
        "Louisville, KY 40213",
        "zoetisreflabs.com",
        "888-965-9652",
        "WILLBANKS, Bentley",
        "Morris Animal Foundation",
        "Tietje, Kathleen",
        "Denver, CO 80246-1912",
        "094-011969",
        "11y5m",
        "10/26/2023 9:49 AM",
        "11/21/2023 12:18 PM",
        "11/21/2023 10:15:00 AM",
        "300 High Rise Drive Suite 300 Louisville, KY 40213",
        "10/31/2023 1:06 PM",
        "H368228",
        "094-01196",
        "702 South Colorado BLVD",
        "102602023",
        "Dr. Kathleen Tietje",
        "11/21/2023",
        "Danielle Nelson, DVM, PhD, DACVP",
        "Danielle.Nelson@ Zoetis.com",
        "ZoetisDx.com",
        "11/8/23",
        "1.888.965.9652",
        "10/26/23",
        "(303) 708-3405",
        "10/31/23",
        "Zoetis",
        "10767202",
        "Zgetis",
        "linical G ﬁ COLLEGE OF VETERINARY MEDICINE",
        "ematopathology COLORADD STATE UNIVERSITY",
        "BENTLEY WILLBANKS",
        "ZOETIS REFERENCE LABORATORIES",
        "1/1/2013",
        "NELSON",
        "11/10/23",
        "11/20/23",
        "(970) 491-1170",
        "ch-lab@colostate.edu",
        "chlab.colostate.edu",
        "970-297-1281"
    ]
    for item in string_array:
        if target_string.lower() in item.lower():
            return 1
    return 0

def scrub_pii_with_hf(input_file, output_file, score_threshold=0.4, model_name='StanfordAIMI/stanford-deidentifier-base'):
    """
    Scrubs PII from a text file using the Hugging Face `deid_roberta_i2b2` model.

    Redacts only entities with a confidence score higher than the specified threshold.

    Parameters
    ----------
    input_file : str
        Path to the input text file.
    output_file : str
        Path to the output text file.
    score_threshold : float, optional
        The minimum score for an entity to be redacted (default is 0.4).
    model_name : str, optional
        The name of the Hugging Face model to use.

    Returns
    -------
    None
        This function does not return any value; it writes the anonymized text to the output file.
    """

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)

    nlp = pipeline("token-classification", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

    with open(input_file, "r") as f:
        text = f.read()

    entities = nlp(text)

    # Filter entities based on score threshold
    filtered_entities = [entity for entity in entities if entity['score'] >= score_threshold]

    anonymized_text = anonymize_text(text, filtered_entities)

    with open(output_file, "w") as f:
        f.write(anonymized_text)
    # Write filtered entities to CSV
    csv_file = output_file.replace(".txt", ".csv")  # Create CSV filename
    with open(csv_file, 'w', newline='') as csvfile:
        fieldnames = ['entity_group', 'score', 'word', 'start', 'end']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(filtered_entities)

def anonymize_text(text, entities):
    """
    Replaces PII entities in the text with "[REDACTED]".

    Parameters
    ----------
    text : str
        The original text.
    entities : list
        A list of entities identified by the model.

    Returns
    -------
    str
        The anonymized text with PII entities redacted.
    """
    redacted_text = text
    for entity in sorted(entities, key=lambda x: x['start'], reverse=True):
        redacted_text = redacted_text[:entity['start']] + f"[REDACTED: {entity['entity_group']}]" + redacted_text[entity['end']:]
    return redacted_text

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrub PII from a text file using Hugging Face model.")
    parser.add_argument("--input_file", help="Path to the input text file.", required=True)
    parser.add_argument("--output_file", help="Path to the output text file.", required=True)
    parser.add_argument("--confidence_threshold", type=float, default=0.5,
                        help="Minimum score for redaction (default: 0.5)")
    parser.add_argument("--model", default="StanfordAIMI/stanford-deidentifier-base",
                        help="The Hugging Face model to use.")
    args = parser.parse_args()

    scrub_pii_with_hf(args.input_file, args.output_file, args.confidence_threshold, args.model)