import argparse
import csv
import os
import re
import time

import yaml
import spacy
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine

EXCLUDE_TYPES = ['IN_PAN']

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

def normalize_text(text):
    """Normalizes text to be CSS friendly.

    This function converts the input text to lowercase, replaces spaces and slashes
    with hyphens, and removes any characters that are not alphanumeric, hyphens,
    or underscores.

    Args:
      text: The text to normalize.

    Returns:
      The normalized text.
    """

    # Build a translation table
    translation_table = str.maketrans({
        " ": "-",
        "/": "-"
    })

    # Normalize the text
    text = text.lower().translate(translation_table)
    text = re.sub(r"[^a-zA-Z0-9_-]", "", text)
    return text

def scrub_pii(text: str, config_file: str, filename: str, threshold: float):
    """
    Scrub PII from a text file using the Presidio library.

    Parameters
    ----------
    text : str
        The text to scrub.
    config_file : str
        The NLP engine configuration file.
    filename : str
        The output filename.
    threshold: float
        The score threshold for our filter.

    Returns
    -------
    str
        The scrubbed text.

    Raises
    ------
    Exception
        If any error occurs during the process, an exception is raised with a descriptive message.
    """

    try:
        # Create NLP engine based on configuration
        provider = NlpEngineProvider(conf_file=config_file)
        nlp_engine = provider.create_engine()

        # Pass the created NLP engine and supported_languages to the AnalyzerEngine
        analyzer = AnalyzerEngine(
            nlp_engine=nlp_engine,
            supported_languages=["en"]
        )

        # Analyze the text for PII entities
        results = analyzer.analyze(
            text=text, language="en"
        )
        # Filter results based on the threshold and exclude 'IN_PAN'
        filtered_results = [
            result for result in results
            if result.score >= threshold and result.entity_type not in EXCLUDE_TYPES
        ]

        with open(f'./output-csv/{filename}.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Original Text", "Type", "Start", "End", "Score", 'Is PII'])  # Write header row
            for result in filtered_results:
                original_text = text[result.start:result.end].replace("\n", "")
                writer.writerow([original_text, result.entity_type, result.start, result.end, result.score, is_pii(original_text)])

        anonymizer = AnonymizerEngine()

        # Anonymize the identified PII entities
        anonymized_results = anonymizer.anonymize(
            text=text, analyzer_results=filtered_results
        )

        return anonymized_results.text

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrub PII from a text file.")
    parser.add_argument("--input_file", required=True, help="Path to the input text file")
    parser.add_argument("--threshold", required=True, help="The target threshold for our scores.")

    args = parser.parse_args()

    configs = [
        'spacy_nlp',
        'bert-base_nlp',
        'deid-roberta-i2b2_nlp',
        'stanford-deidentifier-base_nlp',
        'vet-bert_nlp',
    ]

    for config in configs:
        config_file = f'./config/{config}.yaml'
        filename = f'presidio-{config}-{normalize_text(args.threshold)}-threshold'
        with open(config_file, 'r') as file:
            data = yaml.safe_load(file)

        models = data.get('models')
        for model in models:
            is_transformer = isinstance(model['model_name'], dict)
            if is_transformer:
                name = model['model_name'].get('transformers', data.get('nlp_engine_name'))
            else:
                name = data.get('nlp_engine_name')
            # Try to load the NLP if it isn't already loaded.
            if is_transformer:
                pipeline = model['model_name'].get('spacy')
                try:
                    nlp = spacy.load(pipeline)
                except OSError:
                    print(f"Downloading {pipeline} model/pipeline...")
                    spacy.cli.download(pipeline)
                    time.sleep(2)
                    nlp = spacy.load(pipeline)
            # Build the output filename.
            current_file_path = os.path.dirname(os.path.abspath(__file__))
            output_file = f'{current_file_path}/output/{filename}.txt'

            # Read the input text
            with open(args.input_file, "r") as f:
                orig_text = f.read()

            # Scrub the file based on the model defined.
            scrubbed_text = scrub_pii(orig_text, config_file, filename, float(args.threshold))

            # Write the scrubbed text to the output file
            if scrubbed_text:
                with open(output_file, "w") as f:
                    f.write(scrubbed_text)

            print(f"PII scrubbed successfully. Output written to {output_file}")
