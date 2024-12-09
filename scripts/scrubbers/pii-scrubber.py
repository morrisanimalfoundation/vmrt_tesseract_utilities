import argparse
import glob
import json
import logging
import os
from collections import defaultdict
from typing import Any

from presidio_analyzer import AnalyzerEngine, RecognizerResult
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

EXCLUDE_TYPES = frozenset(['IN_PAN'])  # Use frozenset for efficient lookups

def create_nlp_engine(nlp_config_file: str) -> AnalyzerEngine:
    """Creates and configures a Presidio AnalyzerEngine.

    Sets up a Presidio AnalyzerEngine using the provided
    configuration file, which specifies the NLP engine and other settings.

    Parameters
    ----------
    nlp_config_file : str
        Path to the YAML configuration file for the NLP engine.

    Returns
    -------
    AnalyzerEngine
        A configured Presidio AnalyzerEngine ready for PII analysis.

    Raises
    ------
    Exception
        If any error occurs during the creation of the NLP engine.
    """
    try:
        provider = NlpEngineProvider(conf_file=nlp_config_file)
        engine = provider.create_engine()
        return AnalyzerEngine(nlp_engine=engine, supported_languages=['en'])

    except Exception as e:
        logging.error(f'Error creating NLP engine: {e}')
        raise


def scrub_pii(text: str, analyzer: AnalyzerEngine, threshold: float) -> tuple[str | None, list[RecognizerResult]]:
    """Scrubs PII from text using the Presidio library.

    Parameters
    ----------
    text : str
        The text to scrub.
    analyzer : AnalyzerEngine
        The Presidio analyzer engine.
    threshold : float
        The score threshold for the filter.

    Returns
    -------
    tuple
        A tuple containing the scrubbed text and the filtered results.
    """
    try:
        results = analyzer.analyze(text=text, language='en')
        cleaned_results = [
            result for result in results
            if result.score >= threshold and result.entity_type not in EXCLUDE_TYPES
        ]
        anonymizer = AnonymizerEngine()
        anonymized_results = anonymizer.anonymize(text=text, analyzer_results=cleaned_results)
        return anonymized_results.text, cleaned_results

    except Exception as e:
        logging.error(f'Error scrubbing PII: {e}')
        raise


def write_scrubbed_txt(output_filename: str, anonymized_text: str) -> None:
    """Writes the anonymized text to an output file.

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
        logging.info(f'PII scrubbed successfully. Output written to {output_filename}')

    except Exception as e:
        logging.error(f'Error writing scrubbed output: {e}')
        raise


def write_confidence_record(filename: str, filtered_results: list, original_text: str) -> None:
    """Writes the filtered results to a JSON file.

    Parameters
    ----------
    filename : str
        Base filename for the output files.
    filtered_results : list
        The filtered Presidio results.
    original_text : str
        The original text.
    """
    try:
        results_list = [
            {
                'Original Text': original_text[result.start:result.end].replace('\n', ''),
                'Type': result.entity_type,
                'Start': result.start,
                'End': result.end,
                'Score': float(result.score)
            } for result in filtered_results
        ]
        with open(filename, 'w') as jsonfile:
            json.dump(results_list, jsonfile, indent=2)
        logging.info(f'PII scrubbed confidence record written to: {filename}')

    except Exception as e:
        logging.error(f'Error writing confidence record: {e}')
        raise


def get_output_strategy_from_path(file_path: str) -> str:
    """Determines the type of path based on path segments.

    Parameters
    ----------
    file_path : str
        The file_path to check.

    Returns
    -------
    str
        The extracted path type.
    """
    parts = file_path.split(os.sep)
    if 'doc' in parts:
        return 'doc'
    elif 'page' in parts:
        return 'page'
    elif "unstructured_text" in parts:
        return parts[parts.index("unstructured_text") + 1]
    else:
        return 'page'  # Default to page


def process_files(filepath_data: list, analyzer: AnalyzerEngine,
                  output_dir: str, threshold: float) -> defaultdict[Any, list]:
    """Processes a list of files, scrubbing PII and writing outputs.

    Parameters
    ----------
    filepath_data : list
        List of file data dictionaries.
    analyzer : AnalyzerEngine
        The Presidio analyzer engine.
    output_dir : str
        The output directory to save files to.
    threshold : float
        The confidence threshold for PII detection.

    Returns
    -------
    defaultdict[Any, list]
        A dictionary containing the PII results and the filtered results.
    """
    output_files = defaultdict(list)
    for item in filepath_data:
        input_file = item['output_filepath']
        output_strategy = get_output_strategy_from_path(input_file)

        with open(str(input_file), 'r') as f:
            orig_text = f.read()

        scrubbed_text, result_output = scrub_pii(orig_text, analyzer, threshold)

        input_filename = os.path.basename(str(input_file))
        filename_without_extension = os.path.splitext(input_filename)[0]

        scrubbed_dir = f'{output_dir}/unstructured_text/scrubbed_{output_strategy}'
        os.makedirs(scrubbed_dir, exist_ok=True)  # Create directory if needed
        output_file = f'{scrubbed_dir}/{filename_without_extension}.txt'
        write_scrubbed_txt(output_file, scrubbed_text)
        item['scrubbed_output_filepath'] = output_file
        output_files[output_strategy].append(output_file)

        confidence_dir = f'{output_dir}/unstructured_text/scrubbed_confidence'
        os.makedirs(confidence_dir, exist_ok=True)
        confidence_file = f'{confidence_dir}/confidence-{filename_without_extension}.json'
        write_confidence_record(confidence_file, result_output, orig_text)
        item['scrubbed_confidence_filepath'] = confidence_file
        output_files['confidence'].append(confidence_file)

    return output_files


if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description='Scrub PII from text files.')
    parser.add_argument('filemap_filepath_pattern', type=str, help='Path to the filemap output.')
    parser.add_argument('output_to', type=str, help='The output directory to save files to.')
    parser.add_argument('--config', required=False, type=str,
                        default=f'{script_dir}/config/stanford-deidentifier-base_nlp.yaml',
                        help='The config file for the NLP engine.')
    parser.add_argument('--threshold', required=False, type=float, default=0.0,
                        help='The target confidence threshold for scores.')
    args = parser.parse_args()

    nlp_engine = create_nlp_engine(args.config)

    for filepath in glob.glob(args.filemap_filepath_pattern):
        with open(filepath, 'r') as f:
            filepath_data = json.load(f)

        output_files = process_files(filepath_data, nlp_engine, args.output_to, args.threshold)

        # Update the filemap with the scrubbed files paths.
        with open(filepath, 'w') as filemap:
            json.dump(filepath_data, filemap, indent=2)

        logging.info(f"Processed files: {output_files}")
