import argparse
import json
import os

from presidio_analyzer import AnalyzerEngine, RecognizerResult
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine

from vmrt_tesseract_utilities.database import (TranscriptionInput,
                                               TranscriptionOutput,
                                               get_database_session)
from vmrt_tesseract_utilities.logging import stdout_logger
from vmrt_tesseract_utilities.scrubbing_utils import write_scrubbed_txt

"""
Leverages presidio to attempt automatic PII stripping.
"""

EXCLUDE_TYPES = frozenset(['IN_PAN'])  # Use frozenset for efficient lookups.


def create_nlp_engine(nlp_config_file: str) -> AnalyzerEngine:
    """
    Creates and configures a Presidio AnalyzerEngine.

    Sets up a Presidio AnalyzerEngine using the provided
    configuration file, which specifies the NLP engine and other settings.

    Parameters
    ----------
    nlp_config_file : str
        The filename of the YAML configuration file for the NLP engine.

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
        return AnalyzerEngine(nlp_engine=engine)

    except Exception as e:
        stdout_logger.error(f'Error creating NLP engine: {e}')
        raise


def scrub_pii(text: str, analyzer: AnalyzerEngine, threshold: float) -> tuple[str | None, list[RecognizerResult]]:
    """
    Scrubs PII from text using the Presidio library.

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
        stdout_logger.error(f'Error scrubbing PII: {e}')
        raise


def write_confidence_record(filename: str, filtered_results: list, original_text: str) -> None:
    """
    Writes the filtered results to a JSON file.

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
    except Exception as e:
        stdout_logger.error(f'Error writing confidence record: {e}')
        raise


def process_files(process_filepath_data: list, analyzer: AnalyzerEngine,
                  output_dir: str, threshold: float) -> None:
    """
    Processes a list of files, scrubbing PII and writing outputs.

    Parameters
    ----------
    process_filepath_data : list
        A list of file data dictionaries.
    analyzer : AnalyzerEngine
        The Presidio analyzer engine.
    output_dir : str
        The output directory to save files to.
    threshold : float
        The confidence threshold for PII detection.
    """
    sessionmaker = get_database_session(echo=args.debug_sql)
    with sessionmaker() as session:
        for output_log in process_filepath_data:
            session.add(output_log)  # Ensure the instance is bound to the session
            # Use the replacement file if it exists, otherwise use the OCR output.
            if hasattr(output_log, 'list_replacement_output_file') and output_log.list_replacement_output_file:
                input_filepath = output_log.list_replacement_output_file
            else:
                input_filepath = output_log.ocr_output_file
            # Read the original text from the input file.
            with open(str(input_filepath), 'r') as f:
                orig_text = f.read()
            stdout_logger.info(f"Scrubbing {input_filepath}")
            # Scrub the PII from the text.
            scrubbed_text, result_output = scrub_pii(orig_text, analyzer, threshold)
            # Write the scrubbed text to a file.
            input_filename = os.path.basename(str(input_filepath))
            filename_without_extension = os.path.splitext(input_filename)[0]
            scrubbed_dir = f'{output_dir}/scrubbed_text/{args.document_type}/scrubbed_{args.document_type}'
            os.makedirs(scrubbed_dir, exist_ok=True)  # Create directory if needed
            output_file = f'{scrubbed_dir}/{filename_without_extension}.txt'
            write_scrubbed_txt(output_file, scrubbed_text)
            stdout_logger.info(f"Scrubbed file written to {output_file}")
            # Write the scrubbed confidence data to a file.
            output_log.pii_scrubber_output_file = output_file
            confidence_dir = f'{output_dir}/scrubbed_text/{args.document_type}/scrubbed_confidence'
            os.makedirs(confidence_dir, exist_ok=True)
            confidence_file = f'{confidence_dir}/confidence-{filename_without_extension}.json'
            output_log.pii_scrubber_confidence_file = confidence_file
            write_confidence_record(confidence_file, result_output, orig_text)
            # Log the changes to the database.
            session.add(output_log)
        session.commit()


def get_files_to_process(args: argparse.Namespace) -> list:
    """
    Gets a list of input files to process.

    Parameters
    ----------
    args: argparse.Namespace
        The parsed args.

    Returns
    -------
    results: list
      The list of input files.
    """
    sessionmaker = get_database_session(echo=args.debug_sql)
    with sessionmaker.begin() as session:
        query = (session.query(TranscriptionOutput)
                 .outerjoin(TranscriptionInput.assets)
                 .where(TranscriptionInput.document_type == args.document_type)
                 .where(TranscriptionOutput.ocr_output_file != None)  # noqa: E711
                 .where(TranscriptionOutput.pii_scrubber_output_file == None)  # noqa: E711
                 .limit(args.chunk_size)
                 .offset(args.offset))
    return query.all()


def parse_args() -> argparse.Namespace:
    """
    Parses the required args.

    Returns
    -------
    args: argparse.Namespace
        The parsed args.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description='Scrub PII from text files.')
    parser.add_argument('output_to', help='Path to the output directory.')
    parser.add_argument('--document-type', type=str, default='document',
                        help='The document type we want to produce, document, page or block.')
    parser.add_argument('--chunk-size', type=int, default=1000,
                        help='The number of records to process.')
    parser.add_argument('--offset', type=int, default=0,
                        help='The number of records to skip before beginning processing.')
    parser.add_argument('--debug-sql', action='store_true', help='Enable SQL debugging')
    parser.add_argument('--config', required=False, type=str,
                        default=f'{script_dir}/config/stanford-deidentifier-base_nlp.yaml',
                        help='The config file for the NLP engine.')
    parser.add_argument('--threshold', required=False, type=float, default=0.0,
                        help='The target confidence threshold for scores.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    db_output_logs = get_files_to_process(args)
    nlp_engine = create_nlp_engine(args.config)
    process_files(db_output_logs, nlp_engine, args.output_to, args.threshold)
