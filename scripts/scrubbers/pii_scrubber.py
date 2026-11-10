import argparse
import json
import os
import re
import sys

import yaml
from presidio_analyzer import (AnalyzerEngine, Pattern, PatternRecognizer,
                               RecognizerResult)
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig
from sqlalchemy import inspect
from sqlalchemy.pool import NullPool

from vmrt_tesseract_utilities.cli import normalize_flag_spelling
from vmrt_tesseract_utilities.database import (TranscriptionInput,
                                               TranscriptionOutput,
                                               get_database_session)
from vmrt_tesseract_utilities.grls_id import ocr_subject_id_regex
from vmrt_tesseract_utilities.logging import log_run_summary, stdout_logger
from vmrt_tesseract_utilities.scrubbing_utils import (split_text,
                                                      write_scrubbed_txt)

"""
Leverages presidio to attempt automatic PII stripping.
"""

EXCLUDE_TYPES = frozenset(['IN_PAN'])  # Use frozenset for efficient lookups.

# The values of TranscriptionOutput.pii_scrubber_status. A file is STARTED while it's being scrubbed, RETRYING while
# it's being scrubbed again after a run was cut short on it, and SUCCESS or FAILED once it's done. NULL means it
# hasn't been tried.
STATUS_STARTED = 'started'
STATUS_RETRYING = 'retrying'
STATUS_SUCCESS = 'success'
STATUS_FAILED = 'failed'

# The error saved for a file whose scrubbing was cut short twice, so a file that always gets its run killed (usually
# for running out of memory) is skipped like any other failure instead of stopping every later run.
INTERRUPTED_TWICE_ERROR = ('ScrubInterrupted: scrubbing this file was cut short twice before it could finish: its '
                           'process was killed, most likely for running out of memory, or the run was stopped. Try '
                           'it again with --retry-failed.')

# The most characters of an error that are saved, well within a MySQL TEXT column's 65,535 bytes.
MAX_ERROR_LENGTH = 10_000

# The most characters the analyzer is given at once. Longer texts are analyzed in pieces of up to this length.
# spaCy refuses texts over 1,000,000 characters, and needs about 1 GB of memory per 100,000, so one OCR'd study's
# worth of records (over a million characters) can't be analyzed whole.
MAX_ANALYZE_LENGTH = 100_000

# Adds the columns this script records failures in to a database created before it did.
ADD_STATUS_COLUMNS_SQL = ('ALTER TABLE transcription_output '
                          'ADD COLUMN pii_scrubber_status VARCHAR(20), ADD COLUMN pii_scrubber_error TEXT;')

# GRLS subject IDs (e.g. "094-012345"), including the forms they take in OCR'd text: a space, en dash or nothing
# in place of the dash, a letter O read for the leading zero, and an extra zero-padded digit. The model's ID
# detections are scaled down to 0.4 by the NLP config (low_confidence_score_multiplier), so matching subject IDs
# by pattern keeps them above any reasonable threshold.
SUBJECT_ID_PATTERN = Pattern(name='GRLS subject ID', regex=ocr_subject_id_regex(), score=0.9)

# The same pattern, compiled with the flags PatternRecognizer uses by default, so a long text is never split into
# pieces in the middle of a subject ID, including one the pattern matches across a space or line break.
SUBJECT_ID_RE = re.compile(SUBJECT_ID_PATTERN.regex, re.DOTALL | re.MULTILINE | re.IGNORECASE)


def replace_id(value: str) -> str:
    """
    Replaces an ID with <ID>, keeping any line breaks it was split across, so the lines around it stay apart.

    Parameters
    ----------
    value : str
        The ID's text.

    Returns
    -------
    str
        The replacement.

    Examples
    --------
    >>> replace_id('094-\\n\\n012345')
    '<ID>\\n\\n'
    >>> replace_id('094-012345')
    '<ID>'
    """
    return '<ID>' + '\n' * value.count('\n')


# How each entity type is replaced. Types not listed get the default, <ENTITY_TYPE>.
ANONYMIZE_OPERATORS = {'ID': OperatorConfig('custom', {'lambda': replace_id})}


def create_subject_id_recognizer() -> PatternRecognizer:
    """
    Creates the pattern recognizer for GRLS subject IDs.

    Subject IDs are reported as ID, the same entity type the model uses, so they are still replaced with <ID>.

    Returns
    -------
    PatternRecognizer
        The subject ID recognizer.
    """
    return PatternRecognizer(supported_entity='ID', name='GrlsSubjectIdRecognizer', patterns=[SUBJECT_ID_PATTERN])


def find_missing_model_folders(nlp_config_file: str) -> list[str]:
    """
    Lists the transformers model folders an NLP config names that don't exist.

    The image has the scrubber's model in a fixed folder, which the config names by path. In an image built before
    the model was added, or outside the image, the folder is missing, and transformers would otherwise report it as
    an invalid Hugging Face model ID.

    Parameters
    ----------
    nlp_config_file : str
        The filename of the YAML configuration file for the NLP engine.

    Returns
    -------
    list[str]
        The missing folders. Models named by Hugging Face ID rather than by path are not checked.
    """
    with open(nlp_config_file) as f:
        config = yaml.safe_load(f)
    folders = [model['model_name']['transformers'] for model in config.get('models', [])
               if isinstance(model.get('model_name'), dict) and 'transformers' in model['model_name']]
    return [folder for folder in folders if os.path.isabs(folder) and not os.path.isdir(folder)]


def create_nlp_engine(nlp_config_file: str) -> AnalyzerEngine:
    """
    Creates and configures a Presidio AnalyzerEngine.

    Sets up a Presidio AnalyzerEngine using the provided
    configuration file, which specifies the NLP engine and other settings,
    and adds a pattern recognizer for GRLS subject IDs.

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
    FileNotFoundError
        If a model folder the config names doesn't exist.
    Exception
        If any other error occurs during the creation of the NLP engine.
    """
    missing = find_missing_model_folders(nlp_config_file)
    if missing:
        raise FileNotFoundError(
            f'The PII scrubber model folder {", ".join(missing)} does not exist. The model is built into the '
            f'workspace image, so run this inside a current image: ./run.sh pulls the latest one, or use '
            f'./run.sh --build.')
    try:
        provider = NlpEngineProvider(conf_file=nlp_config_file)
        engine = provider.create_engine()
        analyzer = AnalyzerEngine(nlp_engine=engine)
        analyzer.registry.add_recognizer(create_subject_id_recognizer())
        return analyzer

    except Exception as e:
        stdout_logger.error(f'Error creating NLP engine: {e}')
        raise


def scrub_pii(text: str, analyzer: AnalyzerEngine, threshold: float,
              max_length: int = MAX_ANALYZE_LENGTH) -> tuple[str | None, list[RecognizerResult]]:
    """
    Scrubs PII from text using the Presidio library.

    Text longer than max_length is analyzed in pieces that end at a line break where possible (see `split_text`),
    and the results are shifted to their place in the whole text, which is then anonymized at once. A piece never
    ends inside a subject ID. Another entity is only missed for being split if it runs across a piece's boundary,
    and a boundary only falls inside a line when max_length characters go by without a line break.

    Parameters
    ----------
    text : str
        The text to scrub.
    analyzer : AnalyzerEngine
        The Presidio analyzer engine.
    threshold : float
        The score threshold for the filter.
    max_length : int
        The most characters to analyze at once.

    Returns
    -------
    tuple
        A tuple containing the scrubbed text and the filtered results, with offsets in the whole text.
    """
    results = []
    for offset, piece in split_text(text, max_length, keep_together=SUBJECT_ID_RE):
        for result in analyzer.analyze(text=piece, language='en'):
            result.start += offset
            result.end += offset
            results.append(result)
    cleaned_results = [
        result for result in results
        if result.score >= threshold and result.entity_type not in EXCLUDE_TYPES
    ]
    anonymizer = AnonymizerEngine()
    anonymized_results = anonymizer.anonymize(text=text, analyzer_results=cleaned_results,
                                              operators=ANONYMIZE_OPERATORS)
    return anonymized_results.text, cleaned_results


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


def output_paths(args: argparse.Namespace, input_filepath: str) -> tuple[str, str]:
    """
    Gets where a file's scrubbed text and confidence report are written.

    Parameters
    ----------
    args : argparse.Namespace
        The parsed args.
    input_filepath : str
        The text file to scrub.

    Returns
    -------
    tuple[str, str]
        The paths of the scrubbed text file and the confidence report.
    """
    filename_without_extension = os.path.splitext(os.path.basename(input_filepath))[0]
    scrubbed_dir = f'{args.output_to}/scrubbed_text/{args.document_type}'
    return (f'{scrubbed_dir}/scrubbed_{args.document_type}/{filename_without_extension}.txt',
            f'{scrubbed_dir}/scrubbed_confidence/confidence-{filename_without_extension}.json')


def scrub_file(args: argparse.Namespace, input_filepath: str, analyzer: AnalyzerEngine) -> tuple[str, str]:
    """
    Scrubs one text file, writing the scrubbed text and its confidence report.

    Parameters
    ----------
    args : argparse.Namespace
        The parsed args.
    input_filepath : str
        The text file to scrub.
    analyzer : AnalyzerEngine
        The Presidio analyzer engine.

    Returns
    -------
    tuple[str, str]
        The paths of the scrubbed text file and the confidence report.
    """
    with open(input_filepath, 'r') as f:
        orig_text = f.read()
    scrubbed_text, result_output = scrub_pii(orig_text, analyzer, args.threshold)
    output_file, confidence_file = output_paths(args, input_filepath)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    write_scrubbed_txt(output_file, scrubbed_text)
    stdout_logger.info(f"Scrubbed file written to {output_file}")
    os.makedirs(os.path.dirname(confidence_file), exist_ok=True)
    write_confidence_record(confidence_file, result_output, orig_text)
    return output_file, confidence_file


def remove_outputs(paths: tuple[str, ...]) -> None:
    """
    Removes whichever of a failed file's outputs were written, so none is left looking like a finished scrub.

    Parameters
    ----------
    paths : tuple[str, ...]
        The output paths.
    """
    for path in paths:
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
        except OSError as e:
            stdout_logger.error(f'Could not remove the partial output {path}: {e}')


def record_failure(session, output_log: TranscriptionOutput, input_filepath: str, error: str,
                   failures: list) -> None:
    """
    Logs that a file failed, adds it to the run's failures, and saves it as failed with its error.

    If the failure can't be saved, that's logged and the run carries on: the file is left as it was saved before
    scrubbing started, so a later run tries it again.

    Parameters
    ----------
    session : Session
        The open database session.
    output_log : TranscriptionOutput
        The file's row.
    input_filepath : str
        The file that failed.
    error : str
        The error.
    failures : list
        The run's (input file, error) failures, added to.
    """
    stdout_logger.error(f'Failed to scrub {input_filepath}: {error}')
    failures.append((input_filepath, error))
    if len(error) > MAX_ERROR_LENGTH:
        error = error[:MAX_ERROR_LENGTH] + '... (cut short)'
    output_log.pii_scrubber_status = STATUS_FAILED
    output_log.pii_scrubber_error = error
    try:
        session.commit()
    except Exception as e:
        session.rollback()
        stdout_logger.error(f'Could not save the failure of {input_filepath}: {type(e).__name__}: {e}')


def process_files(args: argparse.Namespace, process_filepath_data: list, analyzer: AnalyzerEngine,
                  failed_ids: set) -> list[tuple[str, str]]:
    """
    Scrubs a chunk of files, saving each one's result as it goes.

    A file that fails is saved as failed with its error, and the rest of the chunk carries on. Each file is saved
    as 'started' before it's scrubbed, so a run that's killed part way through one (by running out of memory, or a
    stopped build) leaves a record of it. A later run tries a 'started' file once more, as 'retrying', and if that
    attempt is cut short too, marks it failed instead of trying again (unless --retry-failed is passed, which
    scrubs it).

    Parameters
    ----------
    args : argparse.Namespace
        The parsed args.
    process_filepath_data : list
        The TranscriptionOutput rows to scrub.
    analyzer : AnalyzerEngine
        The Presidio analyzer engine.
    failed_ids : set
        The IDs of the rows that failed this run, added to.

    Returns
    -------
    list[tuple[str, str]]
        (input file, error) for each file that failed.
    """
    failures = []
    # This runs once per chunk, so don't keep a pool of connections open for each one.
    sessionmaker = get_database_session(echo=args.debug_sql, poolclass=NullPool)
    with sessionmaker() as session:
        for output_log in process_filepath_data:
            session.add(output_log)  # Ensure the instance is bound to the session
            row_id = output_log.id
            # Use the replacement file if it exists, otherwise use the OCR output.
            input_filepath = str(output_log.list_replacement_output_file or output_log.ocr_output_file)
            if output_log.pii_scrubber_status == STATUS_RETRYING and not args.retry_failed:
                # A killed run may have written some of the file's outputs before it stopped.
                remove_outputs(output_paths(args, input_filepath))
                failed_ids.add(row_id)
                record_failure(session, output_log, input_filepath, INTERRUPTED_TWICE_ERROR, failures)
                continue
            if output_log.pii_scrubber_status in (STATUS_STARTED, STATUS_RETRYING):
                stdout_logger.warning(f'{input_filepath}: an earlier run was cut short while scrubbing it, so it '
                                      f'is tried once more.')
                output_log.pii_scrubber_status = STATUS_RETRYING
            else:
                output_log.pii_scrubber_status = STATUS_STARTED
            # An error from an earlier attempt doesn't describe this one.
            output_log.pii_scrubber_error = None
            # Saved before scrubbing starts, so it's still there if this run is killed part way through the file.
            session.commit()
            stdout_logger.info(f"Scrubbing {input_filepath}")
            try:
                output_log.pii_scrubber_output_file, output_log.pii_scrubber_confidence_file = \
                    scrub_file(args, input_filepath, analyzer)
                output_log.pii_scrubber_status = STATUS_SUCCESS
                # Commit per file, so a crash part way through a chunk only loses the file in progress.
                session.commit()
            except Exception as e:
                session.rollback()
                remove_outputs(output_paths(args, input_filepath))
                failed_ids.add(row_id)
                record_failure(session, output_log, input_filepath, f'{type(e).__name__}: {e}', failures)
    return failures


def get_files_to_process(args: argparse.Namespace, failed_ids: frozenset = frozenset()) -> list:
    """
    Gets a list of input files to process.

    Parameters
    ----------
    args: argparse.Namespace
        The parsed args.
    failed_ids: frozenset
        The IDs of rows that failed this run, which aren't returned again.

    Returns
    -------
    results: list
      The list of input files.
    """
    # This runs once per chunk, so don't keep a pool of connections open for each one.
    sessionmaker = get_database_session(echo=args.debug_sql, poolclass=NullPool)
    # Load the rows inside the session, so closing it detaches them for process_files to attach to its own.
    with sessionmaker() as session:
        query = (session.query(TranscriptionOutput)
                 .outerjoin(TranscriptionInput.assets)
                 .where(TranscriptionInput.document_type == args.document_type)
                 .where(TranscriptionOutput.ocr_output_file != None)  # noqa: E711
                 .where(TranscriptionOutput.pii_scrubber_output_file == None))  # noqa: E711
        if not args.retry_failed:
            query = query.where(TranscriptionOutput.pii_scrubber_status.is_distinct_from(STATUS_FAILED))
        if failed_ids:
            query = query.where(TranscriptionOutput.id.notin_(failed_ids))
        query = (query.order_by(TranscriptionOutput.id)
                 .limit(args.chunk_size)
                 .offset(args.offset))
        results = query.all()
    return results


def scrub_chunks(args: argparse.Namespace, analyzer: AnalyzerEngine) -> list[tuple[str, str]]:
    """
    Scrubs one chunk of files, or with --until-done, keeps scrubbing chunks until no files are left.

    Files that fail aren't picked again in the same run, so with --retry-failed each is tried at most once per run.

    Parameters
    ----------
    args : argparse.Namespace
        The parsed args.
    analyzer : AnalyzerEngine
        The Presidio analyzer engine.

    Returns
    -------
    list[tuple[str, str]]
        (input file, error) for each file that failed.
    """
    failures = []
    failed_ids = set()
    count = 0
    while True:
        db_output_logs = get_files_to_process(args, frozenset(failed_ids))
        stdout_logger.info(f'Found {len(db_output_logs)} files to scrub.')
        if not db_output_logs:
            break
        failures += process_files(args, db_output_logs, analyzer, failed_ids)
        count += len(db_output_logs)
        if not args.until_done:
            break
    log_run_summary('Scrubbed', count, failures,
                    note=' and were skipped (see pii_scrubber_error in transcription_output); pass --retry-failed '
                         'to try them again')
    return failures


def check_status_columns(args: argparse.Namespace) -> None:
    """
    Checks that the database has the columns failures are recorded in, before any file is scrubbed.

    `database_setup.py install` doesn't add columns to existing tables, so a database created before this script
    recorded failures doesn't have them.

    Parameters
    ----------
    args : argparse.Namespace
        The parsed args.

    Raises
    ------
    RuntimeError
        If the table or the columns are missing.
    """
    sessionmaker = get_database_session(echo=args.debug_sql, poolclass=NullPool)
    inspector = inspect(sessionmaker.kw['bind'])
    if not inspector.has_table('transcription_output'):
        raise RuntimeError('The database has no transcription_output table. Create the tables with '
                           '`python scripts/database_setup.py install` and run the earlier steps first.')
    columns = {column['name'] for column in inspector.get_columns('transcription_output')}
    if not {'pii_scrubber_status', 'pii_scrubber_error'} <= columns:
        raise RuntimeError(f'The transcription_output table has no pii_scrubber_status and pii_scrubber_error '
                           f'columns, which this script records failures in. The database was created before they '
                           f'were added: add them with\n  {ADD_STATUS_COLUMNS_SQL}')


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
    parser.add_argument('--until-done', action='store_true',
                        help='Keep scrubbing chunks until no files are left, instead of stopping after one.')
    parser.add_argument('--retry-failed', action='store_true',
                        help='Also scrub files that failed on an earlier run.')
    args = parser.parse_args(normalize_flag_spelling(sys.argv[1:]))
    # The offset would be applied to every chunk, so the run would stop with that many files still unscrubbed.
    if args.until_done and args.offset:
        parser.error('--offset can\'t be used with --until-done, which scrubs every file.')
    return args


if __name__ == '__main__':
    args = parse_args()
    try:
        check_status_columns(args)
    except RuntimeError as e:
        stdout_logger.error(f'Stopping before scrubbing anything: {e}')
        sys.exit(1)
    nlp_engine = create_nlp_engine(args.config)
    scrub_chunks(args, nlp_engine)
