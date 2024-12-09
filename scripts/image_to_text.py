import argparse
import json
import os

from vmrt_tesseract_utilities.report_data import ReportData
from vmrt_tesseract_utilities.tesseract_operations import (
    TesseractOperationBlock, TesseractOperationDoc, TesseractOperationPage)

from scripts.scrubbers import pii_scrubber

"""
Runs Tesseract on items in a file map to extract text and calculate confidence scores.
"""


def get_output_strategy(output_directory: str) -> callable:
    """
    Returns an output strategy for use in our Tesseract operations.

    Parameters
    ----------
    output_directory: str
      The directory where we want our output.

    Returns
    -------
    output_strategy: callable
      The strategy callable.
    """
    def output_strategy(strategy_type: str, row: ReportData, ocr_result: str) -> None:
        if len(ocr_result) > 0:
            base_path = f'{output_directory}/unstructured_text/{strategy_type}'
            if not os.path.exists(base_path):
                os.makedirs(base_path)
            file_name = row.get('origin_filename')
            count = 1
            if 'page' in row.data:
                count = row.get('page')
            if 'block' in row.data:
                count = row.get('block')
            bits = os.path.splitext(os.path.basename(file_name))
            output_path = f'{base_path}/{bits[0]}-{count}.txt'
            row.set_output_file(output_path)
            with open(output_path, 'w') as f:
                f.write(ocr_result)
    return output_strategy


def get_scrubber_method(output_directory):
    """
    Returns a scrubber strategy for use in our Tesseract operations.

    Parameters
    ----------
    output_directory: str
      The directory where we want our output.

    Returns
    -------
    scrubber_method: callable
      The callable strategy function.
    """
    def scrubber_method(strategy_type: str, row: ReportData, ocr_result: str) -> None:
        if len(ocr_result) > 0:
            # Load the proper NLP engine.
            nlp_engine = pii_scrubber.create_nlp_engine('stanford-deidentifier-base_nlp.yaml')
            # Scrub the text.
            scrubbed_text, result_output = pii_scrubber.scrub_pii(ocr_result, nlp_engine, 0.4)
            # Output the data from the scrubbing.
            base_path = f'{output_directory}/scrubbed_text/{strategy_type}'
            file_name = row.get('origin_filename')
            filename_without_extension = os.path.splitext(file_name)[0]
            count = 1
            if 'page' in row.data:
                count = row.get('page')
            if 'block' in row.data:
                count = row.get('block')
            # Write the scrubbed text to a file.
            scrubbed_dir = f'{base_path}/scrubbed_{strategy_type}'
            os.makedirs(scrubbed_dir, exist_ok=True)
            output_file = f'{scrubbed_dir}/{filename_without_extension}-{count}.txt'
            pii_scrubber.write_scrubbed_txt(output_file, scrubbed_text)
            row.set('scrubbed_output_filepath', output_file)
            # Write the scrubbed confidence values to a file.
            confidence_dir = f'{base_path}/scrubbed_confidence'
            os.makedirs(confidence_dir, exist_ok=True)
            confidence_file = f'{confidence_dir}/confidence-{filename_without_extension}-{count}.json'
            pii_scrubber.write_confidence_record(confidence_file, result_output, ocr_result)
            row.set('scrubbed_confidence_filepath', confidence_file)
    return scrubber_method


def run_tesseract(args: argparse.Namespace) -> None:
    """
    Runs Tesseract on a list of JSON objects describing files.

    Notes
    -----
    Provides both a modified version of the input JSON and dumps extracted text.

    Parameters
    ----------
    args: argparse.Namespace
      The args from the CLI.
    """
    if args.strategy not in ('doc', 'page', 'block',):
        raise ValueError("'--strategy' must be either 'doc', 'page' or 'block'")
    print(f'Here we go with strategy: {args.strategy}')
    output_strategy = get_output_strategy(args.output_to)
    scrubber_method = get_scrubber_method(args.output_to)
    if args.strategy == 'doc':
        op = TesseractOperationDoc(output_strategy, scrubber_method)
    if args.strategy == 'page':
        op = TesseractOperationPage(output_strategy, scrubber_method)
    if args.strategy == 'block':
        op = TesseractOperationBlock(output_strategy, scrubber_method)
    with open(args.input_file) as f:
        json_map = json.load(f)
    total_items = len(json_map)
    output = []
    for ikey, item in enumerate(json_map):
        item_object = ReportData(data=item)
        if ikey < args.offset:
            continue
        if ikey >= args.offset + args.chunk_size:
            break
        if item_object.get('origin_ext') != 'pdf':
            print(f'Skipping item {ikey} not a pdf file')
            continue
        if not os.path.isfile(item_object.get('origin_filepath')):
            print(f'Skipping item {ikey} file not found')
            continue
        if os.path.getsize(item_object.get('origin_filepath')) == 0:
            print(f'Skipping item {ikey} file is empty')
            continue
        print(f'Processing item {ikey}/{args.offset + args.chunk_size} ({total_items})...')
        output += op.process_row(item_object)
    with open(f'{args.output_to}/filemap_confidence-{args.offset}-{(args.offset + args.chunk_size)}.json', 'w') as f:
        json.dump(output, f, indent=2)


def parse_args() -> argparse.Namespace:
    """
    Calculates Tesseract confidence scores for a list of file describing JSON objects, outputting modified JSON.

    Returns
    -------
    args: argparse.Namespace
        The parsed args.
    """
    parser = argparse.ArgumentParser(
        prog='Outputs the change in two Fisher inventory files.')
    parser.add_argument('input_file', help='Path to the input filemap.')
    parser.add_argument('output_to', help='Path to the output directory.')
    parser.add_argument('--chunk_size', type=int, default=1000)
    parser.add_argument('--offset', type=int, default=0)
    parser.add_argument('--strategy', type=str, default='page')
    return parser.parse_args()


if __name__ == '__main__':
    provided_args = parse_args()
    run_tesseract(provided_args)
