import argparse
from vmrt_tesseract_utilities.report_data import ReportData, process_report_data


def _apply_replacement_list(report_data: ReportData) ->None:
    pass

def apply_replacement_list(args: argparse.Namespace):
    process_report_data(args.filemap_input, f'{args.output_to}/')


def parse_args() -> argparse.Namespace:
    """
    Parses the required args.

    Returns
    -------
    args: argparse.Namespace
        The parsed args.
    """
    parser = argparse.ArgumentParser(
        description='Applies case insensitive csv replacement list to text files in a directory.')
    parser.add_argument('filemap_input', type=str, help='Path to the filemap output.')
    parser.add_argument('replacement_list', type=str, help='A csv with input and output columns.')
    parser.add_argument('output_to', type=str, help='The output directory to save files to.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()