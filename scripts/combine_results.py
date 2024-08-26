import argparse
import json
import os

import numpy as np
import pandas as pd

"""
Utility script for combining Tesseract processing results.
"""


def combine_results(args: argparse.Namespace) -> None:
    """
    Combines a director of results with optional confidence report.

    Parameters
    ----------
    args: argparse.Namespace
        The parsed args.
    """
    records = []
    with os.scandir(args.input_path) as it:
        for entry in it:
            if entry.name.endswith(".json") and entry.is_file():
                with open(entry) as f:
                    contents = json.load(f)
                    if len(contents) > 0:
                        records = records + contents
    df = pd.DataFrame(records)
    df['confidence'] = df['confidence'].astype(np.float64)
    if args.report_bins:
        bins = [-np.inf, 1, 60, 70, 80, 85, 90, 95, np.inf]
        labels = ['0', '1-60', '60-70', '70-80', '80-85', '80-90', '90-95', '95-inf']
        df['bin'] = pd.cut(df['confidence'], bins, labels=labels)
        print(df['bin'].value_counts(dropna=False).sort_index())
        df = df.drop(columns='bin')
    df.to_csv(args.output_path, index=False)


def parse_args() -> argparse.Namespace:
    """
    Parses the required args.

    Returns
    -------
    args: argparse.Namespace
        The parsed args.
    """
    parser = argparse.ArgumentParser(
        prog='Combines JSON results from image to text script.')
    parser.add_argument('input_path', help='Path to JSON files that need to be combined.')
    parser.add_argument('output_path', help='Where to output the combined results.')
    parser.add_argument('--report_bins', help='Print a report of confidence bins.', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    provided_args = parse_args()
    combine_results(provided_args)
