import extract_msg

import argparse
import json
import re
import os

"""
Creates a file list for a given directory and sub directories, extracting GRLS data from the path.
"""


def extract_msg_files(path: str) -> list:
    """
    Creates a file list of json objects describing files for a given directory.

    Parameters
    ----------
    path: str
      The path to look for files in.

    Returns
    -------
      A list of json objects.

    """
    output = []
    for subdir, dirs, files in os.walk(path):
        for file in files:
            filepath = subdir + os.sep + file
            bits = os.path.splitext(file)
            ext = bits[1]
            if ext == '.msg':
                try:
                    msg = extract_msg.openMsg(filepath)
                    msg.save(customPath=f'{subdir}/{bits[0]}_extracted')
                    print(f'Extracted msg to {subdir}/{bits[0]}_extracted')
                except Exception as e:
                    print(e)
    return output


def output_list(file_list: list) -> None:
    """
    Prints our list of json objects.

    Parameters
    ----------
    file_list: list
      A list of json objects.
    """
    print(json.dumps(file_list))


def parse_args() -> argparse.Namespace:
    """
    Parses the required args.

    Returns
    -------
    args: argparse.Namespace
        The parsed args.
    """
    parser = argparse.ArgumentParser(
        prog='Outputs a list of files found in the provided directory/sub directories with GRLS data abstracted.')
    parser.add_argument('path_to_describe')
    return parser.parse_args()


if __name__ == '__main__':
    provided_args = parse_args()
    file_list = extract_msg_files(provided_args.path_to_describe)
    output_list(file_list)

