import argparse
import json
import re
import os

"""
Creates a file list for a given directory and sub directories, extracting GRLS data from the path.
"""

# Pattern to match and extract the dog id.
dog_id_re = r"(094-[0-9]{6})"

# Various enrollment statuses expressed in the folder names.
enrollment_statuses = ('enrolled deceased', 'withdrawn', 'enactive')


def create_file_list(path: str) -> list:
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
        dog_match = re.search(dog_id_re, subdir)
        if dog_match:
            dog_id = dog_match.group(0)
        else:
            continue
        status = 'unknown'
        for possible_status in enrollment_statuses:
            if subdir.lower().find(possible_status) > -1:
                status = possible_status
                break
        for file in files:
            filepath = subdir + os.sep + file
            ext = os.path.splitext(file)[1]
            output_item = {
                'dog_id': dog_id,
                'enrollment_status': status,
                'filepath': filepath,
                'ext': ext,
            }
            output.append(output_item)
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
    file_list = create_file_list(provided_args.path_to_describe)
    output_list(file_list)
