import os
from typing import Self

# Required for object serialization.
import json_fix

"""
Provides a serializable, validated ReportData object.
"""

# Keys our report row may contain.
item_keys = [
    'subject_id',
    'origin_filepath',
    'origin_filename',
    'origin_ext',
    'status',
    'granularity',
    'confidence',
    'output_filepath',
    'output_filename',
    'output_ext',
    'page',
    'block',
    'scrubbed_output_filepath',
    'scrubbed_confidence_filepath',
]


class ReportData:
    """
    A serializable, key validated representation of report data.
    """
    def __init__(self, **kwargs):
        data = kwargs.get('data', {})
        self.data = {}
        if len(data.keys()) > 0:
            self.set_data(data)

    def __json__(self):
        return self.data

    def set_data(self, data: dict) -> Self:
        if len(data.keys()) > 0 and not all(k in item_keys for k in data.keys()):
            invalid_keys = ','.join(list(set(data.keys()) - set(item_keys)))
            raise RuntimeError(f'Invalid data keys: {invalid_keys}')
        for key, value in data.items():
            self.data[key] = value
        return self

    def set(self, key, value):
        if key not in item_keys:
            raise RuntimeError(f'Invalid data key: {key}')
        self.data[key] = value
        return self

    def set_origin_file(self, path) -> Self:
        self.__set_file__('origin', path)
        return self

    def set_output_file(self, path) -> Self:
        self.__set_file__('output', path)
        return self

    def __set_file__(self, seed, path):
        bits = os.path.splitext(path)
        values = {
            f'{seed}_filepath': path,
            f'{seed}_filename': os.path.basename(path),
            f'{seed}_ext': bits[1].replace('.', ''),
        }
        self.set_data(values)

    def get(self, key):
        if key not in self.data.keys():
            raise RuntimeError(f'Invalid data key: {key}')
        return self.data[key]
