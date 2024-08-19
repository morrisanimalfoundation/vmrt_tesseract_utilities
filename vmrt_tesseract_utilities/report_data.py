import os
from typing import Self

import json_fix


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
]

class ReportData:
    def __init__(self, **kwargs):
        self.set_data(kwargs.get('data', {}))

    def __json__(self):
        return self.data

    def set_data(self, data: dict) -> Self:
        if len(data.keys()) > 0 and not all(k in item_keys for k in data.keys()):
            invalid_keys = ','.join(list(set(data.keys()) - set(item_keys)))
            raise RuntimeError(f'Invalid data keys: {invalid_keys}')
        if not hasattr(self, 'data'):
            self.data = {}
        for key, value in data.items():
            self.data[key] = value
        return self


    def set(self, key, value):
        if not key in item_keys:
            raise RuntimeError(f'Invalid data key: {key}')
        self.data[key] = value
        return self



    def set_origin_file(self, path) -> Self:
        self.__set_file('origin', path)
        return self


    def set_output_file(self, path) -> Self:
        self.__set_file('output', path)
        return self

    def __set_file(self, seed, path):
        bits = os.path.splitext(path)
        print(bits)
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
