import logging

"""
A simple logger configured to write output to stdout.
"""

stdout_logger = logging.getLogger('import_utility')
formatter = logging.Formatter('%(asctime)s: %(message)s')
stdout_logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)
stdout_logger.addHandler(ch)
