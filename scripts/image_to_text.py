import argparse
import json
import os

import pdf2image
import tesserocr

"""
Runs Tesseract on items in a file map to extract text and calculate confidence scores.
"""


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
    with open(args.input_file) as f:
        json_map = json.load(f)
    total_items = len(json_map)
    output = []
    for ikey, item in enumerate(json_map):
        if ikey < args.offset:
            continue
        if ikey >= args.offset + args.chunk_size:
            break
        if item['ext'] != '.pdf':
            print(f'Skipping item {ikey} not a pdf file')
            item['status'] = 'skipped'
            output.append(item)
            continue
        if not os.path.isfile(item['filepath']):
            print(f'Skipping item {ikey} file not found')
            item['status'] = 'file-not-found'
            output.append(item)
            continue
        if os.path.getsize(item['filepath']) == 0:
            print(f'Skipping item {ikey} file is empty')
            item['status'] = 'file-empty'
            output.append(item)
            continue
        confidence_scores = []
        try:
            print(f'Processing item {ikey + args.offset}/{args.offset + args.chunk_size} ({total_items})...')
            images = pdf2image.convert_from_path(item['filepath'])
            for pg, image in enumerate(images):
                with tesserocr.PyTessBaseAPI('/usr/share/tessdata') as api:
                    api.SetVariable('debug_file', '/dev/null')
                    api.SetImage(image)
                    ocr_result = api.GetUTF8Text()
                    page_confidence = api.MeanTextConf()
                    confidence_scores.append(page_confidence)
                    if len(ocr_result) > 0:
                        with open(f'{args.output_to}/unstructured_text/{os.path.basename(item["filepath"])}-{pg}.txt','w') as f:
                            f.write(ocr_result)
                print(f'Item {item["filepath"]}:{pg} had a confidence score of {page_confidence}.')
                item['status'] = 'processed'
        except Exception as e:
            item['status'] = 'error'
            print(str(e))
        if len(confidence_scores) > 0:
            item['confidence'] = sum(confidence_scores) / len(confidence_scores)
        else:
            item['confidence'] = 0
        print(f'Item {item["filepath"]} had an average confidence score of {item["confidence"]}.')
        output.append(item)
    with open(f'{args.output_to}/filemap_confidence-{args.offset}-{(args.offset + args.chunk_size)}.json', 'w') as f:
        json.dump(output, f)


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
    parser.add_argument('input_file')
    parser.add_argument('output_to')
    parser.add_argument('--chunk_size', type=int, default=1000)
    parser.add_argument('--offset', type=int, default=0)
    return parser.parse_args()


if __name__ == '__main__':
    provided_args = parse_args()
    run_tesseract(provided_args)
