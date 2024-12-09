from abc import ABC, abstractmethod
from copy import deepcopy

import pdf2image
import tesserocr

from vmrt_tesseract_utilities.report_data import ReportData

"""
Provides Tesseract processing with different output scales (document, page, block).
"""


class TesseractOperationBase(ABC):
    """
    A base operation class for further use.
    """

    def __init__(self, output_strategy=None):
        if output_strategy is not None and not callable(output_strategy):
            raise TypeError('output_strategy must be callable or None.')
        self.output_strategy = output_strategy

    def __output_ocr_data__(self, strategy_type: str, row: ReportData, ocr_output: str):
        if self.output_strategy is not None:
            # If we have an output strategy, call it.
            self.output_strategy(strategy_type, row, ocr_output)

    @abstractmethod
    def process_row(self, row: ReportData) -> list:
        pass


class TesseractOperationDoc(TesseractOperationBase):
    """
    Returns Tesseract results for entire documents.
    """
    def process_row(self, row: ReportData) -> list:
        doc_row = deepcopy(row)
        page_content = ''
        try:
            origin_path = row.get('origin_filepath')
            images = pdf2image.convert_from_path(origin_path)
            print(f'Item {origin_path} has {len(images)} pages.')
            confidence_scores = []
            for pg, image in enumerate(images):
                with tesserocr.PyTessBaseAPI('/usr/share/tessdata') as api:
                    api.SetVariable('debug_file', '/dev/null')
                    api.SetImage(image)
                    ocr_result = api.GetUTF8Text()
                    confidence = api.MeanTextConf()
                    if len(ocr_result) > 0 and confidence > 0:
                        page_content += "\n" + ocr_result
                        confidence_scores.append(confidence)
            doc_confidence = sum(confidence_scores) / len(confidence_scores)
            print(f'Item {origin_path} had a confidence score of {doc_confidence}.')
            doc_row.set('status', 'processed') \
                .set('confidence', doc_confidence)
        except Exception as e:
            doc_row.set('status', 'error')
            print(str(e))
        self.__output_ocr_data__('doc', doc_row, page_content)
        return [doc_row]


class TesseractOperationPage(TesseractOperationBase):
    """
    Returns Tesseract results for entire pages.
    """
    def process_row(self, row: ReportData) -> list:
        output = []
        page_row = None
        try:
            origin_path = row.get('origin_filepath')
            images = pdf2image.convert_from_path(origin_path)
            print(f'Item {origin_path} has {len(images)} pages.')
            for pg, image in enumerate(images):
                page_row = deepcopy(row)
                page_row.set('page', pg + 1)
                with tesserocr.PyTessBaseAPI('/usr/share/tessdata') as api:
                    api.SetVariable('debug_file', '/dev/null')
                    api.SetImage(image)
                    ocr_result = api.GetUTF8Text()
                    page_confidence = api.MeanTextConf()
                    self.__output_ocr_data__('page', page_row, ocr_result)
                print(f'Item {origin_path}:{pg} had a confidence score of {page_confidence}.')
                page_row.set('status', 'processed') \
                    .set('confidence', page_confidence)
                output.append(page_row)
        except Exception as e:
            if page_row is not None:
                page_row.set('status', 'error')
                output.append(page_row)
            print(str(e))
        return output


class TesseractOperationBlock(TesseractOperationBase):
    """
    Returns Tesseract results for individual blocks of text.
    """
    def process_row(self, row: ReportData) -> list:
        output = []
        block_row = None
        try:
            origin_path = row.get('origin_filepath')
            images = pdf2image.convert_from_path(origin_path)
            print(f'Item {origin_path} has {len(images)} pages.')
            for pg, image in enumerate(images):
                with tesserocr.PyTessBaseAPI('/usr/share/tessdata') as api:
                    api.SetImage(image)
                    boxes = api.GetComponentImages(tesserocr.RIL.TEXTLINE, True)
                    for i, (im, box, _, _) in enumerate(boxes):
                        block_row = deepcopy(row)
                        block_row.set('block', i + 1)
                        api.SetRectangle(box['x'], box['y'], box['w'], box['h'])
                        ocr_result = api.GetUTF8Text()
                        box_confidence = api.MeanTextConf()
                        self.__output_ocr_data__('block', block_row, ocr_result)
                        print(f'Item {origin_path}:{pg}:{i} had a confidence score of {box_confidence}.')
                        block_row.set('status', 'processed') \
                            .set('confidence', box_confidence)
                        output.append(block_row)
        except Exception as e:
            if block_row is not None:
                block_row.set('status', 'error')
                output.append(block_row)
            print(str(e))
        return output
