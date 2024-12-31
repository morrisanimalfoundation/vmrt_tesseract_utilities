from abc import ABC, abstractmethod

import pdf2image
import tesserocr

from vmrt_tesseract_utilities.logging import stdout_logger

"""
Provides Tesseract processing with different output scales (document, page, block).
"""


class TesseractOperationBase(ABC):
    """
    A base operation class for further use.
    """

    @abstractmethod
    def process_row(self, input_path: str) -> list:
        pass


class TesseractOperationDoc(TesseractOperationBase):
    """
    Returns Tesseract results for entire documents.
    """
    def process_row(self, input_path: str) -> list:
        output = []
        try:
            images = pdf2image.convert_from_path(input_path)
            stdout_logger.info(f'Item {input_path} has {len(images)} pages.')
            confidence_scores = []
            document_content = ''
            for pg, image in enumerate(images):
                with tesserocr.PyTessBaseAPI('/usr/share/tessdata') as api:
                    api.SetVariable('debug_file', '/dev/null')
                    api.SetImage(image)
                    ocr_result = api.GetUTF8Text()
                    confidence = api.MeanTextConf()
                    if len(ocr_result) > 0 and confidence > 0:
                        document_content += "\n" + ocr_result
                        confidence_scores.append(confidence)
            doc_confidence = sum(confidence_scores) / len(confidence_scores)
            stdout_logger.info(f'Item {input_path} had a confidence score of {doc_confidence}.')
            output.append({'content': document_content, 'confidence': doc_confidence, 'page': 1, 'block': 1})
        except Exception as e:
            stdout_logger.error(str(e))
        return output


class TesseractOperationPage(TesseractOperationBase):
    """
    Returns Tesseract results for entire pages.
    """
    def process_row(self, input_path: str) -> list:
        output = []
        try:
            images = pdf2image.convert_from_path(input_path)
            stdout_logger.info(f'Item {input_path} has {len(images)} pages.')
            for pg, image in enumerate(images):
                with tesserocr.PyTessBaseAPI('/usr/share/tessdata') as api:
                    api.SetVariable('debug_file', '/dev/null')
                    api.SetImage(image)
                    ocr_result = api.GetUTF8Text()
                    page_confidence = api.MeanTextConf()
                print(f'Item {input_path}:{pg} had a confidence score of {page_confidence}.')
                output.append({'content': ocr_result, 'confidence': page_confidence, 'page': pg + 1, 'block': 0})
        except Exception as e:
            stdout_logger.error(str(e))
        return output


class TesseractOperationBlock(TesseractOperationBase):
    """
    Returns Tesseract results for individual blocks of text.
    """
    def process_row(self, input_path: str) -> list:
        output = []
        try:
            images = pdf2image.convert_from_path(input_path)
            stdout_logger.info(f'Item {input_path} has {len(images)} pages.')
            for pg, image in enumerate(images):
                with tesserocr.PyTessBaseAPI('/usr/share/tessdata') as api:
                    api.SetImage(image)
                    boxes = api.GetComponentImages(tesserocr.RIL.TEXTLINE, True)
                    for i, (im, box, _, _) in enumerate(boxes):
                        api.SetRectangle(box['x'], box['y'], box['w'], box['h'])
                        ocr_result = api.GetUTF8Text()
                        box_confidence = api.MeanTextConf()
                        print(f'Item {input_path}:{pg}:{i} had a confidence score of {box_confidence}.')
                        output.append({'content': ocr_result, 'confidence': box_confidence, 'page': pg + 1, 'block': i + 1})
        except Exception as e:
            stdout_logger.error(str(e))
        return output
