"""
Sends PDFs transcribed before issue #82 (all with OCR) back through the pipeline, so pages whose PDF holds their text
get that text instead.

A PDF is redone when it was transcribed (its ocr_status is 'success', or empty in a database that got that column
after its PDFs were transcribed), its output rows have no text_source (they were made before #82), and at least one
of its pages would now be read from its text layer. Redoing it removes everything the
pipeline made from its old text, so every stage picks it up again on its next run, in the usual order:

- its transcription_output rows, their string replacement records, and the files they point to (the OCR text,
  replaced text, scrubbed text and scrubber confidence report);
- the dates the visit date miner found in it, leaving one empty metadata row per subject ID, as registration made;
- its ocr_status and ocr_error, so transcribe_pdfs.py reads it again.

A PDF whose pages would all still be read with OCR keeps its outputs, which are marked text_source 'ocr', so it isn't
checked again. So does a damaged PDF whose text layer poppler can't read, since step 2 would read every page with OCR
too. One whose text layer couldn't be read this time (poppler took too long or was killed) is left as it is, so the
next run checks it again; with --keep-unreadable, it keeps its outputs instead. Each PDF is changed in its own
transaction, so the script can be stopped and run again, and it only ever looks at PDFs it hasn't dealt with yet.

Run it while no pipeline stage is running, and with --dry-run first: that only counts what would be redone.

    python scripts/redo_text_layer.py --document-type=document --dry-run
    python scripts/redo_text_layer.py --document-type=document
"""

import argparse
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext

from sqlalchemy import and_, or_

from vmrt_tesseract_utilities.cli import normalize_flag_spelling
from vmrt_tesseract_utilities.database import (MissingColumnError,
                                               StringReplacementCompletion,
                                               TranscriptionInput,
                                               TranscriptionMetadata,
                                               TranscriptionOutput,
                                               check_text_source_column,
                                               get_database_session)
from vmrt_tesseract_utilities.logging import stdout_logger
from vmrt_tesseract_utilities.pdf_files import is_pdf_column
from vmrt_tesseract_utilities.text_layer import (DamagedPdfError,
                                                 TextLayerError, read_pages)

REDO = 'redo'
KEEP = 'keep'
UNREADABLE = 'unreadable'
# The poppler tools read_pages runs.
TEXT_LAYER_TOOLS = ('pdfinfo', 'pdfimages', 'pdftotext')


def decide(input_file: str) -> tuple[str, str]:
    """
    Decides whether a PDF needs redoing.

    Parameters
    ----------
    input_file : str
        The PDF.

    Returns
    -------
    tuple[str, str]
        REDO if at least one page would be read from its text layer, KEEP if none would (or poppler can't read the
        damaged file, so step 2 would read every page with OCR too), or UNREADABLE if poppler couldn't read its text
        layer this time (it took too long or was killed); and a few words on why.
    """
    try:
        pages = read_pages(input_file)
    except DamagedPdfError as e:
        return KEEP, f'its text layer could not be read, so every page would be read with OCR again: {e}'
    except TextLayerError as e:
        return UNREADABLE, str(e)
    used = sum(page.use_text_layer for page in pages)
    return (REDO if used else KEEP), f'{used} of {len(pages)} pages from the text layer'


def reset_input(session, db_input: TranscriptionInput) -> list[str]:
    """
    Removes what the pipeline made from a PDF's old text, so every stage processes it again. Doesn't commit.

    Parameters
    ----------
    session : Session
        The open database session.
    db_input : TranscriptionInput
        The PDF to redo.

    Returns
    -------
    list[str]
        The files its output rows pointed to, to remove once the change is committed.
    """
    files = []
    for output in list(db_input.assets):
        files += [path for path in (output.ocr_output_file, output.list_replacement_output_file,
                                    output.pii_scrubber_output_file, output.pii_scrubber_confidence_file) if path]
        session.query(StringReplacementCompletion).filter(StringReplacementCompletion.output_id == output.id) \
            .delete(synchronize_session=False)
        db_input.assets.remove(output)
    subject_ids = sorted({m.subject_id for m in db_input.derived_metadata})
    for metadata in list(db_input.derived_metadata):
        db_input.derived_metadata.remove(metadata)
    for subject_id in subject_ids:
        db_input.derived_metadata.append(TranscriptionMetadata(subject_id=subject_id))
    db_input.ocr_status = None
    db_input.ocr_error = None
    return files


def remove_files(paths: list[str]) -> list[str]:
    """
    Removes a redone PDF's old output files. One that can't be removed is logged.

    Returns
    -------
    list[str]
        The files that weren't there, for the run's summary: already removed, or stored as a path relative to a
        directory other than this one.
    """
    missing = []
    for path in paths:
        try:
            os.remove(path)
        except FileNotFoundError:
            missing.append(path)
        except OSError as e:
            stdout_logger.warning(f'Could not remove the old output {path}: {e}')
    return missing


def log_missing_files(missing: list[str]) -> None:
    """
    Warns about old output files that weren't found, so a run that couldn't remove them doesn't look like one that
    did.
    """
    if not missing:
        return
    relative = [path for path in missing if not os.path.isabs(path)]
    why = (f'{len(relative)} of them are relative paths, looked for from {os.getcwd()}: run this script from the '
           f'directory step 2 was run from to remove them' if relative else 'they were probably removed already')
    stdout_logger.warning(f'{len(missing)} old output file(s) of redone PDFs were not found, so they were not removed; '
                          f'{why}. For example: {missing[0]}')


def redo(args: argparse.Namespace) -> dict[str, int]:
    """
    Checks every PDF transcribed before #82, and redoes or keeps each one.

    Parameters
    ----------
    args : argparse.Namespace
        The parsed args.

    Returns
    -------
    dict[str, int]
        How many PDFs were redone (or would be, with --dry-run), kept, and left to check again because their text
        layer couldn't be read.
    """
    counts = {REDO: 0, KEEP: 0, UNREADABLE: 0}
    missing_files = []
    sessionmaker = get_database_session(echo=args.debug_sql)
    check_text_source_column(sessionmaker)
    last_id = 0
    # One pool for the whole run, rather than one per batch.
    pool_context = ProcessPoolExecutor(max_workers=args.workers) if args.workers > 1 else nullcontext()
    with pool_context as pool, sessionmaker() as session:
        while True:
            # Batches by ID, so PDFs this run has kept or redone, which no longer match, don't shift the batches.
            # With --dry-run nothing changes, which this way doesn't matter either.
            batch = (session.query(TranscriptionInput)
                     .where(TranscriptionInput.id > last_id)
                     .where(is_pdf_column(TranscriptionInput.input_file))
                     .where(TranscriptionInput.document_type == args.document_type)
                     # An empty ocr_status, with outputs, is a PDF transcribed before step 2 recorded it.
                     .where(or_(TranscriptionInput.ocr_status == 'success', TranscriptionInput.ocr_status.is_(None)))
                     .where(TranscriptionInput.assets.any(and_(TranscriptionOutput.text_source.is_(None),
                                                               TranscriptionOutput.ocr_output_file.isnot(None))))
                     .order_by(TranscriptionInput.id)
                     .limit(args.chunk_size)
                     .all())
            if not batch:
                break
            last_id = batch[-1].id
            paths = [db_input.input_file for db_input in batch]
            if pool:
                decisions = list(pool.map(decide, paths, chunksize=4))
            else:
                decisions = [decide(path) for path in paths]
            for db_input, (decision, why) in zip(batch, decisions):
                if decision == UNREADABLE and args.keep_unreadable:
                    decision, why = KEEP, f'kept with --keep-unreadable, since its text layer could not be read: {why}'
                counts[decision] += 1
                if decision == UNREADABLE:
                    # Not marked, so the next run checks it again: poppler may only have timed out this time.
                    stdout_logger.warning(f'Item {db_input.input_file}: left as it is, to check again next run, since '
                                          f'its text layer could not be read: {why}')
                    continue
                if args.dry_run:
                    if decision == REDO:
                        stdout_logger.info(f'Would redo {db_input.input_file} ({why}).')
                    continue
                if decision == REDO:
                    files = reset_input(session, db_input)
                    session.commit()
                    missing_files += remove_files(files)
                    stdout_logger.info(f'Redoing {db_input.input_file} ({why}).')
                else:
                    stdout_logger.info(f'Keeping the OCR of {db_input.input_file} ({why}).')
                    for output in db_input.assets:
                        output.text_source = 'ocr'
                    session.commit()
            session.expunge_all()
            stdout_logger.info(f'Checked {sum(counts.values())} PDFs so far.')
    verb = 'Would redo' if args.dry_run else 'Redid'
    stdout_logger.info(f'All done! {verb} {counts[REDO]} PDFs; {counts[KEEP]} kept their OCR (no page has a usable '
                       f'text layer); {counts[UNREADABLE]} left to check again, since their text layer could not be '
                       f'read.')
    log_missing_files(missing_files)
    if counts[UNREADABLE]:
        stdout_logger.info('Run this again to check those once more. If they still can\'t be read, add '
                           '--keep-unreadable to keep their OCR instead, so the redo can finish.')
    if counts[REDO] and not args.dry_run:
        stdout_logger.info('Now run the pipeline steps again, from transcribe_pdfs.py on, to process them.')
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0].strip())
    parser.add_argument('--document-type', type=str, default='document',
                        help='The document type to redo: document, page or block (default: %(default)s).')
    parser.add_argument('--dry-run', action='store_true', help='Only count and log what would be redone.')
    parser.add_argument('--chunk-size', type=int, default=500, help='PDFs checked per batch (default: %(default)s).')
    parser.add_argument('--workers', type=int, default=os.cpu_count() or 1,
                        help='PDFs whose pages are checked at once (default: the number of CPUs, %(default)s).')
    parser.add_argument('--keep-unreadable', action='store_true',
                        help="Keep the OCR of PDFs whose text layer can't be read this time (poppler took too long or "
                             "was killed), instead of leaving them to check again on the next run.")
    parser.add_argument('--debug-sql', action='store_true', help='Enable SQL debugging')
    return parser.parse_args(normalize_flag_spelling(sys.argv[1:]))


def main() -> int:
    args = parse_args()
    if args.document_type not in ('document', 'page', 'block'):
        stdout_logger.error("'--document-type' must be either 'document', 'page' or 'block'")
        return 1
    # Without them every PDF would look unreadable, so stop rather than check them all for nothing.
    missing = [tool for tool in TEXT_LAYER_TOOLS if shutil.which(tool) is None]
    if missing:
        stdout_logger.error(f'Stopping before redoing anything: poppler is not installed or not on PATH (missing '
                            f'{", ".join(missing)}). Run this in the workspace image.')
        return 1
    try:
        redo(args)
    except MissingColumnError as e:
        stdout_logger.error(f'Stopping before redoing anything: {e}')
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
