# Veterinary Medical Record Transcriber (VMRT) Tesseract Utilities

The Golden Retriever lifetime study has thousands of electronic medical records (EMRs) that have valuable information. The VMRT project is an attempt to automate data extraction from these EMRs. This repository contains some very simple and crude Tesseract scripts to help evaluate our dataset. The unstructured text extracted from the EMRs may or may not be valuable, but understanding the quantity of low confidence records is very useful.

Goals:

- Build dataset to understand composition of EMRs
- Homogenize format of PDF and text files (more to come)
- Determine confidence scores for optical character recognition from PDFs
- Automatically scrub personally (or dog) identifiable information (PII)
- Perform plain text substitution on corpus
- Extract metadata, such as subject id, study year, related visit

# Running the scripts

The scripts are easily run via the Dockerfile included in this repo.

1. Copy the example.env file to .env and fill in the values.  
    a. The value for `SQL_CONNECTION_STRING` should be the connection string for the database container. (i.e. `mysql://user:password@vmrt-emr-process-log-mysql:3306/vmrt_emr_transcription`)
2. The easiest way to spin up the docker images is to run the `run.sh` script within the repository root directory.
3. Set up your DB by running `python /workspace/scripts/database_setup.py install` within the container.
4. Get ready for the transcription process by running `python scripts/create_transcription_process.py /data`
5. Use the `transcribe_pdfs.py` script to transcribe the files needed.
    - `python /workspace/scripts/transcribe_pdfs.py /workspace/output`
6. Use the `pii_scrubber.py` script to remove PII from the text.
    - `python /workspace/scripts/scrubbers/pii_scrubber.py /workspace/output`
7. Use the scripts in the `scripts/metadata_miners` directory to find data in the text.
    - `python /workspace/scripts/metadata_miners/visit_date_miner.py /workspace/output --visit_date_tsv=/path/to/vet_visits.tsv --dog_profile_tsv=/path/to/dog_profile.tsv`