# Veterinary Medical Record Transcriber (VMRT) Tesseract Utilities

The Golden Retriever Lifetime Study (GRLS) has thousands of electronic medical records (EMRs) that contain
valuable information. The VMRT project automates data extraction from these EMRs. This repository contains
Tesseract-based scripts to help evaluate the dataset. The unstructured text extracted from the EMRs may or
may not be valuable, but understanding the quantity of low-confidence records is very useful.

Goals:

- Build a dataset to understand the composition of EMRs
- Homogenize the format of PDF and text files (more to come)
- Determine confidence scores for optical character recognition from PDFs
- Automatically scrub personally (or dog) identifiable information (PII)
- Perform plain text substitution on the corpus
- Extract metadata, such as subject id, study year, related visit

Every stage of the pipeline is tracked in a MySQL database: each source file gets a row in
`transcription_input`, and each stage records its output (OCR text, scrubbed text, mined metadata) back onto
that row. This means every script can be re-run safely — it only picks up files that haven't reached that
stage yet.

## Prerequisites

- Docker and Docker Compose. All scripts are meant to run inside the provided container — Tesseract,
  `tesserocr`, and `poppler-utils` are only installed there, not on the host.
- Access to the raw GRLS EMR PDFs (see **Point Docker at your EMR source** below).
- The tabular GRLS exports used for PII substitution and visit-date matching (see **Reference data files**
  below) — these are **not** included in this repository.

## Setup

### 1. Configure your `.env` file

Copy `example.env` to `.env` and fill in the values:

```bash
cp example.env .env
```

- `SQL_CONNECTION_STRING` — the connection string for the database container, e.g.
  `mysql+pymysql://root:<SQL_PASSWORD>@vmrt-emr-process-log-mysql:3306/vmrt_emr_transcription`
- `SQL_PASSWORD` — the MySQL root password. `run.sh` also uses this to initialize the database container.

`.env` is gitignored — never commit it.

### 2. Point Docker at your EMR source

`docker-compose.yml` mounts a local directory into the workspace container at `/data`:

```yaml
volumes:
  emr-source:
    driver: local
    driver_opts:
      o: bind
      type: none
      device: "$HOME/MAF\ Dropbox/GRLS/Operations/ENROLLED\ DOGS"
```

Before running `run.sh`, update the `device` path to point at wherever you have the GRLS EMR PDFs available
locally (a synced Dropbox folder, a mounted network share, etc.). This path is machine-specific and will not
match your setup by default.

### 3. Start the environment

From the repository root:

```bash
./run.sh
```

This builds the Docker images, starts the containers, waits for the MySQL container to come up, and drops
you into an interactive shell inside the `vmrt-emr-workspace` container at `/workspace` (the repository root,
bind-mounted from your host).

### 4. Create the database schema

Inside the container:

```bash
python scripts/database_setup.py install
```

(`python scripts/database_setup.py drop` removes all tables, if you need to start over.)

## Running the pipeline

The pipeline is a sequence of scripts, each of which queries the database for files that haven't reached
that stage yet, processes them, and writes the result back to the database plus an output file under
`output/`. Run these in order; all commands below are run inside the container, from `/workspace`.

Most stages accept:
- `--document-type {document,page,block}` (default `document`) — controls OCR granularity: an entire
  document, one result per page, or one result per detected text line ("block"). Whichever value you use for
  transcription must be reused in every later stage for that batch of files.
- `--chunk-size` / `--offset` — process files in batches.
- `--debug-sql` — echo SQL statements for debugging.

**Note:** flag spelling is not fully consistent between scripts today — some use `--document-type` /
`--chunk-size` (hyphens) and others use `--document_type` / `--chunk_size` (underscores). The exact flags are
called out per command below.

1. **Register files for processing** — walks a directory of EMRs, extracts each dog's subject ID from the
   path (pattern `094-######`), and queues the files:
   ```bash
   python scripts/create_transcription_process.py /data
   ```

2. **Run OCR** — runs Tesseract over the queued PDFs and writes text output under
   `output/unstructured_text/<document-type>/`:
   ```bash
   python scripts/transcribe_pdfs.py /workspace/output
   ```

3. **Replace known identifying strings** — substitutes known values (e.g. subject IDs) from a TSV/CSV column
   with a placeholder, across the OCR output:
   ```bash
   python scripts/replace_strings.py data/dog_profile.tsv subject_id "<ID>" /workspace/output
   ```
   Positional args are `data_file key_column replacement_string output_dir`. Useful flags: `--document_type`,
   `--chunk_size`, `--offset` (underscores), plus `--no-multiprocessing` and `--max-workers` for controlling
   parallelism.

4. **Scrub PII** — runs Presidio (NLP-based) PII detection and anonymization, writing scrubbed text and a
   JSON confidence report per file under `output/scrubbed_text/<document-type>/`:
   ```bash
   python scripts/scrubbers/pii_scrubber.py /workspace/output
   ```
   `--threshold` sets the minimum confidence score to act on (default `0.0`, i.e. redact everything Presidio
   flags). The spaCy model the default config needs (`en_core_web_sm`) is already installed in the Docker
   image. If you switch `--config` to a NLP config that needs a different spaCy model, run
   `scripts/scrubbers/scrub_txt_files.sh /workspace/output` instead — it installs any missing spaCy models
   first, then runs `pii_scrubber.py` for you.

5. **Mine metadata** — cross-references dates found in the text against known visit/birth/death dates to
   identify which visit each document belongs to:
   ```bash
   python scripts/metadata_miners/visit_date_miner.py /workspace/output \
     --visit_date_tsv=data/vet_visits.tsv \
     --dog_profile_tsv=data/dog_profile.tsv
   ```
   Note this script uses underscores throughout (`--visit_date_tsv`, `--dog_profile_tsv`, `--debug_sql`,
   etc.), unlike most of the other scripts.

### Reference data files

Two different things are called "data" in this project, and it's easy to mix them up:

- `/data` (inside the container) — the raw EMR **PDFs**, mounted from wherever you pointed the `emr-source`
  volume in step 2 above. Used as the input to `create_transcription_process.py`.
- `data/dog_profile.tsv` and `data/vet_visits.tsv` (repo-relative, under `/workspace/data` in the container)
  — tabular GRLS exports used by `replace_strings.py` and `visit_date_miner.py`. The `data/` directory is
  gitignored and **not included in this repository** — ask your research lead for the current exports and
  place them there yourself. Expected columns:
  - `dog_profile.tsv`: `subject_id`, `sex_status`, `birth_date`, `enrolled_date`, `study_status`,
    `dog_was_euthanized`, `death_date`, `spay_neuter_date`
  - `vet_visits.tsv`: includes `grls_id` and `visit_date` at minimum (see `get_dates_from_tsv` in
    `scripts/metadata_miners/visit_date_miner.py` for how these are looked up)

## Development

- **Tests**: `pytest` (configured via `pytest.ini`). The test suite mocks the database and does not require
  Tesseract, so it can be run on the host without Docker:
  ```bash
  pip install -r requirements/pytest_requirements.txt
  pytest
  ```
  Run a single test with `pytest tests/pytests/string_replacer_test.py::test_replace_single_string`.
- **Linting**: `./sort_and_lint.sh` runs `isort` and `flake8` (line length is not enforced). Install
  dependencies first with `pip install -r requirements/ci_requirements.txt`.
