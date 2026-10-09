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

## Contents

- [How it works](#how-it-works)
- [What you need before you start](#what-you-need-before-you-start)
- [Quick start](#quick-start)
- [Running the pipeline](#running-the-pipeline)
- [Where the results go](#where-the-results-go)
- [Stopping, restarting, and starting over](#stopping-restarting-and-starting-over)
- [Reference data files](#reference-data-files)
- [Troubleshooting](#troubleshooting)
- [Working with git](#working-with-git)
- [Development](#development)

## How it works

The pipeline is five scripts that you run one after another. Each one picks up where the previous one left
off:

| Step | Script | What it does |
| --- | --- | --- |
| 1 | `create_transcription_process.py` | Finds the EMR files and queues them for processing |
| 2 | `transcribe_pdfs.py` | Runs OCR (Tesseract) to turn each PDF into text |
| 3 | `replace_strings.py` | Replaces known identifying values (e.g. subject IDs) with a placeholder |
| 4 | `pii_scrubber.py` | Finds and removes remaining PII using an NLP model (Presidio) |
| 5 | `visit_date_miner.py` | Matches dates found in the text to known vet visit dates |

Progress is tracked in a MySQL database: each source file gets a row in `transcription_input`, and each
step records its result (OCR text, scrubbed text, mined metadata) in `transcription_output` and
`transcription_metadata`. The scripts use those tables to work out which files still need processing.

Everything runs inside Docker containers, so you do not need to install Python, Tesseract, or MySQL on your
own computer.

## What you need before you start

- **A terminal.** macOS and Linux work out of the box. On Windows, install
  [WSL2](https://learn.microsoft.com/windows/wsl/install) and run every command in this README from the WSL
  terminal — `run.sh` is a Bash script and will not run in PowerShell or Command Prompt.
- **Git** — [installation instructions](https://git-scm.com/downloads). Check with `git --version`.
- **Docker Desktop** (or Docker Engine with the Compose plugin) —
  [installation instructions](https://docs.docker.com/get-started/get-docker/). Docker must be **running**
  before you start. Check with `docker compose version`.
- **The raw GRLS EMR PDFs** available as a folder on your computer (a synced Dropbox folder, a mounted
  network share, etc.).
- **The two GRLS reference files**, `dog_profile.tsv` and `vet_visits.tsv` (see
  [Reference data files](#reference-data-files)). These are **not** included in this repository — ask your
  research lead for them.
- Several gigabytes of free disk space and a good internet connection for the first build.

## Quick start

### 1. Download the code

```bash
git clone https://github.com/morrisanimalfoundation/vmrt_tesseract_utilities.git
cd vmrt_tesseract_utilities
```

This creates a `vmrt_tesseract_utilities` folder in whatever folder your terminal is currently in. Every
command in the rest of this section is run from inside that folder.

### 2. Create your `.env` file

The `.env` file holds the database password. Make your own copy of the example:

```bash
cp example.env .env
```

The example values work as they are, so you can leave the file alone. If you do want a different password,
open `.env` in a text editor and change it in **both** lines — it appears once in `SQL_PASSWORD` and once
inside `SQL_CONNECTION_STRING`, and the two must match. Use only letters and numbers.

### 3. Put the reference files in place

Create a `data` folder and copy the two reference files into it:

```bash
mkdir -p data
```

```
vmrt_tesseract_utilities/
└── data/
    ├── dog_profile.tsv
    └── vet_visits.tsv
```

### 4. Tell Docker where your EMR PDFs are

Open `docker-compose.yml` in a text editor and find the last line:

```yaml
      device: "$HOME/MAF\ Dropbox/GRLS/Operations/ENROLLED\ DOGS"
```

Change the path to the folder on **your** computer that holds the EMR PDFs. Two things to know:

- `$HOME` means your home folder (e.g. `/Users/yourname` on macOS), so `$HOME/Documents/emrs` is the `emrs`
  folder inside your Documents.
- Inside this file, every space in a folder name must be written as `\ ` (backslash, then space), as in
  the example above.

The folder must already exist. The scripts look for the dog's subject ID (`094-` followed by six digits,
e.g. `094-000520`) somewhere in each file's folder path; files that are not under a folder named with a
subject ID are skipped.

> This edit is specific to your computer. **Do not commit it** — see [Working with git](#working-with-git).

### 5. Start the environment

```bash
./run.sh
```

The first run downloads and builds everything, which takes a while — expect to wait several minutes at
least. Later runs are much faster. When it finishes you will see a new prompt that looks something like this:

```
jenkins@3f2a9c1d7e4b:/workspace$
```

You are now **inside the container**. `/workspace` is this repository, and `/data` is the EMR folder you
chose in step 4. The database tables have already been created for you.

Check that your EMRs are visible:

```bash
ls /data
```

If that shows your EMR folders, you are ready to run the pipeline.

## Running the pipeline

Run these five commands in order, inside the container, from `/workspace`. Wait for each one to finish
before starting the next. The copy-paste versions below use the default settings.

**1. Register files for processing** — walks `/data`, reads each dog's subject ID from the folder path, and
queues the files. Run this **once** per session; running it again queues every file a second time.

```bash
python scripts/create_transcription_process.py /data
```

**2. Run OCR** — runs Tesseract over the queued PDFs. This is the slow step.

```bash
python scripts/transcribe_pdfs.py /workspace/output
```

**3. Replace known identifying strings** — replaces every value in one column of a TSV/CSV file with a
placeholder. This example replaces each dog's subject ID with `<ID>`:

```bash
python scripts/replace_strings.py data/dog_profile.tsv subject_id "<ID>" /workspace/output
```

The four arguments are, in order: the data file, the column to read values from, the replacement text, and
the output folder. You can run this step several times with different files or columns (for example, a list
of dog names); each run builds on the previous one.

**4. Scrub PII** — uses an NLP model to find and remove names, addresses, phone numbers, and similar. The
first time you run this in a session it downloads the model, which needs an internet connection.

```bash
python scripts/scrubbers/pii_scrubber.py /workspace/output
```

**5. Mine metadata** — compares dates found in the text with each dog's known visit dates to work out which
visit a document belongs to.

```bash
python scripts/metadata_miners/visit_date_miner.py /workspace/output \
  --visit_date_tsv=data/vet_visits.tsv \
  --dog_profile_tsv=data/dog_profile.tsv
```

### Options

Add `--help` to any script to see everything it accepts, e.g. `python scripts/transcribe_pdfs.py --help`.

Flag spelling is not consistent between scripts — some use hyphens and some use underscores. Use the
spelling in this table:

| Script | Document type | Batch size | Skip records | Show SQL | Other |
| --- | --- | --- | --- | --- | --- |
| `create_transcription_process.py` | `--document-type` | — | — | `--debug-sql` | |
| `transcribe_pdfs.py` | `--document-type` | `--chunk-size` | `--offset` | `--debug-sql` | |
| `replace_strings.py` | `--document_type` | `--chunk_size` | `--offset` | `--debug-sql` | `--max-workers`, `--no-multiprocessing` |
| `pii_scrubber.py` | `--document-type` | `--chunk-size` | `--offset` | `--debug-sql` | `--threshold`, `--config` |
| `visit_date_miner.py` | — | `--chunk_size` | `--offset` | `--debug_sql` | `--visit_date_threshold`, `--search_unstructured_text_dir` |

- **Document type** (`document`, `page`, or `block`; default `document`) controls how finely the OCR output
  is split: one text file per PDF, one per page, or one per detected line of text. Whatever you choose in
  step 1 must be passed to steps 2, 3, and 4 as well, or those steps will find nothing to do.
- **Batch size** defaults to 1000 files per run in steps 2, 3, and 4. If you have more than 1000 files,
  re-run steps 2 and 4 until they stop finding files (step 2 prints `Found 0 files to transcribe`). Step 3 does not skip files it has already
  handled, so use `--offset` there instead (`--offset 1000`, then `--offset 2000`, and so on).
- `--threshold` (step 4) is the minimum confidence score, from 0 to 1, a detection needs before it is
  removed. The default `0.0` removes everything the model flags.
- `--visit_date_threshold` (step 5) is how many days apart a date in the text and a known visit date can be
  and still count as a match. The default is 3.
- `--search_unstructured_text_dir` (step 5) also searches the raw OCR text for dates, not only the PII
  scrubber's report.

`scripts/scrubbers/scrub_txt_files.sh /workspace/output` is an alternative way to run step 4. It installs
any spaCy models listed in `scripts/scrubbers/config/` first, then runs `pii_scrubber.py` with
`--threshold=0.45`.

## Where the results go

Output files are written to the `output` folder in this repository, which you can open from your own
computer as well as from inside the container:

| Path | Contents |
| --- | --- |
| `output/unstructured_text/<document-type>/` | Raw OCR text (step 2) |
| `output/list_replacement_output_file/<document-type>/` | Text after string replacement (step 3) |
| `output/scrubbed_text/<document-type>/scrubbed_<document-type>/` | Text after PII scrubbing (step 4) |
| `output/scrubbed_text/<document-type>/scrubbed_confidence/` | One JSON report per file listing what was detected and with what confidence (step 4) |

OCR confidence scores and the matched visit dates (step 5) are stored in the database, not in files. To
look at them, open a MySQL prompt from a second terminal on your own computer, using the password from your
`.env`:

```bash
docker exec -it vmrt-emr-process-log-mysql mysql -uroot -p vmrt_emr_transcription
```

```sql
SELECT ocr_output_file, ocr_confidence FROM transcription_output LIMIT 10;
SELECT subject_id, visit_date, extracted_date FROM transcription_metadata LIMIT 10;
```

Type `exit` to leave the MySQL prompt.

The `output` folder is ignored by git. It contains text from medical records — **never** commit it, email
it, or copy it anywhere it is not supposed to be.

## Stopping, restarting, and starting over

- **To stop**, type `exit` at the container prompt. This shuts down and removes the containers.
- **The database is temporary.** It is erased when you exit, and `./run.sh` creates a fresh empty one each
  time it starts. The files in `output/` are kept, but the record of which files were processed is not — so
  after restarting you begin again from step 1, and step 2 will redo the OCR. Plan to finish a batch in one
  session, and leave the container shell open while long steps run.
- **To open a second terminal** in the running container (to look at files while a step runs, say), run
  this from a new terminal window on your own computer. Do not run `./run.sh` a second time — that would
  erase the database you are using.
  ```bash
  docker exec -it vmrt-emr-workspace bash
  ```
- **To start completely clean**, exit the container, delete the `output` folder, and run `./run.sh` again.

## Reference data files

Two different things are called "data" in this project, and it is easy to mix them up:

- `/data` (inside the container) — the raw EMR **PDFs**, from the folder you chose in
  [Quick start step 4](#4-tell-docker-where-your-emr-pdfs-are). Used as the input to
  `create_transcription_process.py`.
- `data/dog_profile.tsv` and `data/vet_visits.tsv` (in this repository's `data` folder, which is
  `/workspace/data` in the container) — tabular GRLS exports used by `replace_strings.py` and
  `visit_date_miner.py`. The `data` folder is ignored by git and **not included in this repository**.

Both files must be tab-separated with a header row. Expected columns:

- `dog_profile.tsv`: `subject_id`, `sex_status`, `birth_date`, `enrolled_date`, `study_status`,
  `dog_was_euthanized`, `death_date`, `spay_neuter_date`. The pipeline itself reads `subject_id`,
  `birth_date`, and `death_date`.
- `vet_visits.tsv`: `visit_date`, plus the dog's ID in a column named `grls_id` (or `subject_id`), written
  in the same `094-######` form used in the EMR folder names.

## Troubleshooting

**`./run.sh: Permission denied`** — run `bash run.sh` instead.

**`grep: .env: No such file or directory`** or **`Please set the SQL_PASSWORD variable`** — you have not
created `.env` yet, or `SQL_PASSWORD` is empty. See
[Quick start step 2](#2-create-your-env-file).

**`Cannot connect to the Docker daemon`** — Docker is not running. Start Docker Desktop and try again.

**`Waiting for database container to start...`** — a handful of these lines (with a MySQL error above each)
is normal while the database boots. If it reaches `Timeout waiting for database container`, run
`docker compose down` and then `./run.sh` again.

**An error mentioning `emr-source` and `no such file or directory`** — the path in `docker-compose.yml`
does not exist. Check the spelling, the `\ ` before each space, and that the folder is really there (for
Dropbox, that it has finished syncing).

**`ls /data` is empty or shows the wrong folder after you changed the path** — Docker remembers the old
location. Exit the container, then run:

```bash
docker compose down
docker volume rm vmrt_tesseract_utilities_emr-source
./run.sh
```

**`Access denied for user 'root'`** — the password in `SQL_CONNECTION_STRING` does not match
`SQL_PASSWORD` in your `.env`.

**Step 1 finishes instantly and adds nothing** — no folder under `/data` has a subject ID
(`094-` and six digits) in its name. Check `ls /data`.

**A later step finds 0 files or finishes without doing anything** — the step before it has not been run, the database was
reset by a restart (go back to step 1), or you used a different document type from the one in step 1.

**`python: command not found` or `No module named ...`** — you are running the command on your own
computer rather than inside the container. The prompt should end in `/workspace$`.

## Working with git

If you are new to git, these are the only commands you need to use this project:

- **Get the latest version** of the code (run from the repository folder, outside the container):
  ```bash
  git pull
  ```
  If `./run.sh` behaves differently after a pull, that is expected — it rebuilds whatever changed.
- **See what you have changed:**
  ```bash
  git status
  ```
  `docker-compose.yml` will show as modified because of the path you set in the Quick start. That is fine;
  leave it that way. `.env`, `data/`, and `output/` will not appear at all, because git is told to ignore
  them.
- **If `git pull` refuses** because of your change to `docker-compose.yml`, set it aside, pull, and put it
  back:
  ```bash
  git stash
  git pull
  git stash pop
  ```

Things that must never be committed or pushed: your `.env` file, anything in `data/` or `output/`, any EMR
or text taken from one, and your personal path in `docker-compose.yml`. If you plan to contribute code,
make your changes on a new branch (`git switch -c my-change`) and add files by name (`git add path/to/file`)
rather than with `git add .`, so nothing extra is swept in.

## Development

- **Tests**: `pytest` (configured via `pytest.ini`). The test suite mocks the database and does not require
  Tesseract, so it can be run on your own computer without Docker:
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  pip install -r requirements/pytest_requirements.txt
  pytest
  ```
  Run a single test with `pytest tests/pytests/string_replacer_test.py::test_replace_single_string`.
- **Linting**: `./sort_and_lint.sh` runs `isort` and `flake8` (line length is not enforced). Install
  dependencies first with `pip install -r requirements/ci_requirements.txt`.
- Both checks run automatically on every pull request to `main`.
