#!/bin/bash

# This script processes all files defined in the filemap_confidence JSON files,
# running the `pii-scrubber.py` script on each file to perform PII scrubbing.

# Get the input directory from the command line argument
OUTPUT_DIR="$1"

# Get the directory of the script
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Check if the output directory is provided
if [ -z "$OUTPUT_DIR" ]; then
  echo "Error: Please provide the output directory as the first argument."
  exit 1
fi

# Install the spacy models.
chmod +x "${SCRIPT_DIR}/install_spacy_models.sh"
if ! /bin/bash "${SCRIPT_DIR}/install_spacy_models.sh"; then
  echo "Error: Installing the spaCy models failed, so no files were scrubbed."
  exit 1
fi

# No --threshold, so every detection is removed (the default, 0.0). Some identifiers, such as ZIP codes and
# house numbers, score around 0.4, so a higher threshold leaves them in the text.
# --until-done keeps going until every file is scrubbed, rather than stopping after one chunk of 1000.
python "${SCRIPT_DIR}/pii_scrubber.py" \
  "$OUTPUT_DIR" \
  --config="${SCRIPT_DIR}/config/stanford-deidentifier-base_nlp.yaml" \
  --until-done
# Pass the scrubber's exit code on, so CI can tell 3 (too many files failed) apart from 1 (it couldn't run).
SCRUB_EXIT_CODE=$?
if [ "$SCRUB_EXIT_CODE" -eq 3 ]; then
  echo "Error: Every file, or more than --max-failure-rate of them, failed PII scrubbing. See the list of failed files above."
  exit 3
elif [ "$SCRUB_EXIT_CODE" -ne 0 ]; then
  echo "Error: PII scrubbing failed, so some files were not processed."
  exit "$SCRUB_EXIT_CODE"
fi

echo "All files processed."
