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
/bin/bash "${SCRIPT_DIR}/install_spacy_models.sh"

python "${SCRIPT_DIR}/pii_scrubber.py" \
      /workspace/output/filemap*.json \
      "$OUTPUT_DIR" \
      --config="${SCRIPT_DIR}/config/stanford-deidentifier-base_nlp.yaml" \
      --threshold=0.45

echo "All files processed."