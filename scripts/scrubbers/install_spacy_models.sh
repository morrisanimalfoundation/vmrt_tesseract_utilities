#!/bin/bash

# This script installs all of the spaCy models defined in a directory of YAML configuration files.
#
# The YAML files should have the following structure:
#
# nlp_engine_name: transformers
# models:
#   -
#     lang_code: en
#     model_name:
#       spacy: en_core_web_sm
#       transformers: /opt/models/stanford-deidentifier-base
#
# The script will extract the `spacy` value from the `model_name` section of each YAML file
# and install the corresponding spaCy model using `python -m spacy download`.
#
# Prerequisites:
# * PyYAML, which is installed with the image's Python packages
#
# Usage:
# 1. Save the script as a bash script (e.g., install_spacy_models.sh).
# 2. Make the script executable: chmod +x install_spacy_models.sh
# 3. Update the CONFIG_DIR variable to the path of your YAML files directory.
# 4. Run the script: ./install_spacy_models.sh

# Set the directory containing the YAML configuration files
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
CONFIG_DIR="$SCRIPT_DIR/config"

# Loop through each YAML file in the directory
for config_file in "$CONFIG_DIR"/*.yaml; do
  # Extract the spaCy model name from the YAML file
  spacy_model=$(python -c 'import sys, yaml; print(" ".join(m["model_name"]["spacy"] for m in yaml.safe_load(open(sys.argv[1]))["models"]))' "$config_file")

  # Check if the spaCy model is already installed. This looks only at the installed packages, unlike
  # `spacy validate`, which needs an internet connection, so scrubbing works offline with the image's models.
  if ! python -c 'import sys, spacy.util; sys.exit(0 if spacy.util.is_package(sys.argv[1]) else 1)' "$spacy_model"; then
    echo "Installing spaCy model: ${spacy_model}"
    python -m spacy download $spacy_model
  else
    echo "spaCy model '$spacy_model' is already installed."
  fi
done

echo "All spaCy models installed."
