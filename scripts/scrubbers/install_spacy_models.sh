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
#       transformers: StanfordAIMI/stanford-deidentifier-base
#
# The script will extract the `spacy` value from the `model_name` section of each YAML file
# and install the corresponding spaCy model using `python -m spacy download`.
#
# Prerequisites:
# * yq (command-line YAML processor)
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
  spacy_model=$(yq '.models[].model_name.spacy' "$config_file" | tr -d '"')

  # Check if the spaCy model is already installed
  if ! python -m spacy validate | grep -q $spacy_model; then
    echo "Installing spaCy model: ${spacy_model}"
    python -m spacy download $spacy_model
  else
    echo "spaCy model '$spacy_model' is already installed."
  fi
done

echo "All spaCy models installed."
