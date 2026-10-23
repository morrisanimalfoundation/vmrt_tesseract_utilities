#!/usr/bin/env bash

# Exit on Error.
set -e

# Read our .env file.
export $(grep -v '^#' .env | xargs)

if [[ -z $SQL_PASSWORD ]]; then
  echo "Error: Please set the SQL_PASSWORD variable in your .env file."
  exit 1
fi


# Run the workspace container as the current user (see docker-compose.yml).
export HOST_UID=$(id -u) HOST_GID=$(id -g)

# Get the workspace image: download the prebuilt one, or build it locally with `./run.sh --build`
# (needed when you change the Dockerfile or requirements.txt).
if [[ "$1" == "--build" ]]; then
  docker compose build
elif ! docker compose pull vmrt-emr-workspace; then
  # `docker compose up` below then uses the copy you already have, or builds one if there isn't one.
  echo "Could not download the latest prebuilt image. Using your existing copy, or building it locally if there is none (this takes a few minutes)."
fi

# Start the containers in detached mode.
docker compose up -d

# Wait for the database container to start (with a timeout).
TIMEOUT=30
COUNTER=0
until $(docker exec -i vmrt-emr-process-log-mysql mysql -uroot -p$SQL_PASSWORD -e "DROP DATABASE IF EXISTS vmrt_emr_transcription; CREATE DATABASE vmrt_emr_transcription;") || [[ $COUNTER -eq $TIMEOUT ]]; do
  echo "Waiting for database container to start... ($COUNTER/$TIMEOUT)"
  sleep 1
  COUNTER=$((COUNTER+1))
done

if [[ $COUNTER -eq $TIMEOUT ]]; then
  echo "Error: Timeout waiting for database container."
  exit 1
fi

echo "Database initialized successfully."

# Execute the Python script.
if ! docker exec -t vmrt-emr-workspace python ./scripts/database_setup.py install; then
  echo "Error: Failed to execute Python script."
  exit 1
fi

# Provide an interactive Bash shell within the container.
docker exec -it vmrt-emr-workspace bash

# Stop and remove the containers.
docker compose down