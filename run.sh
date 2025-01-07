#!/usr/bin/env bash

# Exit on Error.
set -e

# Read our .env file.
export $(grep -v '^#' .env | xargs)

if [[ -z $SQL_PASSWORD ]]; then
  echo "Error: Please set the SQL_PASSWORD variable in your .env file."
  exit 1
fi


# Build the Docker images with the current user's ID.
docker compose build --build-arg USER_ID=$(id -u ${USER})

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