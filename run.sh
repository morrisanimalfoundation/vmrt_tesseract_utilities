#!/usr/bin/env bash

docker compose build --build-arg USER_ID=$(id -u ${USER})

docker compose up -d

until $(docker exec -i vmrt-emr-process-log-mysql mysql -uroot -pbmorris -e "DROP DATABASE IF EXISTS vmrt_emr_transcription; CREATE DATABASE vmrt_emr_transcription;"); do
  echo 'Waiting for database container to start...'
  sleep 1
done

echo 'Done!'

docker exec -t vmrt-emr-workspace python /workspace/update.py

docker exec -it vmrt-emr-workspace bash

docker compose down