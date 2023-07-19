#!/usr/bin/env bash
set -eo pipefail

# Create mount directory for service
mkdir -p $MNT_DIR

echo "Mounting GCS Fuse."
gcsfuse --implicit-dirs --debug_gcs --debug_fuse $BUCKET $MNT_DIR 
echo "Mounting completed."

# Run the web service on container startup. Here we use the gunicorn
# webserver, with one worker process and 8 threads.
# For environments with multiple CPU cores, increase the number of workers
# to be equal to the cores available.
# Timeout is set to 0 to disable the timeouts of the workers to allow Cloud Run to handle instance scaling.
exec /env/bin/panel serve /app/hls_gcloud_app.ipynb --address 0.0.0.0 --port 8080 --allow-websocket-origin "*"

# Set to -n to exit immediately when one of the background processes terminate.
wait -y