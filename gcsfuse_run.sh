#!/usr/bin/env bash
set -eo pipefail

# Create mount directory for service
mkdir -p $MNT_DIR

echo "Mounting GCS Fuse."
gcsfuse --implicit-dirs --debug_gcs --debug_fuse $BUCKET $MNT_DIR 
echo "Mounting completed."

# Run the web service on container startup.
exec /env/bin/panel serve /app/hls_gcloud_app.ipynb --address 0.0.0.0 --port 8080 --allow-websocket-origin "*" --keep-alive 30000

# Set to -n to exit immediately when one of the background processes terminate.
wait -y
