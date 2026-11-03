#!/bin/sh
# Start the API and the website; if either one stops, stop the container so the Space restarts it.
set -eu

# Real models: fetch the weights first, each one checked against its pinned SHA-256. The detector and
# the road model are public downloads; the classifier comes from the private repo CLASSIFIER_REPO
# (token HF_TOKEN). Any failure stops the container, so a Space never runs half-loaded.
if [ "${USE_STUBS:-1}" = "0" ]; then
  python -m tools.fetch_detector --model "${DETECTOR_MODEL:-fasterrcnn_mobilenet_v3_large_fpn}"
  python -m tools.fetch_road_model
  python -m tools.fetch_classifier --dest /app/classifier
  export DEFECT_CHECKPOINT=/app/classifier/best.pt
fi

uvicorn api.main:app --host 127.0.0.1 --port 8000 &
api=$!

cd /app/web
node node_modules/next/dist/bin/next start --port "${PORT:-7860}" --hostname 0.0.0.0 &
web=$!

trap 'kill "$api" "$web" 2>/dev/null || true' TERM INT
# dash has no `wait -n`: poll until one of the two is gone.
while kill -0 "$api" 2>/dev/null && kill -0 "$web" 2>/dev/null; do
  sleep 2
done
echo "start.sh: the API or the website stopped; shutting down" >&2
kill "$api" "$web" 2>/dev/null || true
exit 1
