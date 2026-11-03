#!/bin/sh
# Start the API and the website; if either one stops, stop the container so the Space restarts it.
set -eu

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
