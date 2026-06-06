#!/usr/bin/env bash
# start.sh -- Comic Reader launcher (macOS / Linux)
# Usage: bash start.sh [--no-stable] [--no-magenta]

set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
ENV_FILE="$ROOT/webgenta/.env"

# Load environment variables from webgenta/.env
if [ -f "$ENV_FILE" ]; then
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
    echo "Loaded environment from webgenta/.env"
else
    echo "WARNING: webgenta/.env not found -- API keys will not be set"
fi

# Open browser after 5s (background)
(sleep 5 && open "http://localhost:8766/" 2>/dev/null || xdg-open "http://localhost:8766/" 2>/dev/null || true) &

echo ""
echo "Starting comic server on http://localhost:8766/"
echo "Press Ctrl+C to stop."
echo ""

cd "$ROOT/frontend"
python model_server.py "$@"
