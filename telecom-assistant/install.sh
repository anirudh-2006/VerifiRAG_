#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"
if ! command -v "$PYTHON" &>/dev/null; then
  echo "Need python3 on PATH (or set PYTHON=...)." >&2
  exit 1
fi

"$PYTHON" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

echo
echo "Done. Activate the venv with:"
echo "  source .venv/bin/activate"
echo "Then: python -m src.ingest   and   streamlit run main.py"
