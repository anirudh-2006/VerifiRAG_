#!/usr/bin/env bash
# Run every scripts/test_*.py with the project venv and PYTHONPATH=..
# Prints one PASS/FAIL line per script; exits 1 if any fail.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON="$ROOT/.venv/bin/python3"

if [[ ! -x "$PYTHON" ]]; then
  echo "missing venv interpreter: $PYTHON" >&2
  exit 1
fi

cd "$ROOT"
export PYTHONPATH=.

shopt -s nullglob
tests=("$SCRIPT_DIR"/test_*.py)
if [[ ${#tests[@]} -eq 0 ]]; then
  echo "no test_*.py scripts found in $SCRIPT_DIR" >&2
  exit 1
fi

IFS=$'\n' tests=($(printf '%s\n' "${tests[@]}" | sort))
unset IFS

fail=0
for t in "${tests[@]}"; do
  name="$(basename "$t")"
  if "$PYTHON" "$t"; then
    echo "PASS $name"
  else
    echo "FAIL $name"
    fail=1
  fi
done

exit "$fail"
