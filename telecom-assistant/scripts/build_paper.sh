#!/usr/bin/env bash
# Build a paper directory with tectonic after deleting stale aux files.
# Usage: scripts/build_paper.sh <dir-relative-to-telecom-assistant>
# Example: scripts/build_paper.sh paper
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <paper-dir-relative-to-telecom-assistant>" >&2
  echo "example: $0 paper" >&2
  exit 1
fi

rel="$1"
if [[ -z "$rel" || "$rel" == /* ]]; then
  echo "paper directory must be a non-empty path relative to telecom-assistant/, got: ${rel:-<empty>}" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PAPER="$ROOT/$rel"

if [[ ! -d "$PAPER" ]]; then
  echo "paper directory not found: $PAPER" >&2
  exit 1
fi
if [[ ! -f "$PAPER/main.tex" ]]; then
  echo "missing main.tex in $PAPER" >&2
  exit 1
fi
if ! command -v tectonic >/dev/null 2>&1; then
  echo "tectonic not found on PATH" >&2
  exit 1
fi

cd "$PAPER"
rm -f main.bbl main.aux main.blg main.out

log="$(mktemp)"
trap 'rm -f "$log"' EXIT

if ! tectonic main.tex >"$log" 2>&1; then
  cat "$log"
  echo "tectonic failed for $PAPER" >&2
  exit 1
fi
cat "$log"

# Tectonic may also write a conventional .log; search both.
search_files=("$log")
if [[ -f main.log ]]; then
  search_files+=("main.log")
fi
if grep -E -n '\?\?|Citation .* undefined' "${search_files[@]}" >/dev/null 2>&1; then
  echo "WARNING: unresolved references or undefined citations:" >&2
  grep -E -n '\?\?|Citation .* undefined' "${search_files[@]}" >&2 || true
fi
