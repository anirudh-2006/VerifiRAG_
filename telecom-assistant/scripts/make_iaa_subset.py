"""
Build a BLINDED inter-annotator-agreement (IAA) subset for a second annotator.

We select whole queries (so derivation context is preserved) greedily,
prioritising queries that contain human-flagged errors, until every error step
is covered AND at least --min-steps steps are included. For each selected query
we emit a copy with the first annotator's judgements stripped:

  - human_correct        -> null
  - error_type           -> null
  - notes                -> ""          (so annotator 2 isn't primed)
  - final_answer_correct -> null
  - _filled              -> false

`expression_given`, `description`, `verifier_*`, `expected_final`, and
`generated_final` are preserved so annotator 2 sees exactly what annotator 1 saw
(the derivation and the question), minus the labels.

A `_manifest.json` records which (qid, step_number) pairs form the IAA overlap
and the stratification counts, so compute_iaa.py can score exactly this subset.

Usage
-----
  cd telecom-assistant
  PYTHONPATH=. ./.venv/bin/python scripts/make_iaa_subset.py \
      --min-steps 50 --seed 13 --out-dir results/iaa_subset

Then hand results/iaa_subset/ to annotator 2. They set per-step human_correct
(true/false) + error_type, and _filled:true per query. Do NOT let them see
results/label_templates/.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from typing import Any, Dict, List

ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.label_scorer import load_labels  # noqa: E402

DEFAULT_LABEL_DIR = "results/label_templates"
DEFAULT_OUT_DIR = "results/iaa_subset"

# Fields wiped so the second annotator labels blind.
_STRIP_STEP_LABEL_KEYS = ("human_correct", "error_type")


def _query_stats(q: Dict[str, Any]) -> Dict[str, int]:
    steps = q.get("steps", []) or []
    n_steps = len(steps)
    n_err = sum(1 for s in steps if s.get("human_correct") is False)
    return {"n_steps": n_steps, "n_err": n_err}


def _blank_query(q: Dict[str, Any]) -> Dict[str, Any]:
    """Return a deep-ish copy of a query with annotator-1 labels removed."""
    new_q = dict(q)
    new_q["_filled"] = False
    new_q["final_answer_correct"] = None
    new_q["_iaa_blinded"] = True
    new_q["_instructions"] = (
        "IAA re-labelling: for each step set 'human_correct' to true/false and "
        "'error_type' to one of {algebra, sign, missing_term, substitution, "
        "definition, none}. Apply the notation-strictness rule: mark a step "
        "false if a boundary term / IBP notation is written wrong even if the "
        "final math works out. Set '_filled' to true when done. Do NOT consult "
        "any other label files."
    )
    new_steps: List[Dict[str, Any]] = []
    for s in q.get("steps", []) or []:
        ns = dict(s)
        for k in _STRIP_STEP_LABEL_KEYS:
            if k in ns:
                ns[k] = None
        # Wipe annotator-1 free-text so annotator 2 isn't primed.
        if "notes" in ns:
            ns["notes"] = ""
        new_steps.append(ns)
    new_q["steps"] = new_steps
    return new_q


def select_queries(
    labeled: List[Dict[str, Any]],
    *,
    min_steps: int,
    seed: int,
) -> List[Dict[str, Any]]:
    """Greedily pick whole queries: all error-bearing queries first (shuffled),
    then error-free queries (shuffled) until >= min_steps steps are covered."""
    rng = random.Random(seed)
    stats = {q["id"]: _query_stats(q) for q in labeled}

    err_qs = [q for q in labeled if stats[q["id"]]["n_err"] > 0]
    clean_qs = [q for q in labeled if stats[q["id"]]["n_err"] == 0]
    rng.shuffle(err_qs)
    rng.shuffle(clean_qs)

    # Always take every error-bearing query so all error steps are in the overlap.
    selected: List[Dict[str, Any]] = list(err_qs)
    covered = sum(stats[q["id"]]["n_steps"] for q in selected)

    for q in clean_qs:
        if covered >= min_steps:
            break
        selected.append(q)
        covered += stats[q["id"]]["n_steps"]

    # Stable, human-friendly order in the output dir.
    selected.sort(key=lambda q: q.get("id", ""))
    return selected


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a blinded IAA re-label subset.")
    ap.add_argument("--label-dir", default=DEFAULT_LABEL_DIR,
                    help="Annotator-1 filled label templates.")
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR,
                    help="Where to write blinded templates for annotator 2.")
    ap.add_argument("--min-steps", type=int, default=50,
                    help="Minimum total steps to include (whole queries only).")
    ap.add_argument("--seed", type=int, default=13,
                    help="RNG seed for reproducible query sampling.")
    ap.add_argument("--overwrite", action="store_true",
                    help="Allow writing into a non-empty --out-dir.")
    args = ap.parse_args()

    labeled = load_labels(args.label_dir)
    if not labeled:
        print(f"No filled label files found in {args.label_dir}.")
        return

    total_steps = sum(len(q.get("steps", [])) for q in labeled)
    total_err = sum(
        1 for q in labeled for s in q.get("steps", []) if s.get("human_correct") is False
    )
    print(f"Loaded {len(labeled)} queries, {total_steps} steps, "
          f"{total_err} human-flagged error steps.")

    selected = select_queries(labeled, min_steps=args.min_steps, seed=args.seed)

    sel_steps = sum(len(q.get("steps", [])) for q in selected)
    sel_err = sum(
        1 for q in selected for s in q.get("steps", []) if s.get("human_correct") is False
    )
    print(f"Selected {len(selected)} queries -> {sel_steps} steps "
          f"({sel_err}/{total_err} error steps covered).")
    if sel_err < total_err:
        print("  WARNING: not all error steps covered (some error queries missing?).")

    if os.path.isdir(args.out_dir) and os.listdir(args.out_dir) and not args.overwrite:
        print(f"Refusing to write into non-empty {args.out_dir}; pass --overwrite.")
        return
    os.makedirs(args.out_dir, exist_ok=True)

    overlap: List[Dict[str, Any]] = []
    for q in selected:
        blanked = _blank_query(q)
        path = os.path.join(args.out_dir, f"{q['id']}.json")
        with open(path, "w") as f:
            json.dump(blanked, f, indent=2)
        for s in q.get("steps", []):
            overlap.append({"qid": q["id"], "step_number": s.get("step_number")})

    manifest = {
        "source_label_dir": args.label_dir,
        "seed": args.seed,
        "min_steps": args.min_steps,
        "selected_queries": [q["id"] for q in selected],
        "n_queries": len(selected),
        "n_steps": sel_steps,
        "n_error_steps_covered": sel_err,
        "n_error_steps_total": total_err,
        "overlap_steps": overlap,
    }
    with open(os.path.join(args.out_dir, "_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nWrote {len(selected)} blinded templates + _manifest.json to {args.out_dir}/")
    print("Hand this directory to annotator 2. They must NOT see "
          f"{args.label_dir}/.")


if __name__ == "__main__":
    main()
