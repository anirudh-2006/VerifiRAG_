#!/usr/bin/env python3
"""Independently verify the published Table 1 and Table 3 numbers.

Reconstructs every cell of Table 1 (three-way grader comparison) and the
Table 3 cascade rows from the *raw* frozen verdicts in ``results/``, then
asserts they match the values recorded in the root CLAUDE.md.

Design constraints (from the root CLAUDE.md):

* Read-only. Opens files in ``results/`` for reading; writes nothing there
  and makes no LLM API calls. Every input is a frozen artifact.
* The metric definitions are implemented once, here, from the definitions
  given in CLAUDE.md. ``src.error_injection.compute_metrics`` and
  ``scripts/build_threeway_table.py`` are deliberately NOT imported -- an
  independent third implementation is the whole point.
* The summary artifacts (``threeway_table.json``, ``cascade_table.json``)
  are also not read as inputs. Reconstruction starts from per-step and
  per-trial verdicts so that a bug in a builder script cannot hide itself.
* The ``metrics`` block stored inside each error-injection report IS read,
  but only as a value to check: ``check_embedded_metrics`` compares it
  against a reconstruction from that same file's raw ``trials``. This is
  what enforces CLAUDE.md's "the two implementations must agree" rule for
  numbers that ship in the OSF release without appearing in any table.

Governing definitions (CLAUDE.md, "Metrics: two independent implementations"):

    strict recall = TP / (TP + FN + AbsE)     abstain-on-error is a miss
    covered FPR   = FP / (FP + TN)            committed negatives only
    a judge verdict of "error" (crash/unparseable) counts as an abstain,
    not as an excluded trial

and, following the confusion-cell convention of Table 1:

    coverage      = (TP + FP + TN + FN) / N
    precision     = TP / (TP + FP)            None when nothing was failed
    abstain rate  = (AbsE + AbsN) / N

Usage:

    cd telecom-assistant && PYTHONPATH=. ./.venv/bin/python3 \
        scripts/verify_published_tables.py

Exits 0 if every field matches, 1 on any mismatch or reconstruction problem.
"""

from __future__ import annotations

import glob
import json
import os
import re
import sys
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)  # telecom-assistant/
REPO = os.path.dirname(PROJECT)
RESULTS = os.path.join(PROJECT, "results")

# The root instructions file. CLAUDE.md is the documented name; CLAUDE_new.md
# is accepted as a fallback because the working tree currently carries the
# successor under that name while CLAUDE.md is deleted.
ROOT_DOC_CANDIDATES = ["CLAUDE.md", "CLAUDE_new.md"]

# Judge verdict keys inside the injection trial records.
GPTOSS_KEY = "llm_judge:gpt-oss-120b"
CLAUDE_KEY = "llm_judge:claude-sonnet-4-5"

# Verdict vocabulary. Anything not a commit is an abstain, which is how the
# "error"/"unparseable" crash verdicts get folded in.
COMMIT_FAIL = {"failed"}
COMMIT_PASS = {"passed"}

TOL = 5e-4  # published rates are quoted to 3 decimals


def rel(path: str) -> str:
    return os.path.relpath(path, REPO)


# ---------------------------------------------------------------------------
# Metrics -- the single local implementation
# ---------------------------------------------------------------------------


@dataclass
class Cells:
    """Confusion cells for one grader on one testbed.

    Positive class = "this step contains an error". A grader either commits
    (pass/fail) or abstains; abstains are split by the gold label into AbsE
    (abstained on an error step) and AbsN (abstained on a clean step).
    """

    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0
    abs_e: int = 0
    abs_n: int = 0

    @property
    def total(self) -> int:
        return self.tp + self.fp + self.tn + self.fn + self.abs_e + self.abs_n

    @property
    def committed(self) -> int:
        return self.tp + self.fp + self.tn + self.fn

    def metrics(self) -> dict:
        n = self.total
        return {
            "cov": self.committed / n if n else None,
            "prec": (self.tp / (self.tp + self.fp)) if (self.tp + self.fp) else None,
            "rec": (
                self.tp / (self.tp + self.fn + self.abs_e)
                if (self.tp + self.fn + self.abs_e)
                else None
            ),
            "fpr": (self.fp / (self.fp + self.tn)) if (self.fp + self.tn) else None,
            "abst": (self.abs_e + self.abs_n) / n if n else None,
            "TP": self.tp,
            "FP": self.fp,
            "TN": self.tn,
            "FN": self.fn,
            "AbsE": self.abs_e,
        }


def tally(judgements) -> Cells:
    """Fold (is_error, verdict) pairs into confusion cells."""
    c = Cells()
    for is_error, verdict in judgements:
        if verdict in COMMIT_FAIL:
            if is_error:
                c.tp += 1
            else:
                c.fp += 1
        elif verdict in COMMIT_PASS:
            if is_error:
                c.fn += 1
            else:
                c.tn += 1
        else:  # abstain, error, unparseable, anything else non-committal
            if is_error:
                c.abs_e += 1
            else:
                c.abs_n += 1
    return c


def cascade(symbolic, judge):
    """Table 3 routing: symbolic commits win; else the judge; else abstain.

    ``symbolic`` and ``judge`` are equal-length sequences of (is_error,
    verdict) pairs over the same items in the same order.
    """
    out = []
    for (is_error, sym_v), (is_error_j, judge_v) in zip(symbolic, judge):
        assert is_error == is_error_j, "gold labels misaligned across graders"
        committed = sym_v in COMMIT_FAIL or sym_v in COMMIT_PASS
        out.append((is_error, sym_v if committed else judge_v))
    return out


# ---------------------------------------------------------------------------
# Expected values, read from the root CLAUDE.md
# ---------------------------------------------------------------------------

# Column header in the CLAUDE.md tables -> canonical field name.
COLUMN_ALIASES = {
    "grader": "grader",
    "cov": "cov",
    "prec": "prec",
    "strict rec": "rec",
    "fpr": "fpr",
    "abst": "abst",
    "tp": "TP",
    "fp": "FP",
    "tn": "TN",
    "fn": "FN",
    "abse": "AbsE",
}

INT_FIELDS = {"TP", "FP", "TN", "FN", "AbsE"}

# Table 3, quoted verbatim from the root CLAUDE.md:
#
#   "Cascade (Table 3), human set: cascade->gpt-oss recall 0.231, cascade->claude
#    0.385. Injection: cascade->gpt-oss recall 0.894 / FPR 0.000; cascade->claude
#    recall 0.957 / FPR 0.130 / precision 0.968."
#
# Prose, so it is transcribed here rather than parsed. check_table3_prose()
# below re-reads the sentence and flags any drift between it and these values.
EXPECTED_CASCADE = {
    ("human_labels", "cascade->gpt-oss-120b"): {"rec": 0.231},
    ("human_labels", "cascade->claude-sonnet-4-5"): {"rec": 0.385},
    # Injection cascade rows revised 2026-08-24 (gold1 re-fixturing + 429
    # re-collection). See root CLAUDE.md, "Injection rows revised 2026-08-24".
    ("error_injection", "cascade->gpt-oss-120b"): {"rec": 0.894, "fpr": 0.000},
    ("error_injection", "cascade->claude-sonnet-4-5"): {
        "rec": 0.957,
        "fpr": 0.130,
        "prec": 0.968,
    },
}


def find_root_doc() -> str:
    for name in ROOT_DOC_CANDIDATES:
        path = os.path.join(REPO, name)
        if os.path.exists(path):
            return path
    sys.exit(
        "FATAL: no root instructions file found at "
        + " or ".join(os.path.join(REPO, n) for n in ROOT_DOC_CANDIDATES)
    )


def _cell(text: str):
    text = text.strip()
    if text.lower() in {"n/a", "-", "--", ""}:
        return None
    return float(text)


def parse_table1(doc_text: str) -> dict:
    """Pull the two Table 1 markdown blocks out of the root instructions file.

    Returns {(dataset, grader): {field: expected}}. Parsed rather than
    transcribed so the check follows the file if the file is edited.
    """
    markers = [
        ("human_labels", r"Human-labeled set"),
        ("error_injection", r"Error-injection suite"),
    ]
    expected = {}
    for dataset, marker in markers:
        m = re.search(marker, doc_text)
        if not m:
            sys.exit(f"FATAL: could not locate the {dataset} Table 1 block in the root doc")
        block = doc_text[m.end() :]
        rows = []
        for line in block.splitlines():
            line = line.strip()
            if not line.startswith("|"):
                if rows:  # table ended
                    break
                continue
            rows.append([c.strip() for c in line.strip("|").split("|")])
        if len(rows) < 3:
            sys.exit(f"FATAL: {dataset} table in the root doc has too few rows")
        header = [COLUMN_ALIASES.get(h.lower(), None) for h in rows[0]]
        if header[0] != "grader":
            sys.exit(f"FATAL: unexpected header in the {dataset} table: {rows[0]}")
        for row in rows[2:]:  # rows[1] is the |---| separator
            if len(row) != len(header):
                sys.exit(f"FATAL: ragged row in the {dataset} table: {row}")
            grader = row[0]
            fields = {}
            for name, raw in zip(header[1:], row[1:]):
                if name is None:
                    sys.exit(f"FATAL: unrecognised column in the {dataset} table: {rows[0]}")
                val = _cell(raw)
                fields[name] = int(val) if (val is not None and name in INT_FIELDS) else val
            expected[(dataset, grader)] = fields
    return expected


def check_table3_prose(doc_text: str, failures: list, doc_path: str) -> None:
    """Cross-check the transcribed EXPECTED_CASCADE against the doc's prose."""
    nums = re.search(
        r"cascade.{0,3}gpt-oss\s+recall\s+(\d+\.\d+),\s+cascade.{0,3}claude\s+(\d+\.\d+)\."
        r"\s+Injection:\s+cascade.{0,3}gpt-oss\s+recall\s+(\d+\.\d+)\s*/\s*FPR\s+(\d+\.\d+);"
        r"\s+cascade.{0,3}claude\s+recall\s+(\d+\.\d+)\s*/\s*FPR\s+(\d+\.\d+)"
        r"\s*/\s*precision\s+(\d+\.\d+)",
        doc_text,
        re.S,
    )
    if not nums:
        # HARD FAILURE, deliberately. This check exists to catch drift between
        # the Table 3 prose and EXPECTED_CASCADE. If the sentence is reworded
        # so the regex no longer matches, the check stops checking anything --
        # and silently falling back to the transcribed constants would let the
        # prose and the constants diverge with no signal. A check that disables
        # itself on a reword is worse than no check.
        failures.append(
            {
                "row": "Table 3 prose re-parse",
                "field": "sentence shape",
                "expected": (
                    'a sentence of the form "Cascade (Table 3), human set: '
                    "cascade->gpt-oss recall A, cascade->claude B. Injection: "
                    "cascade->gpt-oss recall C / FPR D; cascade->claude recall E "
                    '/ FPR F / precision G."'
                ),
                "actual": "no match -- sentence missing or reworded",
                "source": (
                    f"{rel(doc_path)}: the Table 3 prose sentence could not be parsed. "
                    "Restore the sentence shape, or update the regex in "
                    "check_table3_prose() to match the new wording. Do not leave it "
                    "unparseable: this check fails closed on purpose."
                ),
            }
        )
        return
    g = [float(x) for x in nums.groups()]
    transcribed = [
        EXPECTED_CASCADE[("human_labels", "cascade->gpt-oss-120b")]["rec"],
        EXPECTED_CASCADE[("human_labels", "cascade->claude-sonnet-4-5")]["rec"],
        EXPECTED_CASCADE[("error_injection", "cascade->gpt-oss-120b")]["rec"],
        EXPECTED_CASCADE[("error_injection", "cascade->gpt-oss-120b")]["fpr"],
        EXPECTED_CASCADE[("error_injection", "cascade->claude-sonnet-4-5")]["rec"],
        EXPECTED_CASCADE[("error_injection", "cascade->claude-sonnet-4-5")]["fpr"],
        EXPECTED_CASCADE[("error_injection", "cascade->claude-sonnet-4-5")]["prec"],
    ]
    labels = [
        "human cascade->gpt-oss rec",
        "human cascade->claude rec",
        "injection cascade->gpt-oss rec",
        "injection cascade->gpt-oss fpr",
        "injection cascade->claude rec",
        "injection cascade->claude fpr",
        "injection cascade->claude prec",
    ]
    for label, in_doc, in_script in zip(labels, g, transcribed):
        if abs(in_doc - in_script) > 1e-9:
            failures.append(
                {
                    "row": "Table 3 transcription",
                    "field": label,
                    "expected": in_doc,
                    "actual": in_script,
                    "source": f"{rel(doc_path)} (prose) vs EXPECTED_CASCADE in this script",
                }
            )
    print(f"  Table 3 prose in {rel(doc_path)} agrees with the transcribed constants.")


# ---------------------------------------------------------------------------
# Reconstruction: human-labeled set (n=191)
# ---------------------------------------------------------------------------


@dataclass
class Reconstructed:
    dataset: str
    grader: str
    cells: Cells
    sources: list = field(default_factory=list)


def load_human_set():
    """Return (gold, symbolic, gptoss, claude), each a list of (is_error, verdict).

    Gold labels and the symbolic verdicts both come from the hand-labeled
    templates: `human_correct` is the label, `verifier_is_valid` is what the
    verifier committed (null = abstain). Judge verdicts come from the two
    frozen per-step verdict files, aligned by query id and step number.
    """
    label_glob = os.path.join(RESULTS, "label_templates", "*.json")
    label_files = sorted(glob.glob(label_glob))
    if not label_files:
        sys.exit(f"FATAL: no label templates found at {rel(label_glob)}")

    gold = {}       # (qid, step) -> is_error
    symbolic = {}   # (qid, step) -> verdict
    order = []
    for path in label_files:
        with open(path) as fh:
            doc = json.load(fh)
        qid = doc["id"]
        for step in doc["steps"]:
            key = (qid, step["step_number"])
            if key in gold:
                sys.exit(f"FATAL: duplicate step {key} in {rel(path)}")
            order.append(key)
            gold[key] = step["human_correct"] is False
            valid = step.get("verifier_is_valid")
            # The verifier only ever commits a pass or abstains; a False here
            # would be a committed FAIL and is handled for completeness.
            symbolic[key] = (
                "abstain" if valid is None else ("passed" if valid else "failed")
            )

    def load_judge(fname):
        path = os.path.join(RESULTS, fname)
        with open(path) as fh:
            doc = json.load(fh)
        flat = {}
        for qid, steps in doc["verdicts"].items():
            for step in steps:
                flat[(qid, step["step_number"])] = step["verdict"]
        missing = set(order) - set(flat)
        extra = set(flat) - set(order)
        if missing or extra:
            sys.exit(
                f"FATAL: {rel(path)} does not align with the label templates "
                f"({len(missing)} missing, {len(extra)} extra steps)"
            )
        return [flat[k] for k in order], path

    gptoss, gptoss_path = load_judge("judge_step_verdicts.json")
    claude, claude_path = load_judge("judge_step_verdicts_claude.json")

    labels = [gold[k] for k in order]
    pair = lambda verdicts: list(zip(labels, verdicts))
    return {
        "n": len(order),
        "n_errors": sum(labels),
        "symbolic": (pair([symbolic[k] for k in order]), [rel(label_glob)]),
        "gpt-oss-120b": (pair(gptoss), [rel(label_glob), rel(gptoss_path)]),
        "claude-sonnet-4-5": (pair(claude), [rel(label_glob), rel(claude_path)]),
    }


# ---------------------------------------------------------------------------
# Reconstruction: error-injection suite (n=118 graded)
# ---------------------------------------------------------------------------


def load_injection():
    """Return graded-trial judgements for symbolic and both judges.

    A trial counts as graded when `skipped_reason` is null (the other 78 of
    196 attempted trials are mutations that did not apply). Gold positive =
    severity "error"; "control" and "equivalent" trials are the negatives.
    """
    gptoss_path = os.path.join(RESULTS, "error_injection_report_with_judge.json")
    claude_path = os.path.join(RESULTS, "error_injection_report_claude.json")

    def load(path, judge_key):
        with open(path) as fh:
            doc = json.load(fh)
        trials = [t for t in doc["trials"] if t["skipped_reason"] is None]
        keys, labels, sym, judge = [], [], [], []
        for t in trials:
            keys.append((t["gold_id"], t["step_index"], t["mutation_kind"]))
            labels.append(t["severity"] == "error")
            sym.append(t["verdicts"]["symbolic"]["verdict"])
            if judge_key not in t["verdicts"]:
                sys.exit(f"FATAL: trial {keys[-1]} in {rel(path)} has no {judge_key} verdict")
            judge.append(t["verdicts"][judge_key]["verdict"])
        return keys, labels, sym, judge

    g_keys, g_labels, g_sym, g_judge = load(gptoss_path, GPTOSS_KEY)
    c_keys, c_labels, c_sym, c_judge = load(claude_path, CLAUDE_KEY)

    # The symbolic verifier is deterministic and appears in both reports; if
    # the two copies disagree, every downstream number is suspect.
    if g_keys != c_keys:
        sys.exit("FATAL: the two injection reports enumerate different graded trials")
    if g_labels != c_labels:
        sys.exit("FATAL: the two injection reports disagree on trial severities")
    if g_sym != c_sym:
        n = sum(1 for a, b in zip(g_sym, c_sym) if a != b)
        sys.exit(
            f"FATAL: the two injection reports disagree on {n} symbolic verdicts "
            f"({rel(gptoss_path)} vs {rel(claude_path)})"
        )

    pair = lambda verdicts: list(zip(g_labels, verdicts))
    return {
        "n": len(g_keys),
        "n_errors": sum(g_labels),
        "symbolic": (pair(g_sym), [rel(gptoss_path), rel(claude_path)]),
        "gpt-oss-120b": (pair(g_judge), [rel(gptoss_path)]),
        "claude-sonnet-4-5": (pair(c_judge), [rel(claude_path)]),
    }


# ---------------------------------------------------------------------------
# Cross-implementation check: the metrics block embedded in each report
# ---------------------------------------------------------------------------

# Embedded field name -> field name produced by Cells.metrics().
EMBEDDED_FIELDS = {
    "precision": "prec",
    "recall_strict": "rec",
    "fpr": "fpr",
    "abstain_rate": "abst",
    "tp": "TP",
    "fp": "FP",
    "tn": "TN",
    "fn": "FN",
    "abstain_error": "AbsE",
}

INJECTION_REPORTS = (
    "error_injection_report.json",
    "error_injection_report_with_judge.json",
    "error_injection_report_claude.json",
)


def check_embedded_metrics(failures, checked) -> None:
    """Assert each report's stored ``metrics`` block agrees with the cells.

    CLAUDE.md requires ``src.error_injection.compute_metrics`` and
    ``scripts/build_threeway_table.py`` to agree. Table 1 is built by the
    latter, so a drift in the former reaches nothing published and is
    invisible to the Table 1 checks above -- but it still ships, because
    ``package_osf_release.py`` puts these reports in the reviewer release.

    This compares the ``metrics`` block *as stored in the JSON* against a
    reconstruction from the same file's raw ``trials``. It does not import
    ``compute_metrics``; the independent-third-implementation rule in this
    module's docstring still holds.
    """
    for fname in INJECTION_REPORTS:
        path = os.path.join(RESULTS, fname)
        if not os.path.exists(path):
            failures.append(
                {"row": f"embedded/{fname}", "field": "file",
                 "expected": "present", "actual": "absent", "source": rel(path)}
            )
            continue
        with open(path) as fh:
            doc = json.load(fh)
        trials = [t for t in doc["trials"] if t["skipped_reason"] is None]

        for grader, block in sorted(doc.get("metrics", {}).items()):
            # Overall, then each mutation family, reconstructed from trials.
            groups = {"overall": trials}
            for t in trials:
                groups.setdefault(f"by_family/{t['family']}", []).append(t)

            for group, subset in sorted(groups.items()):
                if group == "overall":
                    stored = block.get("overall") or {}
                else:
                    stored = (block.get("by_family") or {}).get(
                        group.split("/", 1)[1]
                    ) or {}
                judgements = [
                    (
                        t["severity"] == "error",
                        (t["verdicts"].get(grader) or {}).get("verdict", "error"),
                    )
                    for t in subset
                ]
                recomputed = tally(judgements).metrics()
                row_label = f"{fname}:{grader}/{group}"
                sources = [rel(path)]
                present = {
                    embedded: local
                    for embedded, local in EMBEDDED_FIELDS.items()
                    if embedded in stored
                }
                if not present:
                    failures.append(
                        {"row": row_label, "field": "metrics block",
                         "expected": "present", "actual": "absent",
                         "source": rel(path)}
                    )
                    continue
                # The reconstruction is the reference, so it goes in as
                # `expected`; the value stored in the JSON is what is under
                # test, so it goes in as `actual`. Same compare() as the
                # published-cell checks -- same tolerance, same int handling.
                expected = {local: recomputed.get(local) for local in present.values()}
                actual = {local: stored.get(embedded) for embedded, local in present.items()}
                compare(row_label, actual, expected, sources, failures, checked)


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def fmt(value):
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    return f"{value:.6f}"


def compare(row_label, actual, expected, sources, failures, checked):
    for field_name, exp in expected.items():
        act = actual.get(field_name)
        checked.append((row_label, field_name))
        if exp is None or act is None:
            ok = exp is None and act is None
        elif field_name in INT_FIELDS:
            ok = int(act) == int(exp)
        else:
            ok = abs(act - exp) <= TOL
        if not ok:
            failures.append(
                {
                    "row": row_label,
                    "field": field_name,
                    "expected": exp,
                    "actual": act,
                    "source": ", ".join(sources),
                }
            )


def print_row(row_label, actual, expected):
    order = ["cov", "prec", "rec", "fpr", "abst", "TP", "FP", "TN", "FN", "AbsE"]
    bits = []
    for f in order:
        if f not in expected:
            continue
        exp, act = expected[f], actual.get(f)
        if exp is None and act is None:
            mark = "ok"
        elif exp is None or act is None:
            mark = "MISMATCH"
        elif f in INT_FIELDS:
            mark = "ok" if int(act) == int(exp) else "MISMATCH"
        else:
            mark = "ok" if abs(act - exp) <= TOL else "MISMATCH"
        shown = fmt(act) if act is None or isinstance(act, int) else f"{act:.3f}"
        bits.append(f"{f}={shown}{'' if mark == 'ok' else ' <<MISMATCH'}")
    print(f"    {row_label:<46} " + "  ".join(bits))


def main() -> int:
    doc_path = find_root_doc()
    with open(doc_path) as fh:
        doc_text = fh.read()

    print("verify_published_tables.py")
    print(f"  repo root        : {REPO}")
    print(f"  expected values  : {rel(doc_path)}")
    print(f"  reconstructed from: {rel(RESULTS)}/ (read-only; no API calls)")
    if os.path.basename(doc_path) != "CLAUDE.md":
        print(
            f"  WARNING: canonical CLAUDE.md is absent at the repo root; "
            f"falling back to {os.path.basename(doc_path)}."
        )
    print()

    expected_t1 = parse_table1(doc_text)
    failures: list = []
    checked: list = []
    check_table3_prose(doc_text, failures, doc_path)
    print()

    human = load_human_set()
    injection = load_injection()

    print(f"  human set        : n={human['n']} steps, {human['n_errors']} errors")
    print(
        f"  injection suite  : n={injection['n']} graded, {injection['n_errors']} error, "
        f"{injection['n'] - injection['n_errors']} non-error"
    )
    print()

    # Sanity-check the reconstructed denominators against the CLAUDE.md prose.
    for label, got, want, where in [
        ("human set size", human["n"], 191, "CLAUDE.md: 'n=191 steps, 13 errors'"),
        ("human error count", human["n_errors"], 13, "CLAUDE.md: 'n=191 steps, 13 errors'"),
        ("injection graded", injection["n"], 118, "CLAUDE.md: 'n=118 graded'"),
        ("injection errors", injection["n_errors"], 94, "CLAUDE.md: '94 error, 24 non-error'"),
    ]:
        checked.append(("denominators", label))
        if got != want:
            failures.append(
                {"row": "denominators", "field": label, "expected": want,
                 "actual": got, "source": where}
            )

    datasets = [("human_labels", human), ("error_injection", injection)]
    graders = ["symbolic", "gpt-oss-120b", "claude-sonnet-4-5"]

    print("  Table 1 -- three-way comparison")
    for dataset, data in datasets:
        print(f"    [{dataset}]")
        for grader in graders:
            judgements, sources = data[grader]
            actual = tally(judgements).metrics()
            exp = expected_t1.get((dataset, grader))
            if exp is None:
                failures.append(
                    {"row": f"{dataset}/{grader}", "field": "row",
                     "expected": "present in CLAUDE.md Table 1", "actual": "absent",
                     "source": rel(doc_path)}
                )
                continue
            compare(f"{dataset}/{grader}", actual, exp, sources, failures, checked)
            print_row(f"{dataset}/{grader}", actual, exp)
    print()

    print("  Table 3 -- cascade (symbolic commits win; else judge; else abstain)")
    for dataset, data in datasets:
        print(f"    [{dataset}]")
        sym_j, sym_src = data["symbolic"]
        for grader in ["gpt-oss-120b", "claude-sonnet-4-5"]:
            judge_j, judge_src = data[grader]
            actual = tally(cascade(sym_j, judge_j)).metrics()
            key = (dataset, f"cascade->{grader}")
            exp = EXPECTED_CASCADE[key]
            sources = sorted(set(sym_src + judge_src))
            compare(f"{dataset}/cascade->{grader}", actual, exp, sources, failures, checked)
            print_row(f"{dataset}/cascade->{grader}", actual, exp)
    print()

    before = len(failures)
    print("  Embedded metrics blocks vs. reconstruction from the same trials")
    check_embedded_metrics(failures, checked)
    n_embedded = len(failures) - before
    if n_embedded:
        print(f"    {n_embedded} embedded field(s) disagree -- listed below.")
    else:
        print("    every stored metrics block agrees with its own trials.")
    print()

    print(f"  {len(checked)} fields checked, {len(failures)} mismatched.")
    if failures:
        print()
        print("  MISMATCHES")
        for f in failures:
            print(f"    row      : {f['row']}")
            print(f"    field    : {f['field']}")
            print(f"    expected : {fmt(f['expected']) if isinstance(f['expected'], (int, float, type(None))) else f['expected']}")
            print(f"    actual   : {fmt(f['actual']) if isinstance(f['actual'], (int, float, type(None))) else f['actual']}")
            print(f"    source   : {f['source']}")
            print()
        print("  FAIL")
        return 1

    print("  OK -- every published cell reproduces from the frozen verdicts.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
