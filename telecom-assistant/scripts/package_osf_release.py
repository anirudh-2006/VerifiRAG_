#!/usr/bin/env python3
"""Build an anonymous OSF release tree from this repo.

Writes to ../osf_release/ (sibling of the package root) by default:
  - redacted frozen artifacts (no /Users/..., no hostnames, no project names)
  - grading code + table-building scripts
  - human labels + labeling rubric
  - open-weight judge prompt/config (from src/llm_judge.py)
  - anonymized README + MANIFEST

Does NOT copy: .env, .venv, PDFs, vectorstore, judge_cache, paper PDF,
git metadata, or anything under results/archive/.

Usage (from package root):
  PYTHONPATH=. python scripts/package_osf_release.py
  PYTHONPATH=. python scripts/package_osf_release.py --out /tmp/osf_math_graders
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


ROOT = Path(__file__).resolve().parents[1]

# Absolute-path / identity strings to scrub from JSON text.
_PATH_RE = re.compile(r"/Users/[^\"'\s]+")
_IDENTITY_RES = [
    (re.compile(r"Anirudhs-MacBook-Air\.local", re.I), "REDACTED-HOST"),
    (re.compile(r"anirudh-2006", re.I), "REDACTED"),
    (re.compile(r"anirudh", re.I), "REDACTED"),
    (re.compile(r"VerifiRAG_2", re.I), "REDACTED-REPO"),
    (re.compile(r"VerifiRAG", re.I), "REDACTED-REPO"),
    (re.compile(r"telecom-assistant-default", re.I), "grader-study-default"),
    (re.compile(r"telecom-assistant", re.I), "grader-study"),
]


def _scrub_str(s: str) -> str:
    s = _PATH_RE.sub("<REDACTED-PATH>", s)
    for rx, repl in _IDENTITY_RES:
        s = rx.sub(repl, s)
    return s


def _scrub_obj(obj: Any) -> Any:
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            # Drop presence flags that only reveal the author's env.
            if k == "api_keys_present":
                continue
            if k in {"hostname", "username", "user", "python_executable",
                     "cwd", "project_root"} and isinstance(v, str):
                out[k] = "REDACTED"
                continue
            out[_scrub_str(k) if isinstance(k, str) else k] = _scrub_obj(v)
        return out
    if isinstance(obj, list):
        return [_scrub_obj(x) for x in obj]
    if isinstance(obj, str):
        return _scrub_str(obj)
    return obj


def _copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n")


def _copy_scrubbed_json(src: Path, dst: Path) -> None:
    obj = json.loads(src.read_text())
    _write_json(dst, _scrub_obj(obj))


LABELING_RUBRIC = """# Human labeling rubric (step-level)

This is the annotator-facing rubric used to produce
`results/label_templates/Q*.json`. LLM judges **never** see this document;
they receive only the step prompt in `src/llm_judge.py` (`STEP_RUBRIC` /
`STEP_TEMPLATE`). Human-set precision/recall in the paper are therefore
**agreement with this rubric**, not oracle-matched accuracy against a shared
instruction.

## Unit of labeling

Label each derivation step independently, given:
- the question,
- all prior steps (expression + operation),
- the current step's expression and operation.

Mark `human_correct` as `true` or `false`. If `false`, set `error_type` to one of:
`algebra`, `sign`, `missing_term`, `substitution`, `definition`, `none`.

## Decision rule

A step is **correct** if, given the previous step, the stated operation produces
the current expression and the step is a valid move toward answering the
question under standard signals-and-systems conventions used in the corpus.

A step is **incorrect** if it contains a mathematical mistake (wrong algebra,
wrong limits, wrong sign, dropped term, wrong definition/setup, etc.).

## Notation-strictness rule

Mark a step **incorrect** even when the underlying algebra would evaluate to the
right final number, if:

1. **Boundary / IBP notation is written wrong** (e.g. an integration-by-parts
   boundary term at infinity is omitted from the written step even when that
   term is zero), or
2. **A claimed equation is incomplete** (LHS written without the stated RHS, a
   ratio bar with no right-hand side, a “divide both sides” claim that does not
   produce a complete equation), or
3. **Problem setup / definition is wrong relative to the intended target**
   (e.g. bilateral Laplace written where the question requires unilateral;
   wrong pulse/square-wave definition relative to the intended signal).

This rule is deliberate and stricter than pure local algebraic validity. It is
why some human-error labels lie **outside** the symbolic CAS fragment (see
`results/human_error_fragment_labels.json`).

## What not to penalize

- Verifier tooling failures (`parse_fail`, `unclear`) when the written math is
  correct.
- Alternative but equivalent algebraic forms when the operation is valid.
- Style / verbosity alone.

## Provenance note for this release

- 29 labeled queries (Q01–Q11, Q13–Q30); Q12 has no usable derivation.
- 191 steps total; 13 human-flagged errors
  (`definition`=4, `algebra`=3, `missing_term`=6).
- Step text for Q01 and Q07 comes from an earlier generator run with the same
  model ID (`qwen-3-235b-a22b-instruct-2507`); the other 27 queries come from
  the primary frozen eval (`results/eval_primary_redacted.json`). Both use
  temperature 0. Labels are the ground truth for grading comparisons; the eval
  files document generator provenance.
"""


README = """# Step-level math grader comparison — anonymous release

Companion release for the paper:

> *What Symbolic Graders Can't See and LLM Judges Won't Ignore:
> Abstention-Aware Comparison of Step-Level Math Graders*

This package contains everything needed to **regenerate the paper's tables and
figures from frozen artifacts**, and to **re-run the symbolic verifier and
(optionally) the open-weight LLM judge**.

## Contents

| Path | What it is |
| --- | --- |
| `src/` | Symbolic verifier, LLM judge client, error-injection harness, stats |
| `scripts/` | Table/figure builders used in the paper (Appendix F) |
| `data/eval_queries.py` | The 30-query benchmark definitions |
| `data/eval_run_config.json` | Generator / retrieval config (redacted paths) |
| `data/concept_graph.json` | Concept graph used by the RAG pipeline |
| `results/label_templates/` | All human step labels (29 queries, 191 steps) |
| `LABELING_RUBRIC.md` | Human annotation rubric (notation-strictness) |
| `results/human_error_fragment_labels.json` | In/out-of-fragment codes for the 13 errors |
| `results/judge_step_verdicts*.json` | Frozen LLM-judge verdicts (gpt-oss + Claude) |
| `results/error_injection_report*.json` | Frozen injection trials + grader verdicts |
| `results/*table*.json` / `*.md` | Regenerated paper tables |
| `results/figures/` | Paper figures |
| `results/eval_primary_redacted.json` | Primary frozen generator run (27/29 queries), paths scrubbed |
| `results/eval_q01_q07_redacted.json` | Earlier same-model run supplying Q01 and Q07 |
| `requirements-grading.txt` | Minimal deps to rebuild tables / run verifier |
| `MANIFEST.md` | File list + SHA256 |

## What is *not* included (on purpose)

- API keys / `.env`
- Copyrighted course PDFs and the Chroma vectorstore (not needed to reproduce
  the **grading** comparison; derivations are frozen in the label templates and
  eval JSONs)
- LLM judge response cache (verdicts are frozen)
- Author machine paths, hostnames, or repository names
- Inter-annotator-agreement tooling: this submission uses a single human
  annotator and reports no inter-coder agreement statistic (see Limitations,
  Appendix F). A dual-annotation pipeline exists in the authors' working
  repository but is deliberately excluded here so the released code matches
  exactly what was run and reported.

## Reproduce paper tables (no API keys)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-grading.txt
export PYTHONPATH=.
python scripts/build_threeway_table.py
python scripts/build_cascade_table.py
python scripts/build_confidence_curves.py
python scripts/build_judge_agreement.py
python scripts/build_paired_stats_summary.py --skip-rerun
python scripts/build_cost_latency.py
python scripts/render_figures.py
```

## Open-weight judge configuration

The open-weight judge used in the paper is:

- model: `gpt-oss-120b`
- temperature: `0`
- prompt: `STEP_RUBRIC` + `STEP_TEMPLATE` in `src/llm_judge.py`
- backend: OpenAI-compatible client (`JUDGE_BACKEND` default; key via env)

Claude (`claude-sonnet-4-5`) is the second judge; its frozen verdicts are
included. Re-calling Claude requires an Anthropic key and is **not** required
to regenerate tables.

## License / use

Use for reviewing and reproducing the paper. Do not redistribute any
third-party course PDFs if you obtain them separately.
"""


MANIFEST_HEADER = """# OSF release manifest

Generated by `scripts/package_osf_release.py`.
All JSON under `results/` that originated as run envelopes have absolute paths,
hostnames, and identity strings scrubbed.

"""


SRC_FILES = [
    "src/__init__.py",
    "src/verifier.py",
    "src/llm_judge.py",
    "src/error_injection.py",
    "src/stats.py",
    "src/label_scorer.py",
    "src/run_metadata.py",
    "src/load_env.py",
    # Optional but useful if someone re-runs generator pieces later:
    "src/evaluator.py",
    "src/pipeline.py",
    "src/rag_chain.py",
    "src/retriever.py",
    "src/ingest.py",
    "src/concept_graph.py",
]

SCRIPT_FILES = [
    "scripts/build_threeway_table.py",
    "scripts/build_cascade_table.py",
    "scripts/build_confidence_curves.py",
    "scripts/build_judge_agreement.py",
    "scripts/build_paired_judge_test.py",
    "scripts/build_paired_stats_summary.py",
    "scripts/build_cost_latency.py",
    "scripts/build_headline_artifacts.py",
    "scripts/render_figures.py",
    "scripts/symbolic_vs_judge_stats.py",
    "scripts/judge_vs_human.py",
    "scripts/test_stats.py",
    "scripts/test_error_injection.py",
    # NOTE: scripts/make_iaa_subset.py and scripts/compute_iaa.py are
    # deliberately excluded. This submission has a single annotator and
    # reports no inter-annotator agreement (see Limitations, Appendix F);
    # including a working dual-annotation pipeline in the release would
    # contradict that. Reintroduce them only alongside an actual IAA run.
]

RESULT_FILES = [
    "results/threeway_table.json",
    "results/threeway_table.md",
    "results/threeway_table.csv",
    "results/cascade_table.json",
    "results/cascade_table.md",
    "results/cascade_table.csv",
    "results/confidence_curves.json",
    "results/judge_agreement.json",
    "results/judge_agreement.md",
    "results/paired_judge_test.json",
    "results/stats_paired_summary.json",
    "results/stats_paired_summary.md",
    "results/stats_symbolic_vs_judge.json",
    "results/stats_symbolic_vs_judge.md",
    "results/stats_symbolic_vs_claude.json",
    "results/stats_symbolic_vs_claude.md",
    "results/cost_latency.json",
    "results/cost_latency.md",
    "results/human_error_fragment_labels.json",
    "results/judge_step_verdicts.json",
    "results/judge_step_verdicts_claude.json",
    "results/error_injection_report.json",
    "results/error_injection_report_with_judge.json",
    "results/error_injection_report_claude.json",
    "results/label_score_report.json",
    # NOTE: results/headline_table_1.{json,md,csv} and
    # results/headline_figure_data.json are deliberately excluded. They were
    # generated before the crash-as-abstain fix and carry superseded numbers
    # that contradict Table 1 (injection judge recall 0.909 vs 0.851, human
    # judge recall 0.500 vs 0.462, abstain 0.000 vs 0.059), with confusion
    # cells summing to 111 against a stated total of 118. Nothing renders
    # from them -- render_figures.py builds Figure 1 from threeway_table.json.
    # Do not reintroduce them without regenerating from the current frozen
    # inputs and re-checking every cell against Table 1.
]

# Paths that must never appear in the release tree, regardless of contents.
# Checked by presence after the tree is built; does not replace the
# content-level identity sweep above.
_FORBIDDEN_RELEASE_FILES = (
    "CLAUDE.md",
    "PAPER_CONTEXT_FOR_CLAUDE.md",
    "VENUE_STRATEGY_FOR_CLAUDE.md",
)
_FORBIDDEN_RELEASE_DIRS = (
    ".claude",
    ".cursor",
    ".git",
    ".venv",
)


def _assert_no_forbidden_release_paths(out: Path) -> None:
    """Hard-fail if a forbidden path is present anywhere in the release tree."""
    offenders: List[str] = []
    for p in out.rglob("*"):
        rel = p.relative_to(out).as_posix()
        if p.name in _FORBIDDEN_RELEASE_FILES:
            offenders.append(rel)
        elif p.is_dir() and p.name in _FORBIDDEN_RELEASE_DIRS:
            offenders.append(rel + "/")
    if offenders:
        raise SystemExit(
            "forbidden path would be included in the release: "
            + ", ".join(offenders)
        )


FIGURE_GLOBS = [
    "results/figures/fig1_grader_comparison.png",
    "results/figures/fig1_grader_comparison.svg",
    "results/figures/fig2_injection_by_family.png",
    "results/figures/fig2_injection_by_family.svg",
    "results/figures/fig3_precision_coverage.png",
    "results/figures/fig3_precision_coverage.svg",
]


def _extract_qid_from_eval(src: Path, qids: Set[str], dst: Path) -> None:
    raw = json.loads(src.read_text())
    if isinstance(raw, list):
        results = raw
        envelope: Dict[str, Any] = {"schema_version": 1, "results": results}
    else:
        envelope = raw
        results = envelope.get("results", [])
    keep = []
    for r in results:
        qid = r.get("id") or r.get("query_id")
        if qid in qids:
            keep.append(r)
    envelope = _scrub_obj({**envelope, "results": keep}) if not isinstance(raw, list) else {
        "schema_version": 1,
        "note": "Extracted subset; paths/identity scrubbed.",
        "results": _scrub_obj(keep),
    }
    if isinstance(raw, list):
        _write_json(dst, envelope)
    else:
        # Keep run metadata but scrubbed; restrict results.
        env2 = _scrub_obj(raw)
        env2["results"] = _scrub_obj(keep)
        env2.setdefault("note", "Paths/identity scrubbed for anonymous release.")
        _write_json(dst, env2)


def build(out: Path) -> None:
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    # Docs
    (out / "README.md").write_text(README)
    (out / "LABELING_RUBRIC.md").write_text(LABELING_RUBRIC)

    # Minimal grading requirements (no streamlit / chroma needed for tables).
    (out / "requirements-grading.txt").write_text(
        "\n".join(
            [
                "sympy>=1.12",
                "numpy>=1.24",
                "matplotlib>=3.7",
                "python-dotenv>=1.0.0",
                "# Optional — only needed to re-call LLM judges:",
                "# langchain>=0.2.0",
                "# langchain-openai>=0.1.0",
                "# langchain-anthropic>=0.1.0",
                "",
            ]
        )
    )

    # .env.example without secrets
    (out / ".env.example").write_text(
        "\n".join(
            [
                "# Optional — only for re-calling judges, not for table rebuilds",
                "JUDGE_MODEL=gpt-oss-120b",
                "JUDGE_API_KEY_ENV=CEREBRAS_API_KEY",
                "CEREBRAS_API_KEY=",
                "ANTHROPIC_JUDGE_MODEL=claude-sonnet-4-5",
                "ANTHROPIC_API_KEY=",
                "JUDGE_BACKEND=openai_compatible",
                "",
            ]
        )
    )

    for rel in SRC_FILES + SCRIPT_FILES + RESULT_FILES + FIGURE_GLOBS:
        src = ROOT / rel
        if not src.exists():
            raise FileNotFoundError(f"missing required file: {rel}")
        _copy_file(src, out / rel)

    # Labels
    for p in sorted((ROOT / "results" / "label_templates").glob("Q*.json")):
        _copy_file(p, out / "results" / "label_templates" / p.name)

    # Data (no PDFs)
    for rel in [
        "data/__init__.py",
        "data/eval_queries.py",
        "data/eval_run_config.json",
        "data/concept_graph.json",
    ]:
        _copy_scrubbed_json(ROOT / rel, out / rel) if rel.endswith(".json") else _copy_file(
            ROOT / rel, out / rel
        )

    # Redacted evals
    primary = ROOT / "results" / "eval_20260513_011648.json"
    _copy_scrubbed_json(primary, out / "results" / "eval_primary_redacted.json")
    q01q07_src = ROOT / "results" / "archive" / "eval_20260420_121016.json"
    _extract_qid_from_eval(
        q01q07_src, {"Q01", "Q07"}, out / "results" / "eval_q01_q07_redacted.json"
    )

    # Scrub identity strings from copied text sources (comments/docstrings).
    text_suffixes = {".py", ".md", ".txt", ".json", ".csv", ".example"}
    for p in out.rglob("*"):
        if not p.is_file() or p.suffix not in text_suffixes:
            continue
        if p.suffix == ".json":
            try:
                obj = json.loads(p.read_text())
            except Exception:
                text = _scrub_str(p.read_text())
                p.write_text(text)
                continue
            scrubbed = _scrub_obj(obj)
            _write_json(p, scrubbed)
            continue
        text = p.read_text(errors="replace")
        new = _scrub_str(text)
        if new != text:
            p.write_text(new)

    # Final identity sweep on all text-ish files
    bad = []
    needles = [
        b"/Users/",
        b"anirudh",
        b"Anirudh",
        b"VerifiRAG",
        b"telecom-assistant",
        b"MacBook",
        b"sk-ant-",
    ]
    for p in out.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix in {".png", ".svg", ".pyc"}:
            continue
        data = p.read_bytes()
        for n in needles:
            if n.lower() in data.lower() if n[:1].isalpha() else n in data:
                # allow 'sk-' only if it looks like a key assignment with value
                if n == b"sk-" and b"sk-ant-" not in data and b"sk-proj" not in data:
                    continue
                bad.append((str(p.relative_to(out)), n.decode(errors="replace")))
                break
    if bad:
        raise SystemExit(f"identity leak remains in release tree: {bad[:20]}")

    _assert_no_forbidden_release_paths(out)

    # Manifest with hashes
    import hashlib

    lines = [MANIFEST_HEADER, "| path | bytes | sha256 |", "| --- | ---: | --- |"]
    for p in sorted(x for x in out.rglob("*") if x.is_file()):
        rel = p.relative_to(out).as_posix()
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        lines.append(f"| `{rel}` | {p.stat().st_size} | `{h}` |")
    (out / "MANIFEST.md").write_text("\n".join(lines) + "\n")

    print(f"Wrote anonymous OSF tree to: {out}")
    print(f"Files: {sum(1 for _ in out.rglob('*') if _.is_file())}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT.parent / "osf_release",
        help="Output directory (will be replaced)",
    )
    args = ap.parse_args()
    build(args.out.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
