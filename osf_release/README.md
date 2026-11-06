# Step-level math grader comparison — anonymous release

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
