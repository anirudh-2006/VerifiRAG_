# VerifiRAG — Symbolic vs. LLM Graders for Multi-Step Math Derivations

A reproducible comparison of a SymPy-based symbolic verifier against two LLM-as-judge graders (`gpt-oss-120b`, `claude-sonnet-4-5`) for grading multi-step mathematical derivations produced by a retrieval-augmented assistant.

## Overview

Automatic grading of multi-step math derivations is a prerequisite for evaluating math-capable RAG and reasoning systems, yet symbolic checkers and LLM judges are rarely compared under a single protocol. This project provides:

- **Three graders** evaluated on two independent testbeds:
  - A SymPy symbolic verifier (high precision, high abstention)
  - Two LLM judges from independent labs (high coverage, some false positives)
- **Two testbeds**:
  - A human-labeled set (191 steps, 13 errors from a 30-query RAG benchmark)
  - A synthetic error-injection suite (14 mutation operators × 14 mutable step sites = 196 attempted trials, 118 graded)
- **Cascade policies** combining symbolic + LLM graders
- **Paired significance tests** (McNemar + bootstrap CIs)

## Key findings

| Grader | Human-set recall | Injection recall | Injection FPR | Abstain rate |
|---|---|---|---|---|
| Symbolic verifier | 0.000 | 0.234 | 0.000 | 0.529 (human) / 0.644 (inj) |
| gpt-oss-120b | 0.462 | 0.894 | 0.000 | 0.005 |
| claude-sonnet-4-5 | 0.769 | 0.957 | 0.286 | 0.047 |

The symbolic verifier never false-alarms but misses all semantic errors. LLM judges catch errors the verifier cannot, at the cost of false positives on equivalence-preserving rewrites. The qualitative pattern is model-independent; the operating point is model-dependent.

## Repository structure

```
telecom-assistant/
├── src/                    # Core grading code
│   ├── verifier.py         #   SymPy symbolic step verifier
│   ├── llm_judge.py        #   LLM-as-judge client (OpenAI + Anthropic backends)
│   ├── error_injection.py  #   Synthetic mutation harness (14 operators, 5 families)
│   ├── stats.py            #   McNemar, bootstrap CI, Cohen's kappa
│   ├── evaluator.py        #   Batch evaluation driver
│   ├── label_scorer.py     #   Grader-vs-human scoring
│   └── ...                 #   RAG pipeline components
├── scripts/                # Table/figure builders & tests
│   ├── build_threeway_table.py
│   ├── build_cascade_table.py
│   ├── build_confidence_curves.py
│   ├── render_figures.py
│   ├── test_error_injection.py
│   └── ...
├── results/                # Frozen artifacts (read-only research outputs)
│   ├── threeway_table.json         # Table 1: three-way grader comparison
│   ├── cascade_table.json          # Cascade policy results
│   ├── figures/                    # Paper figures (PNG + SVG)
│   ├── label_templates/            # Human step labels (29 queries, 191 steps)
│   ├── judge_step_verdicts*.json   # Frozen LLM judge verdicts
│   ├── error_injection_report*.json # Injection trial results
│   └── ...
├── data/
│   ├── eval_queries.py     # 30-query benchmark definitions
│   ├── pdfs/               # Source corpus (MIT signals & systems)
│   └── concept_graph.json  # Concept graph for RAG retrieval
├── paper/                  # MathNLP 2026 paper source (ACL style, LaTeX)
├── paper-judge/            # JUDGe @ NeurIPS 2026 paper source (NeurIPS style, LaTeX)
└── notes/                  # Internal research notes
osf_release/                # Anonymized public release package
osf_release_README.txt      # Detailed companion documentation
```

## Setup

```bash
cd telecom-assistant
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Environment variables (optional — only for live LLM calls)

Copy `telecom-assistant/.env.example` to `telecom-assistant/.env` and fill in:

```
OPENAI_API_KEY=...        # embeddings + Chroma indexing
CEREBRAS_API_KEY=...      # gpt-oss-120b judge
ANTHROPIC_API_KEY=...     # claude-sonnet-4-5 judge
```

No API keys are needed to rebuild tables and figures from frozen artifacts.

## Reproducing paper results

All tables and figures can be regenerated from the frozen artifacts in `results/` without any API keys:

```bash
cd telecom-assistant
export PYTHONPATH=.
./.venv/bin/python scripts/build_threeway_table.py
./.venv/bin/python scripts/build_cascade_table.py
./.venv/bin/python scripts/build_confidence_curves.py
./.venv/bin/python scripts/build_judge_agreement.py
./.venv/bin/python scripts/build_paired_stats_summary.py --skip-rerun
./.venv/bin/python scripts/render_figures.py
```

## Running the evaluation pipeline

```bash
cd telecom-assistant

# Full benchmark evaluation
PYTHONPATH=. ./.venv/bin/python -u -m src.evaluator

# Symbolic verifier sensitivity (error injection, offline)
PYTHONPATH=. ./.venv/bin/python -u -m src.error_injection

# Score graders against human labels
PYTHONPATH=. ./.venv/bin/python -u -m src.label_scorer
```

## Tests

```bash
cd telecom-assistant
scripts/run_tests.sh
```

This runs `test_error_injection.py` (30 tests) and `test_stats.py`, and exits nonzero on any failure.

## Error injection suite

The injection harness (`src/error_injection.py`) applies 14 mutation operators across 5 families to 9 hand-authored gold derivations:

- **Error families** (grader should flag): `sign_flip`, `coeff_x2`, `coeff_half`, `drop_term`, `exponent_bump`, `constant_shift`, `missing_factor_of_2`, `swap_integral_limits`, `drop_upper_limit`, `rename_variable`
- **Negative families** (grader should NOT flag): `commute_operands`, `expand_power`, `distribute`, `no_op`

## Papers

This codebase produced two papers (shared code, different framing):

1. **MathNLP 2026** (ACL style, 8pp) — full three-way comparison with cascade policies
2. **JUDGe @ NeurIPS 2026** (NeurIPS style, 4pp) — focused on LLM judge disagreement patterns

Both paper sources are included under `telecom-assistant/paper/` and `telecom-assistant/paper-judge/`.

## License

Research code released for reproducibility. If you use these artifacts, please cite the papers. Do not redistribute copyrighted course PDFs.
