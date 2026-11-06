# Telecom assistant

RAG over your course PDFs: **OpenAI embeddings** (`text-embedding-3-small`) in Chroma, answers from a **Cerebras** model via the OpenAI-compatible API.

## Quick install

From this folder (`telecom-assistant/`):

**macOS / Linux**

```bash
chmod +x install.sh
./install.sh
```

**Windows (PowerShell)**

```powershell
.\install.ps1
```

**Manual (any OS)**

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Configuration

1. Copy `.env.example` to `.env`.
2. Set `OPENAI_API_KEY` (embeddings and Chroma indexing).
3. Set `CEREBRAS_API_KEY` (chat model).

## Usage

1. Put PDFs in `data/pdfs/`.
2. Build the vector index (run from `telecom-assistant/` with venv active):

   ```bash
   python -m src.ingest
   ```

3. Start the Streamlit UI:

   ```bash
   streamlit run main.py
   ```

4. Optional: try retrieval from the terminal:

   ```bash
   python -m src.retriever
   ```

## Evaluation and metrics

The evaluation pipeline reports **separated, independently-interpretable metrics** rather than a single conflated pass-rate. The goal is to distinguish three failure modes that the earlier single-number `Pass%` mixed together:

1. the LLM writing unparseable SymPy,
2. a legitimate derivation step that the verifier cannot check symbolically (e.g. substitution, integration), and
3. an actual mathematical mistake.

### What the evaluator does

For each benchmark query in `data/eval_queries.py` we:

1. Retrieve chunks with **System A** (plain similarity search) and **System B** (concept-graph augmented).
2. Generate a structured JSON derivation with the LLM (`src/rag_chain.py`).
3. Verify each derivation step with `src/verifier.py`.
4. Compare the model's `final_result` against the query's `expected_final_result` using symbolic equivalence.
5. Save per-query results to `results/eval_*.json`.
6. Write a per-query human-labelling template to `results/label_templates/Q??.json`.

### Metrics reported

| Metric | Denominator | Interpretation |
|---|---|---|
| `parse_rate_pct`        | all steps                     | Fraction of step-expressions that parsed into SymPy. Directly measures LLM output formatting quality. |
| `equiv_rate_pct`        | steps the verifier checked    | Of the steps we actually attempted equivalence on, how many matched the previous step. |
| `checked_total`         | —                             | Number of steps the verifier attempted equivalence on (excludes first steps, parse failures, natural-language steps, ROC conditions, and integral-skips). |
| `checked_failed`        | —                             | Steps the verifier **confirmed** as mathematically wrong (sign flip, numerical mismatch). Strong signal. |
| `final_answer_match`    | per query                     | `match` / `mismatch` / `unparseable` / `no_expected` — symbolic equivalence between `final_result` and `expected_final_result`. |
| `category_counts`       | all steps                     | Histogram over `passed`, `failed`, `parse_fail`, `natural_lang`, `roc`, `integral_skip`, `first_step`, `unclear`. |

`legacy_pass_rate_pct` is still written for backwards compatibility, but should not be used as a headline metric.

### Grader validation (synthetic error injection)

Pass rates on raw LLM output cannot by themselves tell us whether a grader would catch errors that **are** present. To probe that directly we inject controlled perturbations into hand-authored correct derivations and, for each grader under test, check whether that step is flagged.

Every injected trial carries a severity label:

- `"error"`: a genuine mathematical mistake was introduced — a good grader should return `failed`.
- `"equivalent"`: the step was rewritten into an algebraically-identical but notationally different form — a good grader should NOT return `failed`.
- `"control"`: no perturbation at all — any `failed` here is a false positive on the baseline.

This lets us report the four standard classifier metrics — **precision, recall, FPR, abstain rate** — per grader, rather than just a single "detection rate".

```bash
# symbolic verifier only (offline)
python -u -m src.error_injection

# symbolic + LLM-as-judge (hits the judge endpoint)
python -u -m src.error_injection --with-judge
```

The harness is organized around two abstractions:

- **`MutationSpec`** — 14 mutations in five families:
  - *algebraic* (error): `sign_flip`, `coeff_x2`, `coeff_half`, `drop_term`, `exponent_bump`, `constant_shift`, `missing_factor_of_2`
  - *bounds* (error): `swap_integral_limits`, `drop_upper_limit`
  - *identifier* (error): `rename_variable` (skips bound/dummy variables to avoid mis-labeled trials)
  - *equivalent* (should NOT flag): `commute_operands`, `expand_power`, `distribute`
  - *control*: `no_op`
- **`Grader`** — `SymbolicGrader` (wraps `verify_derivation`) and `LLMJudgeGrader` (wraps `src.llm_judge.judge_step`). Adding a new grader is one small class.

Per-run output lives at `results/error_injection_report.json` and contains:

- `config` (seed, graders, mutation catalog)
- `metrics[grader]` with `overall`, `by_family`, and a raw `(severity × verdict)` confusion matrix
- `trials[…]` with every injected case (original/mutated expressions plus each grader's verdict, confidence, and reasoning)
- `warnings` for any fixtures the viability filter dropped

Gold fixtures that fail to verify cleanly at baseline are removed up-front so they can't confound injection results.

### Human labelling

Because parseability and symbolic verifiability are each limited, the ultimate ground truth is human judgement. The evaluator writes a label template per query to `results/label_templates/Q??.json`. A reviewer then:

1. Sets each step's `human_correct` to `true` or `false`.
2. Sets `error_type` to one of `algebra`, `sign`, `missing_term`, `substitution`, `definition`, `none`.
3. Sets `final_answer_correct` to `true`/`false`.
4. Sets `_filled` to `true`.

Then:

```bash
python -u -m src.label_scorer
```

computes a confusion matrix of the verifier against the human labels, plus precision and recall for error detection, and writes `results/label_score_report.json`.

### How to interpret the numbers

- Low `parse_rate_pct` means the LLM is writing strings SymPy cannot parse — this is an **LLM-output-quality** problem, not a math problem. Improve the prompt first.
- High `parse_rate_pct` and low `equiv_rate_pct` means either the verifier is missing legitimate-rewrite rules (substitution, integration, piecewise) **or** the derivation contains real errors. Disambiguate with human labels and with error-injection results.
- `checked_failed > 0` is the only strong automated signal of a real math mistake.
- `final_answer_match == mismatch` is the headline end-to-end correctness signal.

### Run evaluation

Each run writes **schema v2** JSON: top-level keys `schema_version`, `run` (frozen `config` + environment `metadata`), and `results` (the per-query list). Legacy files that are a bare list still load if you point tooling at them. Override the frozen config path with `TELECOM_EVAL_CONFIG=/path/to.json` (defaults to `data/eval_run_config.json`).

Retrieval behavior is controlled from `data/eval_run_config.json`:

- `retrieval.plain_similarity_k`
- `retrieval.graph_prerequisite_depth`
- `retrieval.graph_prerequisite_keyword_k`
- `retrieval.graph_enabled`
- `retrieval.graph_max_prereqs`
- `retrieval.graph_max_extra_chunks`

Pipeline retrieval mode can be set with `evaluation.pipeline_retrieval_mode` (`graph` or `plain`) or overridden per run via `TELECOM_PIPELINE_RETRIEVAL_MODE=plain`.

`data/eval_queries.py` now supports structured expected finals via `expected_answer`:

- `{"type":"symbolic","expression":"..."}`
- `{"type":"text","value":"..."}`
- `{"type":"inequality","expression":"re(s) > -2"}`
- `{"type":"composite","parts":[...],"aggregation":"all"}` for mixed answers (e.g., transform + ROC)

`expected_final_result` is still accepted for backward compatibility, but new queries should use `expected_answer`.

From `telecom-assistant/` with virtualenv active:

```bash
# Full benchmark
python -u -m src.evaluator

# Re-run a subset
python -u -m src.evaluator rerun Q03 Q08 Q09

# Retrieval ablation (runs both plain + graph pipeline modes)
python -u -m src.evaluator ablate
python -u -m src.evaluator ablate Q03 Q08 Q10

# Verifier sensitivity study
python -u -m src.error_injection

# Score verifier against human labels (after labelling)
python -u -m src.label_scorer

# Paired bootstrap CI + significance (plain vs graph from one ablation file)
PYTHONPATH=. python scripts/paired_stats.py \
  --from-ablation results/eval_ablation_20260420_220157.json \
  --arm-a plain --arm-b graph \
  --metric final_match_binary

# Paired stats between two separate runs (e.g., symbolic vs llm-judge)
PYTHONPATH=. python scripts/paired_stats.py \
  --file-a results/eval_symbolic.json \
  --file-b results/eval_judge.json \
  --arm-a symbolic --arm-b judge \
  --metric final_match_binary

# One-shot paper bundle (final_match, parse, equiv, checked_failed)
PYTHONPATH=. python scripts/paired_stats.py \
  --from-ablation results/eval_ablation_20260420_220157.json \
  --arm-a plain --arm-b graph \
  --bundle paper

# LLM-as-judge vs human (uses the same 85 labeled steps the verifier was scored on)
PYTHONPATH=. python scripts/judge_vs_human.py            # real Cerebras-backed judge
PYTHONPATH=. python scripts/judge_vs_human.py --fake     # offline plumbing test
PYTHONPATH=. python scripts/judge_vs_human.py --resume --continue-on-rate-limit \
  --max-attempts 8 --base-wait-s 30 --max-wait-s 600
```

Per-step judge responses are cached at `results/judge_cache/<sha>.json` so
reruns are free; the cache key embeds the judge model so swapping models
auto-invalidates.

Output files are written to `results/`.

## Layout

| Path | Role |
|------|------|
| `data/pdfs/` | Your textbooks (PDF) |
| `vectorstore/` | Chroma persistence (created by ingest) |
| `src/ingest.py` | PDFs → chunks → embeddings → Chroma |
| `src/retriever.py` | Query Chroma (debug / scripts) |
| `src/rag_chain.py` | Retriever + Cerebras LLM |
| `src/verifier.py` | Symbolic step verification (SymPy) |
| `src/evaluator.py` | Batch benchmark driver with separated metrics |
| `src/error_injection.py` | Synthetic error-injection harness (verifier sensitivity) |
| `src/stats.py` | Paired bootstrap CI + permutation significance utilities |
| `src/label_scorer.py` | Grader-vs-human scoring (works for any grader via `valid_key`) |
| `src/llm_judge.py` | LLM-as-judge grader (Cerebras backend + `FakeJudge` for tests) |
| `scripts/paired_stats.py` | CLI for per-query paired comparisons across runs/arms |
| `scripts/judge_vs_human.py` | Score the LLM-judge on the same human-labeled steps as the verifier |
| `data/eval_run_config.json` | Declarative knobs for eval runs (snapshotted into each result JSON) |
| `src/run_metadata.py` | Build per-run metadata (git, package versions, corpus hashes) |
| `main.py` | Streamlit chat UI |

See `progress.md` for phase checklist and notes.
