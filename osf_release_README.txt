================================================================================
  ARTIFACT RELEASE — Step-Level Math Grader Comparison
  Paper: "What Symbolic Graders Can't See and LLM Judges Won't Ignore:
         Abstention-Aware Comparison of Step-Level Math Graders"
================================================================================

This file accompanies osf_release.zip. It describes the full contents of the
release, how to unpack and run the code, and how to reproduce every table and
figure in the paper.

--------------------------------------------------------------------------------
QUICK START
--------------------------------------------------------------------------------

1. Download osf_release.zip from this OSF project and unzip it:

     unzip osf_release.zip
     cd osf_release

2. Create a Python virtual environment and install dependencies:

     python3 -m venv .venv
     source .venv/bin/activate          # on Windows: .venv\Scripts\activate
     pip install -r requirements-grading.txt

   Required packages (all publicly available via pip):
     - sympy >= 1.12
     - numpy >= 1.24
     - matplotlib >= 3.7
     - python-dotenv >= 1.0.0

   No API keys are needed to regenerate tables and figures.
   API keys are only needed if you want to re-call the LLM judges from scratch
   (see the "Re-running LLM judges" section below).

3. Set PYTHONPATH and rebuild all paper tables and figures:

     export PYTHONPATH=.          # on Windows: set PYTHONPATH=.
     python scripts/build_threeway_table.py
     python scripts/build_cascade_table.py
     python scripts/build_confidence_curves.py
     python scripts/build_judge_agreement.py
     python scripts/build_paired_stats_summary.py --skip-rerun
     python scripts/build_cost_latency.py
     python scripts/render_figures.py

   All output is written back to results/. The --skip-rerun flag tells
   build_paired_stats_summary.py to use the frozen per-judge stat files
   rather than re-calling the LLM judges.

   Every regenerated file will match the frozen one shipped in this release
   byte-for-byte, EXCEPT results/cost_latency.json / .md: its ms_per_step and
   seconds_total figures are wall-clock timings of the symbolic verifier on
   YOUR machine, so they will legitimately differ from the paper's reported
   ~60ms/step (~11s total, one laptop CPU). This is expected and is disclosed
   in the paper (Results, cost/latency paragraph) as machine-dependent; it is
   not a sign that anything is broken.

--------------------------------------------------------------------------------
WHAT IS IN osf_release.zip
--------------------------------------------------------------------------------

The zip extracts to a single top-level folder called osf_release/ with the
following structure:

osf_release/
|
|-- README.md                       Shorter in-package README (this file is
|                                    the detailed companion posted alongside
|                                    the zip on OSF; contents overlap but this
|                                    file has more detail)
|-- LABELING_RUBRIC.md              Full human annotation rubric (see below)
|-- MANIFEST.md                     SHA-256 checksums for every file
|-- requirements-grading.txt        Python dependencies
|-- .env.example                    Template for API keys (optional)
|
|-- data/
|   |-- eval_queries.py             The 30-query benchmark (signals & systems)
|   |-- eval_run_config.json        Generator/retrieval config (paths scrubbed)
|   |-- concept_graph.json          Concept graph used by the RAG pipeline
|
|-- src/
|   |-- verifier.py                 SymPy-based symbolic step verifier
|   |-- llm_judge.py                LLM-as-judge client (swappable backend)
|   |-- error_injection.py          Programmatic mutation harness (14 operators)
|   |-- evaluator.py                Evaluation loop: runs graders over derivations
|   |-- stats.py                    McNemar test, Holm-Bonferroni, bootstrap CIs
|   |-- label_scorer.py             Scoring helpers for human labels
|   |-- pipeline.py                 RAG pipeline orchestration
|   |-- rag_chain.py                Retrieval-augmented generation chain
|   |-- retriever.py                Document retriever
|   |-- concept_graph.py            Concept graph loader
|   |-- ingest.py                   Corpus ingestion utilities
|   |-- run_metadata.py             Schema-v2 reproducibility envelope
|   |-- load_env.py                 Environment variable loader
|   |-- __init__.py
|
|-- scripts/
|   |-- build_threeway_table.py     Builds Table 1 (3-way grader comparison)
|   |-- build_cascade_table.py      Builds Table 3 (cascade policy results)
|   |-- build_confidence_curves.py  Writes confidence_curves.json AND renders
|   |                               Figure 2 (precision-coverage curves;
|   |                               fig3_precision_coverage.png/.svg)
|   |-- build_judge_agreement.py    Builds judge-vs-human agreement stats
|   |-- build_paired_stats_summary.py  Builds Table 2 (McNemar + Holm-Bonferroni
|   |                               over 6 tests)
|   |-- build_cost_latency.py       Computes cost/call-count summary
|   |-- build_headline_artifacts.py Builds a standalone summary covering ONLY
|   |                               symbolic + gpt-oss-120b (no Claude); not
|   |                               the same scope as Table 1 / Figure 1, which
|   |                               include all three graders. Optional: not
|   |                               part of the Quick Start sequence above.
|   |-- render_figures.py           Renders Figure 1 and Figure 3 (PNG + SVG).
|   |                               Figure 2 is rendered separately by
|   |                               build_confidence_curves.py (see above).
|   |-- symbolic_vs_judge_stats.py  Per-judge paired stats (called by above)
|   |-- judge_vs_human.py           Runs judge verdicts against human labels
|   |-- test_error_injection.py     Tests for error injection harness
|   |-- test_stats.py               Unit tests for statistical functions
|
|-- results/
|   |-- label_templates/            Human step labels for all 29 queries
|   |   |-- Q01.json ... Q30.json   Per-query label files (Q12 absent: no
|   |                               usable derivation for that query)
|   |
|   |-- human_error_fragment_labels.json   In/out-of-fragment codes for the
|   |                                      13 human-flagged error steps
|   |-- judge_step_verdicts.json    Frozen gpt-oss-120b verdicts (all 191 steps)
|   |-- judge_step_verdicts_claude.json    Frozen claude-sonnet-4-5 verdicts
|   |-- error_injection_report.json        Symbolic-verifier-only injection
|   |                                      trials (no LLM judge; not used by
|   |                                      the reproduction scripts below)
|   |-- error_injection_report_with_judge.json  Symbolic + gpt-oss-120b
|   |                                           injection trials (used by
|   |                                           build_threeway_table.py and
|   |                                           build_cascade_table.py)
|   |-- error_injection_report_claude.json Symbolic + claude-sonnet-4-5
|   |                                      injection trials (used by
|   |                                      build_threeway_table.py and
|   |                                      build_cascade_table.py)
|   |-- eval_primary_redacted.json  Frozen generator run (27/29 queries),
|   |                               absolute paths and hostnames scrubbed
|   |-- eval_q01_q07_redacted.json  Earlier same-model run for Q01 and Q07
|   |-- threeway_table.json/.csv/.md    Table 1 (grader comparison)
|   |-- stats_paired_summary.json/.md   Table 2 (McNemar + Holm-Bonferroni)
|   |-- cascade_table.json/.csv/.md     Table 3 (cascade results)
|   |-- stats_symbolic_vs_judge.json/.md   Per-judge stats (gpt-oss)
|   |-- stats_symbolic_vs_claude.json/.md  Per-judge stats (Claude)
|   |-- judge_agreement.json/.md        Cohen kappa agreement stats
|   |-- confidence_curves.json          Data for Figure 2 (precision-coverage)
|   |-- cost_latency.json/.md           Cost and call-count summary
|   |-- headline_table_1.json/.csv/.md  Standalone summary covering ONLY
|   |                                   symbolic + gpt-oss-120b (no Claude row
|   |                                   at all). Despite the filename, this is
|   |                                   NOT the paper's Table 1, which has all
|   |                                   three graders -- see threeway_table.*
|   |                                   for the actual Table 1.
|   |-- headline_figure_data.json       Same gpt-oss-only scope as above. Not
|   |                                   consumed to render the actual paper
|   |                                   figure -- render_figures.py builds
|   |                                   Figure 1 from threeway_table.json.
|   |-- label_score_report.json         Verifier-only sanity-check summary
|   |                                   (coverage/FN counts on the human set;
|   |                                   regenerate with
|   |                                   `python -m src.label_scorer`). Not a
|   |                                   paper table; cross-checks against
|   |                                   Table 1's symbolic row.
|   |-- figures/
|       |-- fig1_grader_comparison.png/.svg   Paper Figure 1
|       |-- fig2_injection_by_family.png/.svg  Paper Figure 3 (see note below)
|       |-- fig3_precision_coverage.png/.svg   Paper Figure 2 (see note below)

NOTE ON FIGURE FILENAMES vs. PAPER FIGURE NUMBERS:
The image filenames (fig1_*, fig2_*, fig3_*) reflect the order these figures
were originally drafted in, not their final position in the paper. Sections
were reordered after the figures were named, so the ACTUAL paper numbering is:
    Paper Figure 1 = fig1_grader_comparison.png/.svg   (grader comparison)
    Paper Figure 2 = fig3_precision_coverage.png/.svg  (precision-coverage curves)
    Paper Figure 3 = fig2_injection_by_family.png/.svg (injection by family)
The paper's LaTeX source uses \ref{} for all in-text figure numbers, so the
PDF itself is correct and internally consistent; only the file/script naming
is offset by one. Use the mapping above, not the filenames, to match a figure
in this release to a figure in the paper.

--------------------------------------------------------------------------------
THE HUMAN LABELS (results/label_templates/)
--------------------------------------------------------------------------------

Each Q??.json file contains the ground-truth human labels for every step of
that query's derivation. The labeling rubric used by the annotator is in
LABELING_RUBRIC.md. Key points about the rubric:

  - A step is CORRECT if the stated operation, given the previous step,
    produces the current expression and is a valid move under standard
    signals-and-systems conventions.

  - A step is INCORRECT if it contains a math error: wrong algebra, wrong
    limits, wrong sign, dropped term, wrong definition/setup, etc.

  - NOTATION-STRICTNESS RULE: A step is also marked incorrect if boundary
    or integration-by-parts notation is written wrong, if a claimed equation
    is incomplete, or if the problem setup/definition is wrong relative to the
    intended question -- even when the underlying algebra would evaluate to the
    correct final number. This rule is stricter than pure local algebraic
    validity, and is the reason some human-labeled errors lie OUTSIDE the
    symbolic CAS fragment.

  - IMPORTANT: The LLM judges never see this rubric. They receive only the
    generic step prompt in src/llm_judge.py. Human-set precision and recall
    in the paper therefore measure agreement with this rubric, not
    oracle-matched accuracy.

  - 29 labeled queries (Q01-Q11, Q13-Q30). Q12 has no usable derivation.
  - 191 steps total; 13 human-flagged errors (definition=4, algebra=3,
    missing_term=6).

Data provenance: step text for Q01 and Q07 comes from an earlier frozen run
with the same model (qwen-3-235b-a22b-instruct-2507, temperature 0). All other
27 queries come from the primary frozen eval (eval_primary_redacted.json).
The labels are the ground truth for all grading comparisons; the eval files
document generator provenance.

--------------------------------------------------------------------------------
THE ERROR-INJECTION SUITE
--------------------------------------------------------------------------------

src/error_injection.py implements 14 mutation operators across 5 families:

  ERROR families (grader should flag these):
    - algebraic: sign_flip, coeff_x2, coeff_half, drop_term, exponent_bump,
                 constant_shift, missing_factor_of_2
    - bounds:    swap_integral_limits, drop_upper_limit
    - identifier: rename_variable

  NEGATIVE families (grader should NOT flag these):
    - equivalent: commute_operands, expand_power, distribute
    - control:    no_op (identity mutation)

The 9 hand-authored gold derivations that serve as injection base cases are
standard signals/systems results (Laplace transforms, Fourier properties,
partial fractions, etc.). They are independent of the RAG generator and verified
to pass the symbolic verifier unmutated.

The 9 gold derivations contain 14 mutable step sites in total; every
mutation operator is attempted against every site, for 14 x 14 = 196
attempted (step x mutation) trials. Of these, 78 are skipped because the
mutation does not apply to that step's expression. The graded suite is
118 trials (94 error, 24 non-error: 14 control + 10 equivalent).

--------------------------------------------------------------------------------
THE SYMBOLIC VERIFIER (src/verifier.py)
--------------------------------------------------------------------------------

Uses SymPy to test algebraic equivalence between a derivation step and the
expected continuation. Key behavior:
  - Commits (PASS or FAIL) only when it can establish or refute equivalence.
  - ABSTAINS otherwise -- it never guesses.
  - Additional matchers handle composite expressions and inequalities.

To run the verifier on the frozen derivations without re-calling LLM judges:

    python scripts/build_threeway_table.py

This reads from the frozen label_templates/ and eval JSON files and does not
require any API keys.

--------------------------------------------------------------------------------
RE-RUNNING LLM JUDGES (OPTIONAL)
--------------------------------------------------------------------------------

The frozen judge verdicts are included (judge_step_verdicts.json for gpt-oss
and judge_step_verdicts_claude.json for Claude). You do NOT need to re-call the
judges to reproduce any table or figure in the paper.

To re-call the judges from scratch (not required for reproduction):

1. Install the LLM-client extras. requirements-grading.txt only installs
   what is needed to rebuild tables/figures; the judge backends need two
   more packages that are commented out there by default:

     pip install langchain-openai langchain-anthropic

   Without this, src/llm_judge.py raises ImportError as soon as it tries to
   instantiate either backend (both --fake mode and the table-rebuild
   scripts above are unaffected -- they never reach this code path).

2. Copy .env.example to .env and fill in your API keys:

     cp .env.example .env
     # then edit .env:
     #   CEREBRAS_API_KEY=<your key>       for gpt-oss-120b
     #   ANTHROPIC_API_KEY=<your key>      for claude-sonnet-4-5

3. Run the judge. IMPORTANT: judge_vs_human.py always writes to
   results/judge_step_verdicts.json by default regardless of backend, so you
   MUST pass --out explicitly for the second (Claude) call or it will
   overwrite the frozen gpt-oss verdicts file:

     python scripts/judge_vs_human.py      # gpt-oss (default backend)
     JUDGE_BACKEND=anthropic \
       python scripts/judge_vs_human.py --out results/judge_step_verdicts_claude.json

4. Then rebuild the stats summary:

     python scripts/build_paired_stats_summary.py   # (without --skip-rerun)

The open-weight judge (gpt-oss-120b) is served via a Cerebras OpenAI-compatible
endpoint. Any OpenAI-compatible endpoint can be used by setting JUDGE_BACKEND
and the corresponding API key in .env. The prompt is STEP_RUBRIC + STEP_TEMPLATE
in src/llm_judge.py, temperature 0.

--------------------------------------------------------------------------------
STATISTICAL METHODS (src/stats.py)
--------------------------------------------------------------------------------

The paired significance tests in the paper use:

  - Exact McNemar test on discordant pairs (primary test for paired binary
    outcomes, following Dror et al. 2018).
  - Holm-Bonferroni correction over the family of 6 tests
    (3 metrics x 2 judges) at alpha=0.05.
  - 10,000-replicate bootstrap 95% confidence intervals on mean deltas.
    Bootstrap CIs for caught_error (n=13 error steps) are marked uninformative
    when they hit a {0,1} boundary.

See scripts/build_paired_stats_summary.py for the full pipeline.

Judge-vs-human and judge-vs-judge agreement (Cohen's kappa, e.g. kappa=0.547
gpt-oss vs human, 0.464 claude vs human, 0.306 gpt-oss vs claude) is computed
by scripts/build_judge_agreement.py using the generic cohen_kappa() /
landis_koch_label() helpers in src/stats.py. This is a two-model agreement
statistic, not inter-annotator agreement between human coders (see the note
above on IAA tooling).

--------------------------------------------------------------------------------
WHAT IS NOT INCLUDED (intentional omissions)
--------------------------------------------------------------------------------

  - API keys or any .env file with real credentials
  - Copyrighted course PDFs and the Chroma vectorstore (not needed to
    reproduce the grading comparison; derivations are frozen in label_templates/
    and eval JSON files)
  - LLM judge response cache (verdicts are frozen in judge_step_verdicts*.json)
  - Author machine paths, hostnames, or repository names (all scrubbed)
  - The RAG pipeline's full corpus retrieval (frozen derivations are sufficient)
  - Inter-annotator-agreement tooling: this submission uses a single human
    annotator and reports no inter-coder agreement statistic (see the paper's
    Limitations section and Appendix F). A dual-annotation pipeline exists in
    the authors' working repository but is deliberately excluded from this
    release so the released code matches exactly what was run and reported.

--------------------------------------------------------------------------------
INTEGRITY CHECK
--------------------------------------------------------------------------------

MANIFEST.md contains the SHA-256 hash and byte size of every file in this
release. To verify integrity after unpacking, generate your own hashes and
compare them against the "sha256" column in MANIFEST.md:

  Using Python (from inside the extracted osf_release/ directory):

    python3 -c "import hashlib, pathlib; [print(hashlib.sha256(p.read_bytes()).hexdigest(), p) for p in sorted(pathlib.Path('.').rglob('*')) if p.is_file()]"

  Or, on macOS / Linux, using sha256sum / shasum:

    find . -type f -exec sha256sum {} \;        # Linux
    find . -type f -exec shasum -a 256 {} \;     # macOS

Any mismatch against MANIFEST.md means the file was altered or corrupted
during transfer.

--------------------------------------------------------------------------------
RUNNING THE TEST SUITE
--------------------------------------------------------------------------------

pytest is not in requirements-grading.txt (it is only needed to run these
tests, not to rebuild any paper table or figure). Install it first:

    pip install pytest
    python -m pytest scripts/test_stats.py           # statistical functions
    python -m pytest scripts/test_error_injection.py # injection harness

Both suites pass out of the box against the frozen artifacts in this release
(8 and 29 tests respectively, as of this writing).

--------------------------------------------------------------------------------
LICENSE / USE
--------------------------------------------------------------------------------

Released for the purpose of reviewing and reproducing the results in the
accompanying paper. If you use these artifacts in your own work, please cite
the paper. Do not redistribute any third-party copyrighted course material
if you obtain it separately to extend this work.

================================================================================
END OF README
================================================================================
