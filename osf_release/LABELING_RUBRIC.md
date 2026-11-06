# Human labeling rubric (step-level)

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
