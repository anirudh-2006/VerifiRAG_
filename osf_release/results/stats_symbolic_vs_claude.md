# Symbolic verifier vs LLM judge — paired step-level stats

- Steps paired: 191
- Human-error steps: 13
- Bootstrap reps: 10000, permutations: 10000, CI: 0.95

| metric | mean(symbolic) | mean(judge) | delta (judge-sym) | 95% CI | McNemar p |
| --- | --- | --- | --- | --- | --- |
| committed | 0.471 | 0.953 | 0.482 | [0.403, 0.560] | 0.000 |
| correct_committed | 0.435 | 0.853 | 0.419 | [0.335, 0.497] | 0.000 |
| caught_error | 0.000 | 0.769 | 0.769 | uninformative (n small / CI hits boundary) | 0.002 |

Interpretation: `committed` = coverage (fraction of steps with a verdict); `correct_committed` = fraction of steps where the grader committed AND matched the human label (abstains score 0); `caught_error` = recall on human-labeled error steps. Primary paired binary test is exact McNemar; bootstrap CIs are descriptive. Holm–Bonferroni adjustments over the six-test family are in `stats_paired_summary.json`.
