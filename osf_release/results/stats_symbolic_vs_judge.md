# Symbolic verifier vs LLM judge — paired step-level stats

- Steps paired: 191
- Human-error steps: 13
- Bootstrap reps: 10000, permutations: 10000, CI: 0.95

| metric | mean(symbolic) | mean(judge) | delta (judge-sym) | 95% CI | McNemar p |
| --- | --- | --- | --- | --- | --- |
| committed | 0.471 | 0.995 | 0.524 | [0.450, 0.592] | 0.000 |
| correct_committed | 0.435 | 0.948 | 0.513 | [0.440, 0.586] | 0.000 |
| caught_error | 0.000 | 0.462 | 0.462 | uninformative (n small / CI hits boundary) | 0.031 |

Interpretation: `committed` = coverage (fraction of steps with a verdict); `correct_committed` = fraction of steps where the grader committed AND matched the human label (abstains score 0); `caught_error` = recall on human-labeled error steps. Primary paired binary test is exact McNemar; bootstrap CIs are descriptive. Holm–Bonferroni adjustments over the six-test family are in `stats_paired_summary.json`.
