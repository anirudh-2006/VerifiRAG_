# Paired stats summary (McNemar + Holm–Bonferroni)

3 metrics × 2 judges (exact McNemar); Holm–Bonferroni α=0.05

| judge | metric | Δ | 95% CI | McNemar p | Holm p | reject@0.05 |
| --- | --- | ---: | --- | ---: | ---: | --- |
| gpt-oss-120b | committed | +0.524 | [0.450, 0.592] | 1.58e-30 | 9.47e-30 | yes |
| gpt-oss-120b | correct_committed | +0.513 | [0.440, 0.586] | 1.59e-28 | 7.97e-28 | yes |
| gpt-oss-120b | caught_error | +0.462 | uninformative (n=13) | 0.0312 | 0.0312 | yes |
| claude-sonnet-4-5 | committed | +0.482 | [0.403, 0.560] | 1.59e-22 | 6.37e-22 | yes |
| claude-sonnet-4-5 | correct_committed | +0.419 | [0.335, 0.497] | 3.67e-18 | 1.1e-17 | yes |
| claude-sonnet-4-5 | caught_error | +0.769 | uninformative (n=13) | 0.00195 | 0.00391 | yes |
