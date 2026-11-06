| dataset | grader | coverage | precision | recall | fpr | abstain_rate | tp | fp | tn | fn | abs_e | abs_n | total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| human_labels | symbolic | 0.471 | n/a | 0.000 | 0.000 | 0.529 | 0 | 0 | 83 | 7 | 6 | 95 | 191 |
| human_labels | gpt-oss-120b | 0.995 | 0.667 | 0.462 | 0.017 | 0.005 | 6 | 3 | 175 | 6 | 1 | 0 | 191 |
| human_labels | claude-sonnet-4-5 | 0.953 | 0.370 | 0.769 | 0.100 | 0.047 | 10 | 17 | 153 | 2 | 1 | 8 | 191 |
| error_injection | symbolic | 0.356 | 1.000 | 0.234 | 0.000 | 0.644 | 22 | 0 | 20 | 0 | 72 | 4 | 118 |
| error_injection | gpt-oss-120b | 1.000 | 1.000 | 0.894 | 0.000 | 0.000 | 84 | 0 | 24 | 10 | 0 | 0 | 118 |
| error_injection | claude-sonnet-4-5 | 0.958 | 0.938 | 0.957 | 0.286 | 0.042 | 90 | 6 | 15 | 2 | 2 | 3 | 118 |
