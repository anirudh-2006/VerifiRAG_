# Cost / latency price tag

Symbolic (local): 59.6 ms/step mean, 11.4s for 191 steps, **$0**.

Token estimate per human judge call: ~419 in / ~58 out (n=191).

| policy (human, n=191) | LLM calls | est. USD (gpt-oss) | est. USD (claude) |
| --- | ---: | ---: | ---: |
| judge alone | 191 | $0.036 | $0.406 |
| cascade (sym→judge on abstain) | 101 (47% fewer) | $0.019 | $0.215 |

| policy (injection, n=118) | LLM calls | est. USD (gpt-oss) | est. USD (claude) |
| --- | ---: | ---: | ---: |
| judge alone | 118 | $0.022 | $0.251 |
| cascade | 76 (36% fewer) | $0.014 | $0.162 |

Prices: gpt-oss-120b $0.35/$0.75 per MTok in/out (Cerebras); claude-sonnet-4-5 $3/$15 (Anthropic). LLM wall latency not instrumented.
