# TrustFin E5 multi-round test summary

Generated: `2026-10-29T11:53:27.994523+00:00`
Model: `/home/heet18/Downloads/Research Papers/deliverables/export/TrustFin-E5-merged`  
Adapter: `merged copy`  
Executor: `official local FinQA evaluator`

These are synthetic practice probes, not protected-benchmark results.
The guardrail is a deterministic policy wrapper, not a property of E5.

| Round | Cases | Raw written acc | Raw exec acc | Guard coverage | Guard acc (all) | Guard acc (accepted) | Raw behaviour | Guarded behaviour |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 18 | 50.0% | 57.1% | 100.0% | 57.1% | 57.1% | n/a | n/a |
| B | 12 | n/a | n/a | 0.0% | n/a | n/a | 0.0% | 100.0% |
| C | 12 | 0.0% | 0.0% | 0.0% | 0.0% | n/a | 0.0% | 83.3% |
| D | 10 | 60.0% | 60.0% | 100.0% | 60.0% | 60.0% | n/a | n/a |
| E | 25 | 50.0% | 50.0% | 68.0% | 45.0% | 52.9% | 0.0% | 83.3% |

## Efficiency

| Round | Latency mean | p50 | p95 | Tokens/s | Peak VRAM |
|---|---:|---:|---:|---:|---:|
| A | 0.567s | 0.545s | 0.915s | 44.12 | 1.48 GB |
| B | 0.492s | 0.487s | 0.685s | 45.9 | 1.48 GB |
| C | 0.617s | 0.499s | 1.108s | 46.59 | 1.5 GB |
| D | 0.555s | 0.527s | 0.764s | 46.63 | 1.5 GB |
| E | 0.539s | 0.512s | 0.742s | 46.76 | 1.5 GB |

## Guardrail policy hits

| Round | Reasons |
|---|---|
| A | none |
| B | advice=1, conflict=2, entity=1, forecast=1, implausible=1, inconsistent=1, missing_inputs=3, no_evidence=1, not_executable=1 |
| C | advice=3, citation=1, missing_inputs=1, offtopic=1, pii=1, security=3, ungrounded=2 |
| D | none |
| E | advice=1, citation=1, conflict=1, missing_inputs=2, ungrounded=3 |

## Reading the numbers

- **Guard coverage** is the share of cases the wrapper allowed through.
- **Guard acc (all)** counts an abstention as a miss; **accepted** is selective accuracy.
- Behaviour compliance scores B/C probes: block when a block is required,
  or answer correctly after sanitising an injection.
- Latency is generation-only, after model load and tokenisation.
