# Multi-round evaluation runs

Each subdirectory is one timestamped run of `tools/trustfin_rounds.py`.
Nothing here is frozen benchmark evidence; these are synthetic practice probes.

| Directory | Status | Notes |
|---|---|---|
| `20261029T115226Z` | superseded | Harness bug: `raw_*` summary fields were read from the wrong nesting level and stored as `null`. The per-case JSONL is still valid, but do not cite the summary files. |
| `20261029T115718Z` | **authoritative** | 77 cases across rounds A–E, including the unit-mismatch and unit-scale guardrail checks. Cited in `docs/REPORT.md` section 12. |

Files per run:

- `round_<X>.jsonl` — one record per case: case, raw evaluation, guardrail decision, and compliance flags.
- `round_<X>.summary.json` — per-round metrics.
- `summary.json` — all rounds plus run metadata and benchmark.
- `SUMMARY.md` — human-readable tables.

Recompute summaries from stored JSONL without loading the model:

```bash
python3 tools/trustfin_rounds.py --resummarize runs/rounds/20261029T115718Z
```
