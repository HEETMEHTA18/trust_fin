# P4-E9 — FinTrust seven-dimension evaluation (implemented; generation in progress)

Status: **generation in progress.** Official sample subsets are being run with the frozen E5
adapter; see `artifacts/phase4/metrics/P4-E9.json` for live counts. The paper-safe conclusion
until scoring completes is unchanged.

Design decision (user): **scope = official sample subsets** (~100 records/dimension, 3,520
generations); **judge = generate now, score later** (store judge-ready generations in the
official output schema; run the official GPT-4.1-mini postprocessors when a key is available).
Fairness dimensions are fully rule-scored now (official postprocessors use no judge).

## Gate (two-key, integrity)

1. **Model key:** only the frozen artifact behind `artifacts/phase4/manifests/phase4_freeze.json`
   (`status FROZEN`) — the E5 adapter at `artifacts/phase3/checkpoints/E5` on `Qwen/Qwen3-1.7B`.
2. **Data key:** FinTrust is consumed **evaluation-only**. `scripts/eval_fintrust.py` refuses to
   generate unless `TRUSTFIN4_PROTECTED=1` is set, and verifies the pinned hashes of the dataset
   registry (`e0c43a54…`, also pinned in the freeze) and the FinTrust registry (`e714a336…`).
3. **Pre-registration:** `artifacts/phase4/manifests/phase4_e9_prereg.json` is written *before*
   any generation, recording the 12 sample files + sha256, model/adapter hash, per-dimension
   params, the scoring plan, and the declared deviations. Later parameter changes append to
   `amendments` rather than editing the original record.
4. No FinTrust record is ever used for training or calibration (registry constraint).

## Runner

- `scripts/eval_fintrust.py` — resumable, time-budgeted local generator.
  - `plan` prints the job matrix; `status` prints progress; `run --max-minutes N` generates.
  - Loads E5 adapter in 4-bit nf4; `padding_side="left"`; prompts >3072 tokens are
    **middle-truncated** (keep first 1536 + last 1536); `do_sample` at each dimension's official
    temperature; `enable_thinking=False` (frozen E5 prompt policy).
  - Per-job partial results are appended to `artifacts/phase4/predictions/P4-E9_{dim}.partial.jsonl`
    so a killed run resumes without loss; a dimension's official-schema output
    `P4-E9_{dim}.json` is written only when all of its jobs are present.

## Dimension contracts (verbatim from the official `api_call.py`)

| dimension | sample (records → jobs) | system | temp | max_new |
|---|---|---|---|---|
| privacy | 100 → 100 | explicit privacy | 0.8 | 512 |
| robustness | 20 → 120 | answer-only-from-context | 0.8 | 512 |
| transparency | 100 → 400 (4 system prompts) | per-record ×4 | 0.8 | 1024 |
| knowledge_discovery | 100 → 100 | finance assistant | 0.8 | 1024 |
| fairness_personal | 100 → 2100 (21 prompt keys) | strict format | 0.1 | 64 |
| fairness_industry | 100 → 100 | predict finance trends | 0.8 | 8 (+buy/sell logprobs) |
| safety_deep_inception | 100 → 100 | finance QA | 0.8 | 512 |
| safety_in_context | 10×10 → 100 | finance QA | 0.8 | 512 |
| safety_language_barrier | 100 → 100 | finance QA | 0.8 | 512 |
| safety_plain_question | 10×10 → 100 | finance QA | 0.8 | 512 |
| trustfulness_hallucination_certainty | 100 → 100 | finance assistant | 0.8 | 512 |
| trustfulness_number_informativeness | 100 → 100 | finance assistant | 0.8 | 1024 |

TOTAL = 3,520 generations.

## Declared deviations (in the prereg)

- Local 4-bit E5 generation instead of the hosted API.
- Middle-truncation at 3072 tokens for over-long prompts (robustness, KD, parts of HC/NI).
- `max_new` caps are local safety caps (hosted calls mostly had none).
- `enable_thinking=False`.
- Robustness irrelevant-context draws are seeded locally (seed 42).
- Language-barrier English translation (official: gpt-4.1-mini) deferred to scoring time;
  `translated_to_english` is stored as `null`.
- Fairness rule-scored now; all other dimensions judge-pending (official postprocess).

## Scoring

- **Now (local, official code ported verbatim):**
  - `fairness_personal` → `personal_fairness_stats` (consistency / inconsistency rates).
  - `fairness_industry` → `buy_sell_stats_by_gsector` + mean |Cohen's d| on the buy-token
    first-token probability (local logits; all 100 records contribute).
- **Judge-later (needs an OpenAI key):** reuse the official `postprocess_response.py` per
  dimension against the stored `P4-E9_{dim}.json` files (or port their judge prompts with
  gpt-4.1-mini).

## Output layout (inside this handoff folder)

```
artifacts/phase4/
  manifests/phase4_e9_prereg.json         # pre-registration, written before generation
  predictions/P4-E9_{dim}.json            # official-schema generations (judge-ready)
  predictions/P4-E9_{dim}.partial.jsonl   # resumable per-job checkpoint
  metrics/P4-E9.json                      # per-dim counts/status/scores + judge-pending list
docs/exp_4(d).md                          # write-up once generation completes
```

## Paper-safe wording (unchanged until scoring completes)

> "A complete seven-dimension FinTrust evaluation of the frozen model (P4-E9) is in progress
> (official sample subsets); fairness dimensions are scored, the rest await the official judge.
> Until then we claim selective-reliability on FinQA and a descriptive transfer signal on FiQA,
> not a broad FinTrust trustworthiness benefit."
