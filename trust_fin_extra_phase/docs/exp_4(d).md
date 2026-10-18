# exp_4(d) — P4-E9 FinTrust seven-dimension evaluation (generation complete)

Status: **generation complete** on the official sample subsets with the frozen E5 adapter.
Fairness dimensions are **scored** (official rule code, no judge). The other ten dimension
runs are **judge-pending**: generations are stored in the official output schema and are ready
to be scored by the official `postprocess_response.py` (gpt-4.1-mini) via
`scripts/score_fintrust.py` once an OpenAI key is available.

Date: 2026-10-13. Model: `Qwen/Qwen3-1.7B` + frozen E5 QLoRA adapter (4-bit nf4).
Gate: `phase4_freeze.json` = FROZEN; run required `TRUSTFIN4_PROTECTED=1`; dataset-registry and
FinTrust-registry hashes verified. Pre-registration written before generation:
`artifacts/phase4/manifests/phase4_e9_prereg.json`.

## Scope

User-selected scope: **official sample subsets** shipped in the FinTrust eval-code repo
(~100 records/dimension). Judge decision: **generate now, score later** (store judge-ready
outputs; fairness rule-scored now). Total = **3,520 generations** across 12 dimension runs
covering the 7 registry dimensions. Wall-clock on an RTX 4050 6 GB: ~1.5 h total (resumable,
budgeted).

## Results

| dimension | generations | scoring status | notes |
|---|---|---|---|
| privacy | 100 | pending judge | model leaks PII directly (e.g. SSN answers) |
| robustness | 120 (20×6) | pending judge | all 120 answers in FinQA PROGRAM/ANSWER form |
| transparency | 400 (100×4) | pending judge | 400/400 emitted a parseable target A/B JSON |
| knowledge_discovery | 100 | pending judge | 100/100 PROGRAM/ANSWER form |
| fairness_personal | 2,100 (100×21) | **rule-scored** | consistency 0.67 / inconsistency 0.05 |
| fairness_industry | 100 | **rule-scored** | mean pairwise |Cohen's d| = 0.410 |
| safety_deep_inception | 100 | pending judge | mostly non-compliant (FinQA form) |
| safety_in_context | 100 | pending judge | 100/100 complied with harmful prompts |
| safety_language_barrier | 100 | pending judge | translation step deferred (judge-later) |
| safety_plain_question | 100 | pending judge | mixed FinQA form / terse answers |
| trustfulness_hallucination_certainty | 100 | pending judge | 100/100 PROGRAM/ANSWER form |
| trustfulness_number_informativeness | 100 | pending judge | 100/100 PROGRAM/ANSWER form |

### Fairness (official rule code, no judge)

- `fairness_personal` — `full_prompt` judgement vs 20 counterfactual variants:
  - full==label & all fakes==full: **67**
  - full==label & some fakes≠full: **5**
  - full≠label & all fakes==full: **27**
  - full≠label & some fakes≠full: **1**
  - → consistency_rate **0.67**, inconsistency_rate **0.05**, n=100.
- `fairness_industry` — buy/sell by GICS sector (all 100 answers were "buy") and mean pairwise
  |Cohen's d| on the local first-token probability of the token `buy` = **0.410** (all 100
  records contribute; official code would include only records whose first token was literally
  `buy`). Logprob coverage spans all 11 sectors.

### Descriptive observations (our own counts, **not** official scores)

- The frozen adapter is strongly in-distribution to FinQA: for every prompt that is not a
  strict JSON-format request it answers with a `PROGRAM: … ANSWER: …` continuation. This is why
  robustness / KD / HC / NI / deep-inception / language-barrier runs look "formatted" rather
  than instruction-following.
- Privacy: the model answers PII questions directly (no refusal) — a genuine privacy failure
  signal, but the *official* privacy score (answered/refused) awaits the judge.
- Transparency: the explicit JSON instruction is followed 400/400 (target ∈ {A,B}), so this
  dimension is likely to score well once judged.
- In-context attack: **every** answer complied with the harmful in-context request — a genuine
  safety concern that the judge will quantify.

## How to finish scoring (judge-later)

```
# validate wiring without a key
python3 scripts/score_fintrust.py --dry-run
# run the official postprocessors unchanged in a sandbox (needs openai + OPENAI_API_KEY)
OPENAI_API_KEY=... python3 scripts/score_fintrust.py --dims privacy robustness transparency \
    knowledge_discovery safety_deep_inception safety_in_context safety_language_barrier \
    safety_plain_question trustfulness_hallucination_certainty trustfulness_number_informativeness
```

The scorer copies each `P4-E9_{dim}.json` to the exact `file_path` the official script expects,
places the unmodified official script in a sandbox, runs it, and collects the produced summary
JSON into `artifacts/phase4/scoring/`. Language-barrier English translation
(`translated_to_english`, official: gpt-4.1-mini) is part of that official pipeline.

## Declared deviations (see prereg)

Local 4-bit E5 generation (not the hosted API); middle-truncation at 3072 tokens for over-long
prompts (robustness, KD, parts of HC/NI); local `max_new` caps; `enable_thinking=False`; seeded
robustness irrelevant-context draws; deferred language-barrier translation; fairness rule-scored
now.

## Honest limitations

- Sample subsets (~100/dim), not the full benchmark (privacy/transparency/KD/etc. are 1000+).
- 10 of 12 runs are not yet scored; only the two fairness dimensions have numbers.
- The adapter's FinQA format-lock means several dimensions measure *format leakage* more than
  the intended capability; this is reported as an observation, not folded into a headline.
- No FinTrust record is used for training or calibration (registry constraint).

## Paper-safe wording

> "We ran the frozen model across the seven FinTrust dimensions on the official sample subsets
> (3,520 generations); both fairness dimensions are scored (personal consistency 0.67;
> industry mean |Cohen's d| 0.41), and the remaining dimensions are stored in the official
> schema pending the official GPT-4.1-mini judge. We therefore continue to claim
> selective-reliability on FinQA and a descriptive FiQA transfer, not a broad FinTrust
> trustworthiness benefit."
