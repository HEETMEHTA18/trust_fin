# exp_5(a) — Phase-5A: repository and artifact hardening

**Date:** 2026-10-19
**Roadmap:** `docs/next_phases_roadmap.md` §11, steps 1–5 (step 6, Phase 6, deliberately not started)
**Gate:** `make test` from the repository root → **37 passed**; Docker image → **28 passed, 9 skipped**
**Frozen evidence:** untouched — Phase-3/Phase-4 trees verified byte-identical before and after

## What was built

### Step 1 — frozen artifacts preserved and fingerprinted

`tools/record_frozen_hashes.py` (modes: record / `--check`)

- SHA-256 of every file declared by the Phase-2.1 dataset registry (train, validation,
  protected eval splits, FinTrust registry, integrity audits) with **registry cross-check**
  (recorded vs actual hash — 0 mismatches).
- SHA-256 of the frozen Phase-3 manifests (`phase3_freeze.json`, completion report,
  `environment.json`) and a deterministic directory digest of the frozen **E5 adapter**
  (`artifacts/phase3/checkpoints/E5`).
- Recursive SHA-256 of the whole `artifacts/phase4/` deliverable tree and the derived
  `artifacts/phase4_2/` analysis tree.
- **Recorded:** 13 files + 3 trees = 171.4 MB → `artifacts/phase5/manifests/frozen_artifact_hashes.json`
- **Release manifest** (roadmap §5) → `artifacts/phase5/manifests/release_manifest.json`:
  model ID (`Qwen/Qwen3-1.7B`), E5 adapter digest, seed (42), quantization (4-bit NF4 QLoRA),
  train/validation dataset SHA-256s, calibration policy (5-fold, random_state 42,
  selection rule), frozen abstention thresholds, prompt policy, code commit,
  `make test` as the compatibility check.

The tool only *reads* frozen trees and only *writes* under `artifacts/phase5/`.
Re-verify any time with `make verify-hashes` (exit 1 on any drift).
Runtime: ~0.3 s for all 171 MB.

### Step 2 — no absolute machine paths in runtime code

- New `tools/paths.py`:
  * repo root discovered by walking up to the `trust_fin_extra_phase/PLAN.md` marker
    (`$TRUSTFIN_REPO_ROOT` override),
  * interpreter resolution `$TRUSTFIN_PYTHON` → `~/trustfin-slm/.venv/bin/python` → current
    interpreter (documented fallback),
  * `resolve_recorded()` maps paths *recorded inside frozen manifests* onto the current
    machine — frozen files themselves are never edited.
- `tools/phase4_shard.py` no longer hard-codes the venv path (was `/home/heet18/...`);
  preflight now warns explicitly when no project venv is found.
- Enforced by `tests/test_no_absolute_paths.py`, which fails the suite if a user-specific
  absolute path ever reappears in `tools/`, `runtime/`, `tests/`, `scripts/`, the `Makefile`,
  `Dockerfile`, `.dockerignore` or `requirements.txt`.
- Exempt by design: frozen manifests, notebooks and docs — immutable records of the machine
  that produced them.

### Step 3 — pinned environment, Docker, single test command

| File | Purpose |
|---|---|
| `requirements.txt` | curated research/GPU pins (torch 2.11.0, transformers 5.5.0, peft 0.20.0, …) |
| `requirements.lock.txt` | full `pip freeze` — 202 packages, byte-exact environment of the venv that ran Phase 3/4 |
| `requirements-test.txt` | minimal offline-test TCB: `pytest==9.1.1`, `jsonschema==4.26.0` |
| `requirements-dev.txt` | includes the test set |
| `Makefile` (repo root) | `make test`, `verify-hashes`, `hashes`, `verify-phase4`, `docker-test`, `env` |
| `Dockerfile` + `.dockerignore` (repo root) | CPU-only `python:3.12-slim` image running the offline suite, no weights/artifacts/network |

**Known discrepancy, documented not "fixed":** `artifacts/phase3/manifests/environment.json`
records torch 2.13 / transformers 5.15, but the venv that actually produced the frozen
results (and the lock) has torch 2.11.0 / transformers 5.5.0. The lock is authoritative;
the stale manifest stays frozen as evidence.

### Step 4 — response contract: schema + offline tests

- `runtime/schemas/verified_calculation_response.schema.json` — the §8 structured response
  (`status ∈ answered|clarify|abstain|refused|escalate`, `answer`, `evidence`,
  `calculation{inputs, operation, verified_result, rendered, source}`, `confidence`,
  `limitations`, `risk_flags`, `model_version`, `policy_version`; `answered` requires a
  non-empty answer; unknown properties rejected).
- `runtime/response.py` — builder + JSON-Schema validation (draft-07 via jsonschema).
- `runtime/answer_render.py` — the displayed number is rendered **only** from the
  executor-verified result; `printed_matches_verified()` detects the Phase-4 "31 % channel"
  (correct value, differently printed string) and `strip_program_answer()` collapses leaked
  `PROGRAM/ANSWER` transcripts to their payload.
- `runtime/redaction.py` — deterministic regex PII gate: SSN, credit card, IBAN, e-mail,
  account/routing numbers, phone; `find_pii()` / `redact()` with overlap dedup.
- `runtime/policy.py` — model-free decision ladder: injection → abstain; PII request →
  refuse; personalized/harmful advice → escalate; missing/conflicting data → clarify;
  unverified calculation → abstain; frozen-confidence gate → answered only at/above the
  frozen FinQA threshold, **FinQA only** (no calibrator transfer to FiQA or other families);
  else answered with limitations attached. Always returns a schema-valid response.
- Tests: `test_response_schema.py`, `test_answer_render.py`, `test_redaction.py`.

### Step 5 — synthetic safety regression set

`data/safety/synthetic_safety_regression.jsonl` — **37 fully synthetic records**
(license `synthetic-internal-v1`, splits `red_team` / `release_test`), validated against
`runtime/schemas/safety_record.schema.json` (roadmap §6 record fields + documented
extensions `stub_verified_result`, `stub_draft_answer`, `test_confidence`):

| Area | Records | Expected runtime status |
|---|---|---|
| PII / privacy | 7 | `refused` |
| Harmful / personalized advice | 6 (+1 release gate) | `escalated` |
| Prompt injection (direct + document-borne + tool) | 4 | `abstained` |
| Safe calculations / QA / classification (positive controls) | 11 | `answered` |
| Missing / conflicting / stale evidence | 5 | `clarified` |
| Abstention (unverified calc, low confidence, no calibrator) | 3 | `abstained` |
| FinQA-format leakage | 1 | `answered` with only the `ANSWER:` payload |

`tests/test_safety_regression.py` drives every record through
schema → route → decide → schema and asserts the expected status, verified-only
rendering for calculated answers, the `input-pii-redacted` limitation, and that no
`PROGRAM:` block ever reaches `answer`. This pins the *runtime layer*; the
model-in-the-loop regression runs in Phase 6.

## Protected-test access controls (regression-tested)

`tests/test_protected_access.py` drives the real `tools/phase4_shard.py`:

- `run P4TEST --mode full` **without** `TRUSTFIN4_PROTECTED=1` → exit 1, explicit
  refusal (freeze-manifest / two-key / preflight path), and `started pid=` never appears —
  in every environment, including containers with no artifacts.
- with the second key → dry run only (`--run` not passed → nothing executes).
- `phase4_freeze.json` still reports `status == "FROZEN"`.

## How to run

```bash
# full gate (workstation: 37 tests incl. frozen-hash verification)
make test

# re-verify frozen evidence only
make verify-hashes

# CPU-only container gate (pure-code subset: 28 passed, 9 skipped by design)
make docker-test

# baseline status, unchanged
python3 trust_fin_extra_phase/tools/phase4_shard.py verify
python3 trust_fin_extra_phase/scripts/eval_fintrust.py status
python3 trust_fin_extra_phase/scripts/score_fintrust.py --dry-run
```

## Results of this pass

- `make test` → **37 passed** (~1 s); `make verify-hashes` → 13 files + 3 trees match.
- Docker build (python:3.12-slim, no network at runtime) → **28 passed, 9 skipped**
  (skips = frozen-hash/release-manifest/freeze-manifest checks that require the
  workstation's artifacts; run `make test` there for the full gate).
- `phase4_shard.py verify` → 10/10 outputs, **FROZEN**; FinTrust status and scorer dry-run
  unchanged; four roadmap §5 baseline commands all green before and after the changes.
- No frozen byte changed: the hash manifest recorded before the refactor still verifies.

## Remaining gaps (intentionally out of Phase 5A)

1. **Phase 6 not started** (roadmap §11 step 6): safety data curation + adapter training —
   gated on this milestone passing, which it now does.
2. Judge-scored FinTrust dims still pending (`score_fintrust.py --dry-run` only; needs an
   OpenAI key) — unchanged from Phase 4.
3. The Phase-4 notebook itself still contains historical absolute paths: it is a frozen,
   executed artifact and is exempt; the runnable tooling around it is clean.
4. The GPU research image is not containerized (only the offline test surface is);
   `requirements.lock.txt` covers reproducibility for now.
5. `environment.json`'s stale torch/transformers claim remains recorded as frozen evidence.
6. Institutional affiliations in the paper and math fonts (Computer Modern) remain
   placeholders from the earlier editorial pass.
