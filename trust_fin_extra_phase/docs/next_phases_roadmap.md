# TrustFin-SLM: roadmap after Phase 4

**Status:** planning document for the next implementation phases — **Phase 5A implemented
2026-10-19, steps 1–5 complete and green (37 offline tests); see `docs/exp_5(a).md`.
Step 6 (Phase 6) not started.**  
**Baseline:** Phase-3 E5 + Phase-4 EVG-CAS, frozen and protected-test evaluated  
**Scope:** trustworthy, usable financial-task SLM; not autonomous investment advice

## 1. The honest starting point

Phase 4 is complete, but it does **not** yet demonstrate a generally trustworthy financial
assistant. It demonstrates a narrower result:

- E5 is `Qwen/Qwen3-1.7B` with the balanced joint QLoRA adapter.
- FinQA protected-test execution accuracy is about `0.656`.
- FinQA selective prediction reduces answered error by about `31%` at approximately `80%`
  coverage when the model is allowed to abstain.
- The current model still has known safety failures: direct PII responses and harmful
  in-context prompt compliance.
- The model is strongly locked to the FinQA `PROGRAM/ANSWER` format on unrelated tasks.
- FiQA abstention does not transfer usefully, and most FinTrust dimensions are still
  judge-pending or sample-subset-only.

Therefore, do **not** put the current adapter directly in front of financial clients. The next
work must add a safety/alignment layer, deterministic tools, provenance, abstention, human
escalation, and production controls. A single SLM should not be expected to memorize current
market facts or make unsupervised investment, tax, legal, credit, or regulatory decisions.

## 2. Recommended phase sequence

| Phase | Name | User-visible result | Release gate |
|---|---|---|---|
| 5 | Reproducible release foundation | A clean-machine rebuild and offline test command | No release from an unpinned environment |
| 6 | Financial safety alignment | Refusal, privacy, escalation, and instruction-following behavior | No critical PII or harmful-advice failures on the red-team gate |
| 7 | Trust and robustness validation | Evidence across tasks, seeds, clients, and OOD conditions | Metrics meet task-specific thresholds on untouched evaluation data |
| 8 | Verified financial runtime | Tool-grounded answers, citations, schemas, and safe abstention | Every high-risk answer is verified or escalated |
| 9 | Shadow deployment and governance | Monitoring, audit, rollback, and controlled pilot | Pilot passes incident and drift gates before expansion |

The phases are deliberately sequential. Phase 6 should not be trained on protected Phase-4
test records, and Phase 7 must not select a model using those records. Create new development,
validation, and release-test partitions for the new safety task rather than reopening frozen
test artifacts for tuning.

## 3. Target product boundary

Start with low- and medium-risk assistive tasks where the answer can be checked:

1. Annual-report and filing question answering with cited source passages.
2. Ratio, growth, margin, variance, and cash-flow calculations using a calculator tool.
3. Structured extraction from financial documents.
4. Financial sentiment or stance classification with calibrated probabilities.
5. Drafting summaries, explanations, and report notes from supplied documents.
6. Internal analyst workflow assistance with a human approving the final output.

Keep these out of autonomous mode:

- personalized buy/sell/hold recommendations;
- portfolio, credit, tax, legal, or regulatory decisions;
- actions that move money, place trades, approve credit, or contact a client;
- answers based on stale or unverified market data;
- disclosure of customer or employee PII.

For those requests the runtime should explain the boundary, provide a safe general
alternative where appropriate, and route the case to an authorized human.

## 4. The architecture to build

Use the SLM as one component, not as the whole financial system:

```text
client request
   -> tenant/authentication and input redaction
   -> task/risk router
   -> policy and prompt-injection checks
   -> approved retrieval and/or deterministic financial tool
   -> SLM adapter for drafting or reasoning
   -> executor/schema/citation verification
   -> calibrated confidence decision
      -> answer with evidence
      -> ask for clarification
      -> abstain and escalate to a human
   -> immutable audit event and monitoring
```

Important design rules:

- Current facts come from versioned retrieval or approved APIs, not from model memory.
- Arithmetic comes from a deterministic calculator or domain executor; never trust a printed
  number when the calculation can be recomputed.
- The final displayed numeric value must be rendered from the verified tool result. This
  directly addresses the Phase-4 finding that a correct execution can still be followed by a
  wrong printed answer string.
- Each answer stores source identifiers, source timestamps, calculation inputs, tool versions,
  model/adapter versions, and the decision to answer, clarify, abstain, or escalate.
- Tenant-specific knowledge belongs in isolated retrieval indexes and policy configuration,
  not in a shared model fine-tune containing client data.

## 5. Phase 5 — reproducible release foundation

Before more training, make the result rebuildable by another person on a clean Linux machine.
The current repository has absolute paths, conflicting recorded dependency versions, no
project-local lockfile, no Dockerfile, and no single offline test command. Fix these first.

### Deliverables

- Pin Python and CUDA-compatible dependency versions in a project-local lockfile.
- Add a Dockerfile that installs system dependencies and runs the offline checks.
- Replace hard-coded `/home/heet18/...` paths with repository-root discovery and environment
  variables with documented defaults.
- Add an artifact manifest containing model ID, adapter revision, dataset hashes, code commit,
  tokenizer revision, seed, quantization settings, and evaluation configuration.
- Add `make test` (or an equivalent single command) that works without network access.
- Add tests for JSON schemas, artifact hashes, answer rendering, redaction, abstention, and
  protected-test access controls.
- Keep Phase-4 frozen artifacts read-only; write new work under `artifacts/phase5/` onward.

### First commands to run tomorrow

Run from the repository root, not from the `trust_fin_extra_phase` folder:

```bash
cd "/home/heet18/Downloads/Research Papers"
git status --short
python3 trust_fin_extra_phase/tools/phase4_shard.py verify
python3 trust_fin_extra_phase/scripts/eval_fintrust.py status
python3 trust_fin_extra_phase/scripts/score_fintrust.py --dry-run
```

The last two commands are status/wiring checks, not a deployment approval. Do not use a
protected-test score to tune the next model. After Phase 5 is implemented, the required clean
machine command becomes:

```bash
make test
```

## 6. Phase 6 — safety alignment and policy routing

Do not begin by blindly mixing safety examples into the existing FinQA/FiQA corpus. Define a
task and risk contract first, then create a separate safety dataset with provenance and a
reviewed policy label.

### Training data families

Use synthetic or properly licensed/de-identified examples for:

- **Safe answer:** document-grounded questions with an evidence span and timestamp.
- **Calculation:** input values, operation plan, tool result, and rendered result.
- **Uncertainty:** missing data, ambiguous wording, conflicting documents, and stale data.
- **Privacy:** refuse or redact SSNs, bank details, account numbers, credentials, and private
  client records; do not reproduce PII merely because it appears in the prompt.
- **Harmful financial requests:** refuse individualized high-risk advice and route to an
  authorized professional; still answer safe educational questions.
- **Prompt injection:** treat retrieved documents and quoted text as untrusted data, not as
  instructions.
- **Instruction following:** honor the requested output schema without inheriting FinQA's
  `PROGRAM/ANSWER` format on unrelated tasks.
- **Fairness and language:** equivalent financial questions across names, sectors, regions,
  languages, and dialects should receive equivalent treatment.

Every training/evaluation record should carry at least:

```json
{
  "id": "stable-example-id",
  "task_family": "calculation|document_qa|classification|privacy|safety|escalation",
  "risk_level": "low|medium|high",
  "messages": [],
  "expected_action": "answer|clarify|abstain|refuse|escalate",
  "required_evidence": [],
  "pii_status": "none|synthetic|redacted",
  "source_license": "documented-source",
  "split": "train|validation|red_team|release_test"
}
```

### Training order

1. Continue to use E5 as the reproducible Phase-4 baseline.
2. Train a safety/policy adapter on reviewed examples while preserving the domain task set.
3. Add preference pairs only after policy labels are stable: safe grounded answer versus
   hallucinated answer, correct refusal versus unsafe compliance, and verified calculation
   versus unsupported number.
4. Use a held-out safety validation set for early stopping and threshold selection.
5. Keep a completely untouched release-test set for the final gate.
6. Compare the adapted model to E5 and the base model; never assume safety tuning improved
   financial accuracy.

The result should be a multi-head or routed system conceptually, even if it is implemented
with one SLM: task classification, policy decision, answer generation, and verification must
be separately measurable. A smaller model that reliably abstains is preferable to a larger
model that confidently fabricates financial facts.

## 7. Phase 7 — trust, robustness, and generalization

Evaluate the actual product task matrix, not only FinQA. Use at least three random seeds for
any model-selection claim, and report confidence intervals. Keep benchmark roles explicit:

| Evaluation family | Purpose | Tuning allowed? |
|---|---|---|
| training | parameter fitting | yes |
| development validation | model/threshold selection | yes |
| internal red team | bug discovery and remediation | only via a new versioned split |
| release test | final claim | no |
| live shadow traffic | monitoring | no automatic tuning |

Required evaluation slices:

- numerical accuracy and exact execution correctness;
- printed-answer versus executor-result consistency;
- document citation correctness and source freshness;
- abstention risk/coverage per task family;
- calibration (Brier, ECE, NLL, reliability plots);
- missing, contradictory, stale, and adversarial inputs;
- prompt-injection and tool-confusion attacks;
- PII extraction and memorization probes;
- harmful financial-advice compliance/refusal;
- fairness across relevant client and language slices;
- latency, memory, throughput, and cost at the intended concurrency.

The current Phase-4 FinQA calibration must not be copied to FiQA or other tasks without a new
validation study. Calibrators and thresholds are task- and deployment-specific.

## 8. Phase 8 — verified inference runtime

The first usable interface should return a structured response, not unrestricted text:

```json
{
  "status": "answered|clarify|abstain|refused|escalate",
  "answer": "...",
  "evidence": [{"source_id": "...", "quote": "...", "as_of": "..."}],
  "calculation": {"inputs": [], "operation": "...", "verified_result": null},
  "confidence": null,
  "limitations": [],
  "model_version": "...",
  "policy_version": "..."
}
```

Runtime behavior:

- Reject or redact sensitive inputs before they reach the model where possible.
- Route high-risk tasks before generation.
- Require evidence for factual answers and a tool trace for calculations.
- Validate the model output against a strict schema; malformed output is an abstention, not a
  best-effort answer.
- Use the Phase-4 selective approach only where validation proves it works. For FiQA-like
  tasks, collect a new calibration set before enabling automatic thresholds.
- Present an explicit “not financial advice” or escalation message only where the policy calls
  for it; do not use a disclaimer as a substitute for verification.
- Log enough metadata for audit while excluding raw secrets and unnecessary PII.

## 9. Phase 9 — shadow deployment and governance

Start with a shadow mode: the SLM produces an internal result while the existing human or
system remains authoritative. Measure real task mix, abstentions, unsafe generations, stale
retrieval, latency, and disagreement with reviewers before users see the output.

Minimum operational controls:

- immutable model, adapter, prompt, tool, policy, and retrieval-index versions;
- tenant isolation and least-privilege access;
- encrypted logs with a documented retention period and redaction policy;
- rate limits and abuse monitoring;
- drift and calibration monitoring by task family;
- incident severity levels, owner, response time, and rollback procedure;
- approval record for every model/policy release;
- a kill switch that disables autonomous answers while preserving human workflows.

Expand from shadow to a small pilot only when the release-test gate passes. Expand to more
clients by validating the new client's documents, policies, languages, and risk profile; do
not infer generalization from a high aggregate benchmark score.

## 10. Suggested release gates

These are engineering gates to agree with compliance and domain experts, not a claim of
regulatory certification:

1. **Data:** every training/evaluation record has a provenance/license decision; no unapproved
   customer data is in shared training artifacts.
2. **Privacy:** no unresolved critical PII leak in the release red team; redaction and refusal
   tests are regression-tested.
3. **Safety:** harmful high-risk requests are refused or escalated; safe educational requests
   remain answerable; prompt-injection tests cannot authorize an unsafe tool action.
4. **Numerical correctness:** accepted calculations are executor-verified and the displayed
   number is generated from the verified result.
5. **Grounding:** factual answers include valid evidence and freshness metadata; unsupported
   answers abstain or escalate.
6. **Calibration:** each enabled task has a validation-backed risk/coverage policy; no threshold
   is chosen from release-test outcomes.
7. **Fairness:** no unexplained material disparity on approved client and language slices; all
   exceptions have a documented owner and remediation plan.
8. **Operations:** reproducible build, offline tests, audit logs, monitoring, rollback, and a
   human escalation path are all exercised before pilot.

## 11. Immediate next milestone

The next executable milestone is **Phase 5A: repository and artifact hardening**, not another
large training run. Finish it in this order:

1. ✅ Preserve the frozen Phase-3/Phase-4 artifacts and record their hashes
   (`tools/record_frozen_hashes.py`; `artifacts/phase5/manifests/frozen_artifact_hashes.json`,
   `release_manifest.json`; `make verify-hashes`).
2. ✅ Remove runtime dependence on absolute machine paths
   (`tools/paths.py` discovery + `$TRUSTFIN_REPO_ROOT` / `$TRUSTFIN_PYTHON`;
   enforced by `tests/test_no_absolute_paths.py`).
3. ✅ Pin the environment and add Docker plus `make test`
   (`requirements{.txt,-dev.txt,-test.txt}` + `requirements.lock.txt`, `Dockerfile`,
   `.dockerignore`, root `Makefile`).
4. ✅ Add schemas and offline tests for a single verified-calculation response
   (`runtime/schemas/verified_calculation_response.schema.json`, `runtime/{response,
   answer_render,redaction,policy}.py`).
5. ✅ Add a small synthetic safety regression set covering PII, harmful advice, prompt
   injection, abstention, and the FinQA-format leakage
   (`data/safety/synthetic_safety_regression.jsonl`, 37 records,
   `tests/test_safety_regression.py`).
6. ☐ Only after those checks pass, start Phase 6 safety data curation and adapter training.

This creates a usable foundation without invalidating the published Phase-4 experiment. It
also gives a clear answer to “when is it ready for clients?”: only after the runtime and all
release gates pass for the specific task families and risk levels being offered.
