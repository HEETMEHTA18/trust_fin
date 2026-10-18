# Novelty check — EVG-CAS vs the 2024–2026 selective-abstention literature

Status: audited 2026-10-12 against web search (livecrawl). Sources are title/venue/arXiv or
ACL id; **author metadata for the bib entries must be completed from the downloaded PDFs**
before submission (see `paper/TrustFin_Conference_Paper.tex` bibitems added this pass).

## The claimed mechanism

For a **1.7B domain-adapted financial SLM**, on programmatic numerical reasoning (FinQA):

1. Sample *n* candidate programs; keep those that run on the **official evaluator**.
2. Use **agreement across executed answers** `A(q)` as a per-record trust feature.
3. Platt-calibrate a coverage-targeted threshold **frozen on validation only**.
4. Transfer to a **never-opened protected test** and report answered-error under a
   pre-registered protocol (freeze manifest, split-sealed).
5. Compare against confidence- and best-of-n abstention **at matched coverage** (bootstrap CIs).

The distinctiveness is the **combination**: evaluator-executed agreement as a *decision
signal* (not a vote), calibration to a coverage meaning, a frozen threshold, and matched
coverage comparison — together on a small financial model.

## Located adjacent work (reasoned gaps)

| Work | Year/Venue | Mechanism | Gap vs ours |
|---|---|---|---|
| **Second Guess** (arXiv:2605.25394) | 2026 | two-pass prompt perturbation; answer stability abstention for 2–8B SLMs, MCQ | MCQ only; no program execution; no evaluator; no calibration threshold or split-sealed protocol |
| **CISC** — Confidence-Informed Self-Consistency (2025.findings-acl.1030) | Findings ACL 2025 | confidence-weighted majority vote; within-question discrimination | voting (accuracy) framing, general domains; no abstention decision rule; **agrees with us that calibrated confidence is least useful for selection** (supports our finding) |
| **SConU** — Selective Conformal Uncertainty | ACL 2025 (long) | conformal selective prediction with guarantees | distribution-free guarantees machinery; not execution-verified programs; not a domain-adapted financial SLM |
| **LEC** — Linear Expectation Constraints (arXiv:2512.01556) | 2025 | selection-conditioned risk control under constraints | theory framing; no executed-program agreement feature |
| **ASPIRE** (arXiv:2310.11689) | 2023/2024 | adaptation + self-evaluation for selective prediction of LLMs | selective prediction framing for base LLM quality, not calibrated abstention on candidate-population features of a financial SLM |
| **FinMAN** — David vs. Goliath (2025.findings-emnlp.225) | Findings EMNLP 2025 | multi-agent financial QA with verification, 8B, code execution | multi-agent accuracy/cost; **no per-record calibrated abstention**; no frozen-threshold transfer |
| **GBFR** — Graph-Bounded Financial Reasoning | 2026 | financial abstention via metric knowledge graph; safe abstention on unanswerable queries | prompt-based abstention for KG-bounded queries, not per-record calibrated threshold with coverage targets on a fixed benchmark |
| **FinChain** (arXiv:2506.02515) | 2025 | verifiable CoT benchmark in finance | benchmark for verification, not a model abstention mechanism |

## Nearest claims to ours (no overlap in the mechanism)

- **How to make a small financial model "know when it is wrong" per record** under a
  pre-registered, calibration-grounded, coverage-targeted abstention rule — **not found**.
- **Agreement of answers across execution-verified sampled programs** used as the *decision*
  feature (vs self-consistency's vote) — **not found** in any located source.
- **Matched-coverage comparison that agreement > confidence > best-of-n** for a budget
  financial model — **not found**.

## Distinguishing sentences to keep in the paper

"Do not claim 'first to abstain in finance' (FinMAN/GBFR/FinTrust exist) and do not claim
'first self-consistency in finance'. Claim: *the first calibration-grounded, coverage-targeted
abstention for a 1.7B financial SLM whose decision feature is agreement among
execution-verified candidate programs, validated on a frozen pre-registered protocol with
matched-coverage baselines.*"

## Follow-ups flagged

- Download PDFs (`2025.findings-acl.1030`, `2025.findings-emnlp.225`, arXiv:2605.25394,
  2512.01556, 2310.11689, 2506.02515), fill author metadata, and re-verify quoted claims
  before submission.
- CISC's "most-calibrated confidence least useful for selection" is a strength to cite
  explicitly (independent corroboration of our Fig. 1-type finding), not a risk.
- GBFR/FinMAN should be cited on the finance abstention object side.