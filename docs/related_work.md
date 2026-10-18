# Related Work — literature map and taxonomy

**Scope.** Narrative review that positions the Phase-4 method (EVG-CAS: execution-verified
generation + agreement-calibrated abstention for a financial SLM). Full metadata lives in
`docs/trustfin_references.bib`; the per-paper tracking matrix is in
`docs/literature_reading_matrix.md`. BibTeX tags (`[finllm]`, `[verif]`, …) are the
taxonomy families below.

---

## 1. Taxonomy of the literature

Six families cover the space this paper sits in. For each: what the family is, its main
lines, and the specific gap it leaves that EVG-CAS addresses.

### Family A — Small language models & efficient adaptation [smlm]
SLMs (≈1–3B) are surveyed as a deliberate compute/performance trade-off
[Subramanian et al. 2025; Abdin et al. 2024; Zhang et al. 2024]. Efficient adaptation via
LoRA/QLoRA makes a frozen pre-trained backbone adaptable on consumer hardware
[Hu et al. 2022; Dettmers et al. 2023]. **Gap:** the SLM literature optimises accuracy and
efficiency; *trustworthiness* (knowing when the model is wrong) is named as an open,
under-explored area by the KDD'25 SLM survey, and none of these works build a
transferable abstention mechanism for the models they promote.

### Family B — Financial LLMs and domain adaptation [finllm]
BloombergGPT trains a finance LM from scratch on proprietary data [Wu et al. 2023];
FinGPT [Yang et al. 2023], InvestLM [Yang et al. 2023] and FinBERT [Huang et al. 2020]
adapt open backbones with domain instruction tuning; Kronos [Shi et al. 2025] targets
market time-series. Phogat et al. (2024) — **the closest comparable work** — show that
*smaller* models fine-tuned for financial QA are competitive under tight compute, exactly
our regime. **Gap:** financial adaptation work reports accuracy and sometimes expert
judgement, but not calibrated confidence or abstention on a *protected* test set; no
pre-registered decision policy.

### Family C — Financial reasoning, QA and evaluation [finqa]
FinQA [Chen et al. 2021], ConvFinQA [Chen et al. 2022], TAT-QA [Zhu et al. 2021] define
programmatic numerical reasoning over financial documents, with **execution** (running the
program) as the ground truth for the *answer*, separate from the generated program text.
This is what makes "execution-verified" sampling well-defined for our task.
FinBen [Xie et al. 2024] and FinTrust [Hu et al. 2025] are evaluation benchmarks: FinBen
spans 42 datasets/24 tasks; FinTrust is the trustworthiness diagnostic, and explicitly
lists "appropriately abstain when confidence is low" as part of its *trustfulness*
dimension. **Gap:** FinTrust *measures* whether models abstain (an evaluator's tool, run on
GPT-4-scale models) but provides no mechanism a 1.7B model can actually use. We supply the
mechanism.

### Family D — Consistency and verification for reasoning [verif]
Self-consistency samples k reasoning paths and majority-votes [Wang et al. 2023].
Step-by-step and process supervision verify intermediate steps with learned reward models
[Lightman et al. 2024; Wang et al. 2024]. Execution-guided program synthesis uses the
runtime as a hard filter during decoding [Chen et al. 2019]. **Gap:** these methods are
framed and evaluated as *accuracy* gains, moderate via arefit/tuned verifiers (e.g.,
Math-Shepherd's PRM), and almost never report negative results when the gain is zero. They
also discard the information that *agreement itself* is a confidence signal. EVG-CAS keeps
the sampling cost but spends it on a **decision signal**, not an accuracy vote — and
reports the accuracy outcome honestly (a tie).

### Family E — Selective prediction and risk-coverage [selcal]
The classic theory: a classifier may abstain; the goal is to hit a coverage target with
minimal selective risk, evaluated by risk-coverage/accuracy-coverage curves
[El-Yaniv & Wiener 2010]. Deep variants add trainable reject options (SelectiveNet;
softmax-margin reject) [Geifman & El-Yaniv 2017, 2019]. For LLMs, confidence is taken from
log-probs or from verbalised probabilities [Lin et al. 2022]. **Gap:** the classic
literature operates on classifier logits; the LLM version (Lin et al.) is answered at the
*generation* level, not driven by **population-level agreement across the model's own
samples**, and none of it is evaluated with a pre-registered threshold on a true held-out
test that was never touched during selection.

### Family F — Calibration and trustworthy/explainable finance [xai]
Platt scaling and isotonic regression turn scores into probabilities [Platt 1999;
Zadrozny & Elkan 2002]; Guo et al. (2017) popularise temperature scaling and ECE for
neural nets. In finance, responsible-AI guidance frames explainability/trust as a
regulatory requirement [Fritz-Morgenthal et al. 2022]. **Gap:** calibration papers that
use ECE/Brier alone can miss *ranking* (calibrators are monotone); finance guidance is
aspirational. Our protocol measures Brier/ECE (probability quality), e-AURC (ranking),
**and** risk@coverage transfer (the operational quantity regulators care about).

---

## 2. Positioning: the research gap

The map above converges on one unfilled cell:

> **Accuracy-oriented verification (self-consistency, process/outcome reward models) has
> never been connected to selective prediction for a *domain-adapted SLM*, evaluated with
> a pre-registered, calibration-grounded abstention threshold on a fully protected test
> set. Benchmarks measure abstention (FinTrust) but do not build it; selective-prediction
> theory builds it but on classifier logits, not on sample-agreement signals; financial
> SLM work never measures it.**

EVG-CAS fills exactly that cell: it adds **execution-verified sampling** to a frozen,
QLoRA-adapted SLM, derives trust features from the *sample population* (agreement among
executed answers, executability), calibrates them with out-of-fold Platt scaling, and
freezes an abstention threshold on validation data before the protected test is opened.
The protocol was fixated in `PLAN.md` §4 before any protected-test access
(guardrail-based, two-key gate).

## 3. How each family enters the paper

| Paper section | Families cited |
|---|---|
| §1 Introduction | [smlm], [finqa], [verif], [selcal] |
| §2.1 Financial SLMs | [finllm], [smlm] |
| §2.2 Programmatic reasoning (why execution is well-defined) | [finqa] |
| §2.3 Verification & consistency | [verif] |
| §2.4 Selective prediction & calibration | [selcal], [xai] |
| §2.5 Benchmarks & diagnosis | [finqa] (FinBen, FinTrust) |
| §5 Protocol (pre-registration, protected sets) | [verif] (emphatic honesty), [xai] |

## 4. Citation-budget check (guide §16 & §19)

- 29 entries in `trustfin_references.bib`; **15 are core** (rows checked "CORE" in the
  reading matrix), exceeding the guide's 10–15 minimum.
- Every factual claim in the paper draft has a citation; the three numbers claimed as
  "our contribution" are sourced to `docs/exp_4(a).md` / `exp_4(b).md` / the freeze
  manifests, not to any external paper.
- `papers/` (7 PDFs) is a seed set; the corpus above supplies at least 8 fully
  identifiable arXiv entries the student can download to reach 15 physical PDFs.