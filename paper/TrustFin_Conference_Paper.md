# TrustFin-SLM — Phase-4 Conference-Paper Draft (Markdown mirror)

**Canonical LaTeX source:** `paper/TrustFin_Conference_Paper.tex` (compiles with `pdflatex`
+ `natbib`). This Markdown is the 1:1 readable mirror for review and for the showcase page.
All numbers are from the completed, validated runs (`docs/exp_4(a).md`,
`docs/exp_4(b).md`, `artifacts/phase4/manifests/`); the freeze was never modified after the
protected test was opened.

---

**Title:** Execution-Verified Generation with Agreement-Calibrated Abstention for Trustworthy Small Language Models in Finance

**Author:** Heet Mehta — TrustFin-SLM Research Project (TODO: affiliation/contact)

---

## Abstract

Programmatic numerical reasoning benchmarks such as FinQA let a language model compute an
answer by *executing the program it writes* against the official evaluator. We add
**EVG-CAS** — execution-verified generation with agreement-calibrated abstention — to a
frozen, QLoRA-adapted 1.7B financial model, with one pre-registered study question: does
paying a 3.7× sampling cost for execution-verified generation make the model *more
accurate*, *more calibrated*, or both? The answer is threefold and honest.

1. It does **not** make the model more accurate: on the protected FinQA held-out test,
   execution-verified selection (65.56%) statistically ties the frozen greedy baseline
   (65.74%) — a negative result we report rather than bury.
2. It transforms *trustworthiness*: the agreement among the model's own executed answers is
   a steep, monotone correctness signal (2.9% → 79.2% accuracy across agreement bins) that
   is nearly invisible to the model's own log-probabilities (ROC-AUC 0.58).
3. Calibrating that signal with out-of-fold Platt scaling produces a pre-registered
   abstention rule that *transfers*: at an 80% coverage target frozen on validation, the
   protected test suffers **31.0% fewer errors** among the answers it keeps (76.25%
   accuracy), vs. 34.2% predicted.

On FiQA the abstention benefit is negligible, and we report that too. A zero-shot probe on
FinBen-FOMC shows the frozen adapter also improves out-of-domain stance classification
(macro-F1 0.21 → 0.30, parse 0.90 → 0.99). Generation-level verification does not make a
benchmark-tuned small model *right* more often; it makes it *know when it is wrong* — the
property finance deployment actually needs.

**Keywords:** small language models · financial reasoning · selective prediction ·
calibration · execution-verified generation · abstention · FinQA · FiQA · FOMC

---

## 1. Introduction

Autoregressive language models are now small enough to run on a laptop and strong enough to
reason over financial documents [Phogat et al. 2024; Chen et al. 2021]. The remaining
obstacle to deploying a 1.7B model in finance is not raw accuracy, it is *trust*: a
financial system that silently gives a wrong number is unusable even if it is right 65% of
the time. Benchmarks have begun to measure this deficit — FinTrust lists "appropriately
abstain when confidence is low" as a first-class trustfulness dimension [Hu et al. 2025] —
but they diagnose it on GPT-4-scale models and provide no mechanism a small model can
actually use.

We provide the mechanism, and report the outcome honestly. We study a task class where
*execution* is a well-defined ground-truth operation: programmatic numerical reasoning
[Chen et al. 2021]. The model writes a program; the answer is what the program computes.
That property lets us sample *n* candidate programs per question, keep only those that run
on the official evaluator, and observe how much the surviving *answers* agree. Prior work
uses such sampling to *vote* (self-consistency [Wang et al. 2023]) or to train outcome /
process reward models [Lightman et al. 2024; Wang et al. 2024], framing the payoff as
accuracy. On protected data that framing is wrong for our model: execution-verified
selection does not beat the greedy baseline (Table 1). We therefore spend the sampling
budget where it pays: agreement is a trust feature, and a calibrated, pre-registered
abstention policy built on it cuts protected-test answered-error by **31%** at 80%
coverage (Table 4).

**Contributions.** **(C1)** An honest, pre-registered accuracy verdict: EVG does not
improve FinQA/FiQA accuracy for a QLoRA-tuned 1.7B model; it ties it. **(C2)** A calibration
construction that gives the model probability-meaningful confidence, cutting Brier
0.28 → 0.16 and ECE 0.26 → 0.05. **(C3)** A selective-prediction policy whose threshold is
fixed on validation and whose benefit transfers to a protected test (−31.0% test
answered-error at coverage 0.8, predicted −34.2%).

## 2. Related work

- **Small financial models & adaptation.** 1–3B models are a deliberate
  performance/efficiency point [Subramanian et al. 2025; Abdin et al. 2024; Zhang et al.
  2024]; LoRA/QLoRA make adaptation cheap [Hu et al. 2022; Dettmers et al. 2023].
  Financial adapters: Phogat et al. 2024 (closest comparable — fine-tuning SLMs on FinQA),
  FinGPT [Yang et al. 2023], InvestLM [Yang et al. 2023], BloombergGPT [Wu et al. 2023],
  FinBERT [Huang et al. 2020], Kronos [Shi et al. 2025]. All report accuracy only.
- **Programmatic financial reasoning.** FinQA [Chen et al. 2021], ConvFinQA [Chen et al.
  2022], TAT-QA [Zhu et al. 2021]: execution is the answer's ground truth. FinBen [Xie et
  al. 2024] benchmarks it; FinTrust [Hu et al. 2025] measures abstention as a trust deficit.
- **Consistency & verification.** Self-consistency [Wang et al. 2023]; step-by-step /
  process supervision [Lightman et al. 2024; Wang et al. 2024]; execution-guided synthesis
  [Chen et al. 2019]. All accuracy-centric; ours re-purposes them as a *decision signal*.
- **Selective prediction & calibration.** Risk-coverage [El-Yaniv & Wiener 2010; Geifman &
  El-Yaniv 2017, 2019]; Platt/isotonic/temperature [Platt 1999; Zadrozny & Elkan 2002; Guo
  et al. 2017]; verbalised uncertainty [Lin, Hilton & Evans 2022]. XAI guidance in finance:
  [Fritz-Morgenthal et al. 2022].

**Gap (the fill-in).** Accuracy-oriented verification has never been connected to selective
prediction for a *domain-adapted SLM*, evaluated with a pre-registered, calibration-grounded
abstention threshold on a fully protected test set. Benchmarks measure abstention
(FinTrust); selective-prediction theory builds it on classifier logits; financial SLM work
never measures it. Full map: `docs/related_work.md`, `docs/literature_reading_matrix.md`.

## 3. Problem statement and protocol

**Setup.** A question *q* maps to a program *p*; the answer is *a* = exec(*p*) if *p* is
valid. Model is a frozen, domain-adapted SLM π_θ. Three pre-declared claims (fixated in
`PLAN.md` before any protected-test access):

- **C1 (accuracy):** execution-verified selection improves the answer's
  execution-ground-truth accuracy over greedy decoding.
- **C2 (probability quality):** calibrating trust features lowers Brier/ECE and makes a
  coverage threshold carry probability meaning.
- **C3 (ranking):** trust features from the sample population (agreement, executability)
  improve the calibrator's ordering (lower expected AURC) whenever they are non-monotone
  with the raw confidence.

**Guardrails (integrity before novelty).** All selection of model, method, and
hyper-parameters (*n*=8, temperature 0.7, top-p 0.9, 5-fold out-of-fold calibration,
coverage targets {0.80, 0.90}) happened on *validation* data only. The protected test sets
were opened exactly once, after a freeze manifest registered status `FROZEN` and the run
set an explicit protected flag; the freeze SHA-256 is unchanged across the access. Full
integrity protocol: `docs/exp_4(a).md` §6 and `docs/exp_4(b).md`.

## 4. Method: EVG-CAS

### 4.1 Sampling and execution verification
Sample *n* programs per question (temp 0.7, top-p 0.9). Run each with the official
evaluator; an answer exists iff the program is valid and executes. Evidence is carried by
the candidate *population*, not a single decode.

### 4.2 Trust features from the sample population
For question *q* with *n* candidates:

```
A(q) = (1/(n(n-1))) Σ_{i≠j} 1[a^(i) == a^(j)]      # agreement among executed answers
R(q) = (1/n) Σ_i e^(i)                              # executability (valid programs)
C(q) = mean log π_θ(p* | q)                         # self-confidence of selected candidate
```

Feature vector *f*(*q*) = [A, R, C]. Gold answers are used only to *score* features, never
to select a candidate or threshold (guardrail: nothing refit on test).

### 4.3 Calibration (C2)
Out-of-fold Platt scaling: p̂(*q*) = σ(*w*·*f* + *b*), fit by logistic regression on
inner-fold predictions, applied to outer folds (5-fold), so every validation prediction is
out-of-sample. Because a single-feature monotone calibrator cannot reorder answers, we
compare (i) a single-feature baseline f=[C] and (ii) the full vector (C3).

### 4.4 Freeze and abstention rule
For coverage target 1−α (pre-registered α ∈ {0.20, 0.10}):

```
θ = min{ t : Coverage_val(t) ≥ 1−α },  Coverage(t) = (1/|V|) Σ 1[p̂(q) ≥ t]
```

Decision rule on any future question: answer iff p̂(q) ≥ θ, else abstain.
Selective risk: R_sel(θ) = Σ ℓ(q)·1[p̂(q)≥θ] / Σ 1[p̂(q)≥θ], with ℓ(q) the correctness
indicator.

**Algorithm (fit once on validation, frozen thereafter):** (1) on validation, sample n
programs per question, compute f(q); (2) fit out-of-fold Platt (w,b), score p̂ OOF;
(3) pick θ per coverage target; (4) write `phase4_freeze.json` (FROZEN, w,b,θ,α, sha256);
(5) protected phase — opened once, after FROZEN + protected flag — then per test question
draw n candidates and emit answer(q) iff p̂(q) ≥ θ, else abstain.

## 5. Experimental setup

- **Data & aligned evaluation.** FinQA (train 6,251 / val 883 / held-out test 1,147):
  printed accuracy (the emitted string), program accuracy (parse success), execution
  accuracy (executed answer matches gold) via the official evaluator. FiQA (val 117 / test
  234): MAE, R², Pearson r at tolerance 0.10. FOMC (test 496; neutral 245 / dovish 129 /
  hawkish 122): zero-shot stance, accuracy / macro-F1 / parse.
- **Model.** Frozen Phase-3 winner `Qwen/Qwen3-1.7B` with balanced-joint QLoRA, canonical
  (FinQA) and explicit-schema (FiQA); exact hyper-parameters in `phase3_freeze.json`,
  fixed before this work.
- **Metrics.** Brier, ECE-10, expected AURC (lower better), coverage, selective risk.
  "validation" = out-of-fold / tuned on validation; "test" = frozen system, unchanged since
  the completed protected run.

## 6. Results

### 6.1 C1 — rejected, reported (Table 1)

**FinQA test (n=1,147), execution accuracy:**

| rule | execution accuracy |
|---|---|
| frozen greedy (baseline) | **0.6574** |
| candidate-0 (prints) | 0.6399 |
| self-consistency vote | 0.6478 |
| best-of-n | 0.6556 |
| **execution-verified (ours)** | 0.6556 |

**FiQA test (n=234):**

| aggregator | MAE | accuracy@0.10 |
|---|---|---|
| frozen greedy | **0.1301** | **0.568** |
| median-of-n (ours) | 0.1306 | 0.556 |

Execution-verified selection is −0.18 pp vs. greedy (18 questions gained, 20 lost; exact
McNemar p=0.871, bootstrap 95% CI on the delta [-0.0122, +0.0087]);
validation was a tie (0.6750 = 0.6750). The one clean C1 win: **program hygiene** — the
invalid-program rate in emitted answers drops from 3.5e-3 to exactly **0**.

### 6.2 Agreement is the trust signal (Table 2, protected test)

| EVG agreement bin | n records | execution accuracy |
|---|---|---|
| (0.00, 0.25] | 35 | **0.029** |
| (0.25, 0.50] | 122 | 0.238 |
| (0.50, 0.75] | 181 | 0.448 |
| (0.75, 1.00] | 809 | **0.792** |

The model reports disagreement on records that are 2.9% correct and agreement on records
that are 79.2% correct; its own log-probability cannot separate these regimes (ROC-AUC 0.58).

### 6.3 Calibration and ranking (Table 3)

**e-AURC on FinQA validation (frame = exec-verified), lower is better:**

| calibrator (feature set) | e-AURC |
|---|---|
| raw confidence | 0.2029 |
| + isotonic (confidence) | 0.2028 |
| + Platt (confidence) | 0.2496 |
| **+ Platt (agreement, executability, confidence)** | **0.1232** |
| *(baseline greedy frame, best method)* | *0.1897* |

**Probability quality (frozen FinQA policy):** Brier 0.2845 → **0.1579**; ECE 0.2555 →
**0.0475**. Single-feature calibrators cannot move ranking (monotone maps); the
agreement-driven vector drops e-AURC by 39.3%.

### 6.4 The frozen policy transfers (Table 4)

**FinQA (frozen: exec-verified + Platt-full + θ):**

| target | θ | test coverage | selective risk | selected accuracy | abstained | err. rel. Δ |
|---|---|---|---|---|---|---|
| 0.80 | 0.409 | 0.786 | **0.2375** | **0.7625** | 246 | **−31.0%** |
| 0.90 | 0.221 | 0.926 | 0.3004 | 0.6996 | 85 | −12.8% |

**FiQA (frozen: greedy + raw confidence):** coverage 0.80 → risk 0.4190 vs. full 0.4316
(−2.9%); coverage 0.90 → −2.6%. As the freeze predicted: FiQA errors are not rankable.

### 6.5 Evaluation rigor (Table 6, protected test)

**C1 significance — paired test:**

| | greedy | exec-verified |
|---|---|---|
| execution accuracy | 0.6574 | 0.6556 |
| delta | −0.0017 (95% CI [−0.0122, +0.0087]) | |
| McNemar exact p | 0.871 | |

**Error taxonomy of wrong answers (n=1,147):** parse failure 0 (0.0%); invalid program 0
(0.0%); executes but wrong 395 (34.4%); executes right but prints the wrong string 0 (EVG)
vs. **355 (31.0%) for greedy** — the only error channel execution-verification fails to fix
is the human-facing *printed* answer (roadmap R1).

**Reasoning-depth stratification (abstention @θ0.80):** 0-op programs: full 0.844 /
selected 0.884, abstained 4.4%; 1-op: 0.649 / 0.764, 23.1%; 2-op: 0.662 / 0.748, 18.9%;
3–6-op: 0.552 / **0.739**, abstained **31.3%**, error cut **41.7%**. The policy spends its
abstention exactly where reasoning is deepest. Bootstrap CIs for all figures are in
`trust_fin_extra_phase/docs/eval_rigor.md`.

### 6.6 Out-of-domain robustness (Table 5, FOMC, zero-shot)

| system | accuracy | macro-F1 | parse | F1 dovish | F1 neutral | F1 hawkish |
|---|---|---|---|---|---|---|
| base Qwen/Qwen3-1.7B | 0.454 | 0.215 | 0.899 | 0.163 | 0.650 | 0.046 |
| **+ frozen E5 adapter** | 0.484 | **0.304** | **0.986** | 0.333 | 0.621 | 0.260 |

### 6.7 Cost
EVG n=8: 4.8 s/question vs. 1.3 s greedy (3.7×, shared prefill); peak VRAM 3.1 GB / 6 GB.

## 7. Discussion and limitations

- **What we did not claim.** EVG-CAS does not raise accuracy; reporting a tie rather than a
  p-value-won headline is the point of the pre-registration.
- **Domain limits.** On FiQA (117 val / 234 test, regression), the EVG features carry no
  signal; the frozen policy correctly abstains to no effect (−2.9%) — a second reported
  negative result.
- **General limits.** One backbone, one seed, one harness; between-seed variance of the
  agreement signal unmeasured; n=8 pre-declared, not tuned; 3.7× latency buys abstention
  power, not accuracy.
- **Threats to validity.** The transfer claim rests on the freeze hash being unchanged
  across protected access (verified: yes). Any leakage would invalidate C3's transfer.
- **Ethics.** Abstention is a refusal signal, not a guarantee; the residual error of kept
  answers is 23.8% at coverage 0.8. No financial advice implied.

## 8. Conclusion

For a QLoRA-adapted 1.7B financial model, execution-verified generation ties greedy
decoding on accuracy but produces a steep, monotone agreement signal the model's own
confidence cannot see. Calibrating that signal yields a pre-registered abstention threshold
that transfers to a once-opened protected test, cutting answered-error by **31%** at 80%
coverage. The contribution is that generation-level verification does not make a small
model *right* more often — it makes it *know when it is wrong*, the property finance
deployment needs most. Artifacts, freeze manifests, and the full experiment trail are
public; **P4-E9** (FinTrust per-dimension evaluation) is the declared next step.

## References

See `docs/trustfin_references.bib` (29 entries) and the manual bibliography in the LaTeX
source. Core 15 rows are marked CORE in `docs/literature_reading_matrix.md`.