# Literature Reading Matrix — TrustFin-SLM

Guide §4/§16 deliverable. 15 **CORE** entries (guide minimum: 10–15); √ = PDF already in
`papers/`. Columns: what they do · our take · gap they leave for EVG-CAS.

| # | Paper (year, venue) | Core | Family | Main technique | Reported result | Our use | Gap they leave |
|---|---|---|---|---|---|---|---|
| 1 | **FinQA** — Chen et al. (2021, EMNLP) √ | CORE | finqa | Program-based numerical-reasoning dataset; executor as answer ground truth | Defines the task; ~60–80 % program accuracy SOTA at release | Task + **official executor** (basis of execution-verified sampling) | No confidence/abstention; accuracy-only |
| 2 | **ConvFinQA** — Chen et al. (2022, EMNLP) √ | CORE | finqa | Conversational multi-hop numerical reasoning | Chains of programs over dialogue turns | Related task; motivates program-level eval | Same as above |
| 3 | **TAT-QA** — Zhu et al. (2021, ACL) | CORE | finqa | Hybrid tabular+text financial QA with operators | SOTA on hybrid reasoning | Shows execution-style operators generalise across financial QA | No trust layer |
| 4 | **Fine-tuning Smaller LMs for FinQA** — Phogat et al. (2024, Findings EMNLP) √ | CORE | finllm | PEFT/instruction tuning of small LMs on financial QA | Strong FinQA accuracy at small scale | **Closest comparable**: same compute-regime premise we inherit | No calibration, no abstention, no protected-test protocol; accuracy-only reporting |
| 5 | **FinGPT** — Yang et al. (2023) √ | CORE | finllm | Open financial LLM ecosystem, RLHF/sentiment | Open alternative to proprietary finance LLMs | Contrast: adaptation without uncertainty handling | Sentiment-tuned; no decision policy |
| 6 | **InvestLM** — Yang et al. (2023, arXiv 2309.13064) | CORE | finllm | LoRA instruction tuning (r=16) for investment QA | Expert-rated ∝ GPT-3.5/4 on advisory | Evidence LoRA suffices for financial adaptation | Expert eval ≠ protected quantitative eval; no uncertainty |
| 7 | **BloombergGPT** — Wu et al. (2023) | CORE | finllm | 50B finance LM trained from scratch | Beats generic on finance, close on general | Motives a *smaller, tunable* alternative | Closed weights; unmeasurable uncertainty policy |
| 8 | **Kronos** — Shi et al. (2025, arXiv 2508.02739) √ | CORE | finllm | Time-series foundation model for market language | Financial forecasting gains | Positions SLM scope (text-reasoning vs market data) | Different modality; not QA reasoning |
| 9 | **FinBen** — Xie et al. (2024, NeurIPS D&B) | CORE | finqa | 42 datasets / 24 financial tasks benchmark | LLMs strong at IE, weak at reasoning/forecasting | FOMC stance + FiQA tasks come from this benchmark | Evaluates, does not fix; no abstention mechanism |
| 10 | **FinTrust** — Hu et al. (2025, EMNLP) | CORE | finqa | 7-dimension trustworthiness benchmark (15 680 pairs), incl. "abstain when confidence low" | Even GPT-4-scale models abstain poorly | **Direct motivation**: trustfulness/robustness dimension is a *measured* deficit | Diagnostic only — no mechanism transferable to 1.7B models |
| 11 | **FinBERT** — Huang et al. (2020) | CORE | finllm | Pre-trained financial language model | Strong financial sentiment/NER | Predecessor; motivates domain pretraining for SLMs | Encoder-era; no generation/execution |
| 12 | **Self-Consistency** — Wang et al. (2023, ICLR) | CORE | verif | k sampled reasoning paths → majority vote | Large gain on arithmetic/commonsense | Direct ancestor of our sampling; **agreement is our feature, not our vote** | Evaluated as accuracy gain only; no selective prediction; no negative-result honesty |
| 13 | **Let's Verify Step by Step / Math-Shepherd** — Lightman et al. (2024, ICLR); Wang et al. (2024, ACL) | CORE | verif | Process reward models verify each reasoning step | PRM reranking beats self-consistency | Shows verification boosts *selection* — we verify by *execution*, not a learned RM | Needs a trained verifier/refit; never frames abstention or protected transfer |
| 14 | **SelectiveNet / Selective classification** — Geifman & El-Yaniv (2017 NeurIPS, 2019 ICML) | CORE | selcal | Trainable reject option; risk-coverage optimisation | Selective error far below full-coverage | **Framework** for our abstention evaluation (risk@coverage, e-AURC) | Built for classifier logits, not sampling-population signals from generative models |
| 15 | **Calibration** — Guo et al. (2017, ICML); Platt (1999); Zadrozny & Elkan (2002); Lin et al. (2022, TMLR) | CORE | selcal | Temperature/Platt/isotonic calibration; verbalised uncertainty | Calibrated ECE on modern nets and GPT-3 | Calibration backbone (OOF Platt); **caveat**: monotone calibrators can't fix ranking | None evaluate *transfer of a frozen threshold to a protected test* |

**Secondary (informative, not core):** El-Yaniv & Wiener 2010 (risk-coverage theory),
Smith et al. — none; Execution-Guided Program Synthesis (Chen et al. 2019),
SLM survival (Subramanian et al. 2025; Abdin et al. 2024; Zhang et al. 2024),
Qwen3 (Qwen Team 2025), LoRA/QLoRA (Hu et al. 2022; Dettmers et al. 2023),
XAI in risk (Fritz-Morgenthal et al. 2022).

## Reading status

| Status | Row(s) |
|---|---|
| Full PDF on disk (`papers/`) | 1, 2, 4, 5, 8 |
| Deep-read into method details | 1, 4, 12, 14, 15 (families we build on) |
| Read for positioning | 6, 7, 9, 10, 13 |
| **To download (arXiv IDs known)** | 3 (2105.07624), 6 (2309.13064), 9 (2402.12659), 10 (2510.15232), 12 (2203.11171), 13 (2312.08935), 15 (2006.08097/2205.14334), plus [smlm] refs |

Once 8 of the "to download" set are fetched, `papers/` reaches ≥15 physical PDFs.