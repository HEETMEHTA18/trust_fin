# Deep Explanation — `Phase2_1_TrustFin_Research_Validation_and_Data_Contract.ipynb`

**TrustFin-SLM · Phase 2.1 · Research Validation, FiQA Sentiment Integration, and Phase-3/4 Data Contract**

This document explains every cell of `/home/heet18/Downloads/Research Papers/Phase2_1_TrustFin_Research_Validation_and_Data_Contract.ipynb` (47 cells). It covers the purpose, logic, key variables/functions, and the integrity gates added by each cell, plus the artifacts the notebook writes and how the PASS/FAIL contract works.

---

## 1. What the notebook does (30-second version)

Phase-2.1 is a **research-ready revision** of Phase-2. It replaces the unavailable FinNLI dataset with **FiQA Sentiment** (`TheFinAI/fiqa-sentiment-classification`), validates every FinQA gold program with the **official FinQA executor**, and adds mandatory research-integrity checks (exact leakage, near-duplicate audit, protected-test isolation, source/artifact fingerprints, and final PASS/FAIL gates).

| Source | Role | Split policy |
|---|---|---|
| **FinQA** (`czyssrs/FinQA`) | Financial numerical reasoning, program+answer supervision | canonical `train`/`dev`/`test`; test is **protected** |
| **FiQA Sentiment** (`TheFinAI/fiqa-sentiment-classification`) | Aspect-aware financial sentiment regression, continuous score in [-1,1] | canonical `train`/`valid`/`test`; test is **protected** |
| **FinBen-FOMC** (`TheFinAI/finben-fomc`) | FOMC stance classification (`dovish`/`hawkish`/`neutral`) | single official `test` split → **entirely evaluation-only** |
| **FinTrust** (`HughieHu/FinTrust`) | 7-dimension trust benchmark | **evaluation-only**, kept as a registry (never flattened into SFT) |

Design philosophy:

1. **Never hand-edit gold.** Training targets are generated directly from official records.
2. **No privileged evidence.** Gold evidence is stored as annotation metadata, never put in the model input.
3. **No chain-of-thought fabrication.** Only the official answer formats are used (`PROGRAM:`/`ANSWER:` for FinQA, `SENTIMENT_SCORE:` for FiQA, or a label for FOMC).
4. **No evaluation tuning.** Calibration/threshold selection is restricted to validation; all test + FinBen-FOMC + FinTrust are protected.
5. **Official executor validation.** FinQA gold programs are re-executed with the official `evaluate.py` from the FinQA repository.
6. **Stable interface.** Downstream phases import `phase2_1_interface.py` and read JSONL/JSON artifacts instead of re-downloading benchmarks.

**Every integrity check is recorded as a `Gate`.** The run passes only if all `ERROR`-severity gates pass (`PHASE-2.1 STATUS: PASS`).

---

## 2. Data flow overview

```
FinQA (GitHub raw)     ──► raw download (+SHA-256 manifest)  ──► flatten → gates ──► records(train/dev/test) ──┐
FiQA Sentiment (HF)    ──► HF load + deduplicate IDs          ──► normalize → gates ─► records(train/valid/test) ─┤
FinBen-FOMC (HF)       ──► token-authenticated load           ──► normalize → gates ─► records(test, eval-only)  ─┼─► pools → leak audit → export artifacts
FinTrust (HF+GitHub)   ──► snapshot + evaluator clone         ──► manifest + gates ─► registry (eval-only)       ─┘                                  │
                                                                                                                                                          ▼
        phase3_data_config.json · phase4_eval_config.json · dataset_registry.json · phase2_1_validation_report.json · phase2_1_interface.py ◄── gates
```

Output root: `PROJECT_ROOT = $TRUSTFIN_ROOT` or `./trustfin_slm_workspace`, then `artifacts/phase2_1/`.

---

## 3. Before running

- **Python deps:** `pandas`, `numpy`, `requests`, `datasets`, `huggingface_hub`, `pyarrow`, `scikit-learn`, `tqdm`, `datasketch`, `sympy` (cell 2 auto-installs missing ones via `pip`).
- **Hugging Face token:** FinBen-FOMC may be gated → set `HF_TOKEN` in the environment before running. Cell 28 warns if it is missing.
- **Network access:** FinQA files are downloaded from GitHub raw; FiQA, FinBen-FOMC, and FinTrust are loaded from HuggingFace; FinTrust evaluator code is cloned from GitHub.

---

## 4. Directory layout produced

```
trustfin_slm_workspace/
├── data/raw/
│   ├── finqa/                          # train.json, dev.json, test.json (downloaded, unchanged)
│   ├── finqa_official_evaluator/       # official evaluate.py from FinQA repo
│   ├── fintrust_hf/                    # FinTrust HF snapshot (raw files untouched)
│   └── fintrust_eval_code/             # official postprocess_response.py evaluator repo
└── artifacts/phase2_1/
    ├── datasets/
    │   ├── train.jsonl                 # SFT training (FinQA train + FiQA train)
    │   └── validation.jsonl            # validation + calibration (FinQA dev + FiQA valid)
    ├── eval/
    │   ├── finqa_test.jsonl            # protected FinQA test
    │   ├── fiqa_test.jsonl             # protected FiQA test
    │   ├── finben_fomc_test.jsonl      # protected FinBen-FOMC test
    │   └── fintrust_registry.json      # protected FinTrust registry
    ├── manifests/
    │   ├── dataset_registry.json
    │   ├── source_fingerprints.json
    │   ├── phase3_data_config.json
    │   ├── phase4_eval_config.json
    │   └── phase2_1_validation_report.json
    ├── audits/
    │   ├── finqa_official_gold_execution_audit.csv
    │   └── near_duplicate_train_vs_protected_test.csv
    └── python/
        └── phase2_1_interface.py        # downstream loader helpers
```

---

## 5. Cell-by-cell walkthrough

Notation: **[markdown]** = documentation/explanation · **[code]** = executable Python.

---

### Cell 0 — [markdown] Title, purpose, design rules
- States the research context (Phase-3 → SLM baselines/QLoRA; Phase-4 → TrustFin-SLM: verification, calibration, selective prediction, FinTrust evaluation).
- Enumerates the **four corrections** applied before final modeling:
  1. Replace unavailable FinNLI with FiQA sentiment using native continuous score in [-1,1].
  2. Keep FinBen-FOMC strictly test-only as an external financial generalization benchmark.
  3. Validate every FinQA gold program with the official FinQA executor.
  4. Make research-integrity checks mandatory.
- Lists the **downstream policy**:
  - Phase-3 training/validation: FinQA + FiQA only.
  - Protected evaluation: FinQA test + FiQA test + FinBen-FOMC + FinTrust.
  - FinTrust never enters SFT, calibration, threshold selection, or prompt-example pools.
  - FinBen-FOMC never enters SFT or threshold selection.
  - All downstream notebooks should read exported registry/config files rather than reloading raw datasets.

---

### Cell 1 — [markdown] Environment note
Explains that the dependency cell installs only missing packages and that the notebook uses deterministic serialization and fixed seeds for auditability.

---

### Cell 2 — [code] Auto-installs missing dependencies
- Checks for: `pandas`, `numpy`, `requests`, `datasets`, `huggingface_hub`, `pyarrow`, `sklearn`, `tqdm`, `datasketch`, `sympy`.
- Runs `pip install -q` for any missing packages via `subprocess.check_call`.
- Prints "All required packages are already available." if nothing is missing.

---

### Cell 3 — [code] Imports, seed, environment probe
- `from __future__ import annotations`, standard library imports (`csv`, `hashlib`, `json`, `os`, `platform`, `random`, `re`, `shutil`, `time`, `Counter`, `defaultdict`, `@dataclass`, `Path`, typing helpers).
- Third-party: `numpy`, `pandas`, `requests`, `tqdm`.
- **Seed** `SEED = 42`, applied to `random` and `numpy`.
- Prints Python / platform / pandas versions → reproducibility breadcrumbs for the paper.

---

### Cell 4 — [markdown] Configuration overview
Documents the workspace convention inherited from Phase-2 and explains that Phase-2.1 writes to a **new artifact directory** and never overwrites Phase-2 exports.

---

### Cell 5 — [code] Root paths, directories, and all constants
This is the **central configuration cell**. Everything downstream depends on these names:

- **Paths:**
  - `PROJECT_ROOT` — `$TRUSTFIN_ROOT` env var, default `./trustfin_slm_workspace`.
  - `RAW_DIR`, `ARTIFACT_DIR`, `EXPORT_DIR`, `EVAL_DIR`, `MANIFEST_DIR`, `MODULE_DIR`, `AUDIT_DIR` — all `mkdir(parents=True, exist_ok=True)`.
- **FinQA:** `FINQA_BASE_URL` (GitHub raw), `FINQA_EVALUATOR_URL` (official evaluate.py), `FINQA_EXPECTED = {train:6251, dev:883, test:1147}`.
- **FiQA Sentiment:** `FIQA_HF_REPO = "TheFinAI/fiqa-sentiment-classification"`, `FIQA_SPLIT_MAP = {train: "train", valid: "valid", test: "test"}`, `FIQA_EXPECTED = {train:822, valid:117, test:234}`.
- **FinBen-FOMC:** `FINBEN_FOMC_HF_REPO = "TheFinAI/finben-fomc"`, `FINBEN_FOMC_SPLIT = "test"`, `FINBEN_FOMC_EXPECTED = 496`, `FINBEN_FOMC_LABELS = ["dovish","hawkish","neutral"]`.
- **Auth:** `HF_TOKEN = os.environ.get("HF_TOKEN", "").strip()`.
- **FinTrust:** `FINTRUST_HF_REPO`, `FINTRUST_GITHUB`, and the seven official dimensions (`trustfulness, robustness, safety, fairness, privacy, transparency, knowledge_discovery`).
- **Integrity knobs:** `DOWNLOAD_TIMEOUT=90`, `RUN_NEAR_DUP_AUDIT=True`, `NEAR_DUP_THRESHOLD=0.90`, `MINHASH_PERMUTATIONS=96`, `WORD_SHINGLE_N=3`.
- **Contract version:** `SCHEMA_VERSION = "trustfin-phase2.1-v1.0"` — stamped into every record and manifest.
- Prints the resolved root, whether Phase-2 registry exists, and whether a token is configured.

---

### Cell 6 — [markdown] Phase-2 provenance bridge
Explains that if the Phase-2 registry exists, its SHA-256 hash is recorded for explicit lineage from Phase-2 to Phase-2.1.

---

### Cell 7 — [code] Phase-2 provenance check
- If `PHASE2_REGISTRY_PATH.exists()`, computes `sha256` of the file and stores it in `phase2_parent_provenance`.
- Otherwise prints "Parent Phase-2 registry not found; Phase-2.1 will run standalone."

---

### Cell 8 — [markdown] Shared utilities and validation gates
States that the helpers in the next cell are reused by later phases via the exported module.

---

### Cell 9 — [code] **Serialization, hashing, I/O, and the Gate framework**
The workhorse of the notebook. Notable pieces:

- `json_default(x)` — converts NumPy/Path/NaN types into JSON-safe values.
- `canonical_json(obj)` — deterministic JSON (`sort_keys=True`, compact separators) → same object always serializes identically → safe input to hashing.
- `sha256_text`, `sha256_file` — content hashing for provenance manifests.
- `normalized_text(x)` — lowercase + collapse whitespace; the canonical form for leakage signatures.
- `input_signature(*parts)` — `sha256` of normalized parts joined by a sentinel. **This is the leakage-detection unit**: it fingerprints the *model input*, not the source ID.
- `write_jsonl` / `read_jsonl` — stable record file I/O.
- `save_json`, `load_json_or_jsonl`, `load_tabular_file` — tolerant reading of JSON/JSONL/CSV/TSV/Parquet (used to inventory FinTrust).
- **`Gate` dataclass** (`name, passed, observed, expected, severity, note`) and the **`GATES` ledger list** with `add_gate(...)`. Every check in the notebook funnels through `add_gate`, which prints `[PASS]/[FAIL]/[WARN]` immediately and appends to `GATES`.

---

### Cell 10 — [code] Download helpers
- `download_file(url, path, overwrite=False)` — idempotent HTTP download with `DOWNLOAD_TIMEOUT`; `raise_for_status()` on failure.
- `hf_dataset_to_pandas(repo_id, split)` — wraps `datasets.load_dataset` → pandas DataFrame.
- `safe_float(x)` — tolerant float conversion.
- `assert_record_contract(records, name)` — verifies every record has the canonical 13 fields: `record_id`, `source_id`, `dataset`, `task`, `split`, `system`, `instruction`, `input`, `target`, `target_answer`, `target_label`, `gold_program`, `gold_evidence`, `metadata`.

---

## Part A — FinQA

### Cell 11 — [markdown] Part A intro — FinQA download
Notes that raw JSON files are kept **unchanged** and SHA-256 hashes are recorded for traceability.

---

### Cell 12 — [code] FinQA download + source manifest
- `FINQA_RAW_DIR` created.
- Downloads `train.json`/`dev.json`/`test.json` from the canonical GitHub release; paths stored in `finqa_raw_paths`.
- `finqa_source_manifest` — per split: absolute path, **sha256 of the raw file**, byte size. Displayed as a table.

---

### Cell 13 — [code] FinQA flattening into a DataFrame
- `finqa_table_to_text(table)` — renders the nested list-of-lists table as `row1 | col1 | col2 ...` lines.
- `finqa_gold_evidence_to_text(gold_inds)` — serializes gold evidence (dict/list).
- `flatten_finqa(raw_rows, split)` — unpacks each record: `pre_text`/`post_text`, `table`, `question`, `answer` (human annotation), `exe_ans` (executable numeric result), `program`/`program_re`, `steps`, `gold_inds`, `final_step_result` (last step's `res`), plus derived `context_text` (pre+post joined), `table_text`, `gold_evidence_text`.
- Builds `finqa_splits = {split: DataFrame}` and concatenates into `finqa_df`.

---

### Cell 14 — [markdown] FinQA structural integrity
Explains that `steps[-1].res` is retained only as **annotation metadata** and is not used as the research-valid execution check. The decisive check uses the official FinQA evaluator.

---

### Cell 15 — [code] FinQA integrity gates + diagnostics
The first big gate cluster:

- `parse_number_loose(x)` / `loose_equivalent(a, b)` — tolerant numeric equivalence (handles `23.6%` vs `0.236`, commas, `$`, `NA`), compared with `np.isclose` using `atol=1e-4, rtol=1e-3`.

**Gates added:**

| Gate | Severity | Meaning |
|---|---|---|
| `FinQA {train/dev/test} count` | ERROR | matches `FINQA_EXPECTED` |
| `FinQA IDs globally unique` | ERROR | no duplicate IDs |
| `FinQA required field present: {id, question, exe_ans, program, gold_inds, table}` | ERROR | no missing gold/supervision fields |
| `FinQA final step matches an official answer representation` | WARNING | ≥99% of final-step results equal `answer` or `exe_ans` after loose comparison |
| `FinQA program operation vocabulary recognized` | ERROR | only known ops (`add, subtract, multiply, divide, greater, exp, table_sum, table_average, table_max, table_min`) |
| `FinQA no exact full-input leakage across splits` | ERROR | no question+context+table signature appears in >1 split |

- Also computes `input_signature` for every row, prints agreement rates and duplicate-question counts.

---

### Cell 16 — [markdown] FinQA official executor
Explains that Phase-2.1 downloads the official `evaluate.py` from the FinQA repository, fingerprints it, tokenizes each gold program, executes it, and compares with `qa.exe_ans`.

---

### Cell 17 — [code] Download and load official FinQA evaluator
- Downloads `evaluate.py` from `FINQA_EVALUATOR_URL`.
- Records `finqa_evaluator_manifest` (url, path, sha256).
- Dynamically imports the module via `importlib.util.spec_from_file_location`.
- Asserts `program_tokenization` and `eval_program` functions exist.

---

### Cell 18 — [code] Official FinQA gold-program execution validation
- `validate_finqa_gold_execution(raw_path, split)` — for each item: tokenizes the gold program with `finqa_official.program_tokenization`, executes with `finqa_official.eval_program`, compares result with `qa.exe_ans` using exact Python equality.
- Runs across all 3 splits. For each split:
  - **Gate:** `FinQA {split}: official gold programs execute successfully` — all `invalid_flag == 0`.
  - **Gate:** `FinQA {split}: official execution equals qa.exe_ans` — all `official_match == True`.
- Saves full audit to `finqa_official_gold_execution_audit.csv`.

---

### Cell 19 — [markdown] FinQA record design
Defines the record contract: input = full report text + table + question (no gold); target = structured `PROGRAM: ...\nANSWER: <exe_ans>`.

---

### Cell 20 — [code] FinQA canonical records
- `FINQA_SYSTEM` — system prompt for a financial reasoning assistant.
- `make_finqa_record(row)` — builds a **record** dict:
  - `record_id = "finqa::{id}"`, `source_id`, `dataset="FinQA"`, `task="financial_numerical_reasoning"`, `split`.
  - `system`, `instruction`, `input` (evidence text + table + question), `target` (`PROGRAM:`/`ANSWER:`).
  - `target_answer = exe_ans`, `gold_program`, `gold_evidence = gold_inds`.
  - `metadata`: `schema_version`, `has_table`, `annotated_answer` (human), `execution_answer` (executable), `input_signature`, and **`evaluation_only = (split == "test")`**.
- Builds `finqa_records = {split: [records]}`; prints counts and a pretty JSON sample.

---

## Part B — FiQA continuous financial sentiment

### Cell 21 — [markdown] Part B intro — FiQA Sentiment
Explains that FiQA is used in its native continuous sentiment formulation. Phase-2.1 does **not** discretize scores into arbitrary sentiment classes.

Canonical task:
- input: financial sentence + target entity + annotated aspect.
- target: continuous sentiment score in [-1,1].
- Phase-4 primary metrics: MSE and R², with MAE/Pearson as secondary diagnostics.

---

### Cell 22 — [code] FiQA HF load
- Iterates `FIQA_SPLIT_MAP` and loads each split from `TheFinAI/fiqa-sentiment-classification` via `hf_dataset_to_pandas`.
- Builds `fiqa_source_manifest` per split: mode, repo, split, rows, sha256 (canonical JSON of all records sorted by `_id`).
- Displays the manifest table.

---

### Cell 23 — [markdown] FiQA schema normalization
Documents the required columns (`_id`, `sentence`, `target`, `aspect`, `score`, `type`) and the integrity gates.

---

### Cell 24 — [code] FiQA normalization + integrity gates + deduplication fix
- `FIQA_REQUIRED_COLUMNS = ["_id", "sentence", "target", "aspect", "score", "type"]`.
- `normalize_fiqa_frame(df, split)`:
  - **Deduplication fix:** Appends `_0`, `_1`, ... suffix to `_id` values that appear more than once within the same split. This resolves upstream FiQA duplicate `_id` values that would otherwise cause `record_id` collisions.
  - Maps columns: `id`, `split`, `sentence`, `entity_target`, `aspect`, `score`, `text_type`.
  - Computes `input_signature`.

**Gates added:**

| Gate | Severity | Meaning |
|---|---|---|
| `FiQA {train/valid/test} count` | ERROR | matches `FIQA_EXPECTED` |
| `FiQA IDs globally unique` | WARNING | upstream IDs may be reused across splits; record_ids include split prefix |
| `FiQA sentence non-empty` | ERROR | every row has text |
| `FiQA target entity non-empty` | ERROR | every row has an entity target |
| `FiQA aspect non-empty` | ERROR | every row has an aspect |
| `FiQA score numeric` | ERROR | no NaN scores |
| `FiQA score bounded in [-1, 1]` | ERROR | all scores in valid range |
| `FiQA no exact task-input leakage across splits` | ERROR | no sentence+entity+aspect signature appears in >1 split |

- Displays per-split summary statistics (count, mean, std, min, max of scores) and `text_type` distribution.

---

### Cell 25 — [markdown] FiQA record policy
Documents that the score is serialized without label discretization. Phase-3 can use the same generative SFT trainer as FinQA by treating the completion as a structured numeric target.

---

### Cell 26 — [code] FiQA canonical records
- `FIQA_SYSTEM` — system prompt for a financial sentiment assistant.
- `format_fiqa_score(x)` — formats float to `".10g"` for lossless round-trip.
- `make_fiqa_record(row)` — builds a **record** dict:
  - `record_id = "fiqa::{split}::{id}"`, `source_id`, `dataset="FiQA"`, `task="financial_sentiment_regression"`, `split`.
  - `target = "SENTIMENT_SCORE: <float>"`, `target_answer = score` (the raw float).
  - `gold_program = None`, `gold_evidence = None` (not applicable for regression).
  - `metadata`: `entity_target`, `aspect`, `text_type`, `sentiment_score`, `input_signature`, `evaluation_only = (split == "test")`.
- Builds `fiqa_records = {split: [records]}`.

---

## Part C — FinBen-FOMC protected external evaluation

### Cell 27 — [markdown] Part C intro — FinBen-FOMC
Notes that the public repository exposes a single 496-example test split. Phase-2.1 therefore hard-codes the methodological rule that **zero FOMC examples may enter training or validation**.

---

### Cell 28 — [code] FinBen-FOMC loading
- `load_finben_fomc_hf(repo_id, split)` — wraps `datasets.load_dataset`.
- **Warns (non-fatal) if `HF_TOKEN` is unset** — the dataset may be gated.
- Loads `finben_fomc_raw`; prints row count and columns.
- Builds `finben_source_manifest`: `mode="huggingface"`, repo, split, `rows`, sha256 over canonical JSON sorted by `id`.

---

### Cell 29 — [code] FinBen-FOMC normalization + gates
- `first_existing(columns, candidates)` — column-agnostic resolution.
- `normalize_fomc_label(x)` — lowercases and **rejects** anything outside the 3-vocabulary.
- `normalize_fomc_frame(df, split)` — maps columns, coerces IDs, strips text, lowercases answers, keeps `query`, parses `choices`/`gold`, sets `label`, computes `input_signature`.
- `FINBEN_AVAILABLE` flag.

**When available, gates added:**

| Gate | Meaning |
|---|---|
| `FinBen-FOMC source configured` | split loaded with N rows |
| `FinBen-FOMC test count` | == 496 |
| `FinBen-FOMC IDs globally unique` | no id collisions |
| `FinBen-FOMC text non-empty` | every statement has text |
| `FinBen-FOMC labels in expected vocabulary` | exactly `{dovish, hawkish, neutral}` |
| `FinBen-FOMC choices[gold] consistent with answer` | `choices[gold] == answer`; WARNING if no gold column |

---

### Cell 30 — [code] FinBen-FOMC canonical records
- `FINBEN_SYSTEM` — FOMC stance classifier prompt.
- `make_finben_record(row)` — record dict with `record_id = "finben-fomc::{id}"`, `dataset="FinBen-FOMC"`, `task="fomc_stance"`, `input = "Statement:\n{text}"`, `target = target_label = label`, and **`evaluation_only = True` unconditionally**.
- `finben_records = {"train": [], "dev": [], "test": [...]}` — train/dev always empty by design.

---

### Cell 31 — [code] FOMC isolation gates
Three hard isolation checks:

| Gate | Expected |
|---|---|
| `FinBen-FOMC contributes zero training examples` | 0 |
| `FinBen-FOMC contributes zero validation examples` | 0 |
| `FinBen-FOMC all public records are evaluation-only` | all True |

---

## Part D — FinTrust protected evaluation registry

### Cell 32 — [markdown] Part D intro — FinTrust
Explains that FinTrust is *not* converted to SFT: seven heterogeneous dimensions with task-specific schemas and official evaluators; Phase-2.1 builds a manifest instead.

---

### Cell 33 — [code] FinTrust snapshot + evaluator clone
- `snapshot_fintrust_hf()` — `huggingface_hub.snapshot_download` into `FINTRUST_RAW_DIR`.
- `clone_or_update_fintrust_eval_repo()` — `git pull --ff-only` if already cloned, else `git clone --depth 1`.
- Wrapped in try/except: on failure `FINTRUST_AVAILABLE = False`.

---

### Cell 34 — [code] FinTrust dimension-aware manifest
- `infer_fintrust_dimension(path, root)` — walks relative path parts; returns the first of the seven dimensions found.
- `classify_fintrust_file_role(path)` — `generated_or_result`, `candidate_eval_data`, or `unknown`.
- Iterates every snapshot file (skipping `.cache`): loads it, records `dimension`, paths, `role`, `rows`, `columns`, `sha256`, `evaluation_only=True`.
- Displays manifest table and per-dimension raw-row total.

---

### Cell 35 — [code] FinTrust integrity gates
- `FinTrust all seven dimensions represented` — each of the 7 dimensions appears in the manifest.
- `FinTrust data files load without parser errors` — no `load_error` entries.
- **Evaluator presence (WARNING):** for each dimension, checks if an evaluator script was found. Missing candidates must be resolved in Phase-4; do not replace with a generic metric.

---

## Part E — Unified Phase-3/4 contract

### Cell 36 — [markdown] Part E intro — Pools
States the pool construction explicitly: train = FinQA train + FiQA train; validation = FinQA dev + FiQA valid; protected = FinQA test + FiQA test + FinBen-FOMC test.

---

### Cell 37 — [code] Build pools + first no-leak gates
- `train_records = finqa train + fiqa train`, `validation_records = finqa dev + fiqa valid`, `protected_eval_records = finqa test + fiqa test + finben test`.
- Prints counts with per-dataset `Counter`.
- `assert_record_contract` for train, validation, and protected evaluation.

**Gates added:**

| Gate | Expected |
|---|---|
| `Training datasets exactly FinQA + FiQA` | `{"FinQA", "FiQA"}` |
| `No evaluation_only records in training` | all False |
| `No evaluation_only records in validation` | all False |
| `No FinTrust records in training/validation` | all True |
| `No FinBen-FOMC records in training/validation` | all True |

---

### Cell 38 — [markdown] Leakage audit explanation
Emphasizes that leakage is measured on **normalized model inputs** (signatures), not source IDs; four intersections: train↔val, train↔FinQA-test, train↔FiQA-test, train↔FinBen-FOMC-test.

---

### Cell 39 — [code] Exact leakage audit
- `record_input_sig(r)` / `sig_set(records)` — computes signature sets.
- Computes four pairwise intersections and adds an ERROR gate per pair (`Exact input leakage {name}`, expected 0).

---

### Cell 40 — [markdown] Near-duplicate audit
Explains MinHash-LSH: flags high-overlap train↔eval inputs but is an **audit only** (no automatic deletion), because financial templates can legitimately resemble each other. **In Phase-2.1, this audit is mandatory (on by default).**

---

### Cell 41 — [code] MinHash near-duplicate audit (mandatory)
- `word_shingles(text, n=3)` — 3-word shingles of normalized text.
- `duplicate_audit_text(record)` — uses `record["input"]` as the audit text.
- `run_minhash_audit(train, eval, threshold)` — `datasketch.MinHash` (96 perms) + `MinHashLSH`; for each eval record, queries the LSH for candidate train records, recomputes exact Jaccard on word-shingle sets, returns a ranked DataFrame.
- Saves full audit to `near_duplicate_train_vs_protected_test.csv`.

**Gate:** `Near-duplicate audit has no candidates >= 0.90` — WARNING (candidates require manual inspection; they are not deleted automatically).

---

### Cell 42 — [code] Protected-record ID isolation
- Computes record_id sets for train, validation, and protected test pools.
- Adds 6 gates:
  - `Train/Validation/Protected test record IDs unique` — ERROR.
  - `No protected record IDs in training/validation` — ERROR.
  - `Train/validation record IDs disjoint` — ERROR.

---

## Part F — Stable downstream artifacts

### Cell 43 — [markdown] Part F intro — Export
States that these exported files are the **only supported downstream dataset interface**.

---

### Cell 44 — [code] Export all JSONL files + FinTrust registry
- Writes: `train.jsonl`, `validation.jsonl`, `finqa_test.jsonl`, `fiqa_test.jsonl`, `finben_fomc_test.jsonl`.
- Builds **`fintrust_registry`**: dataset, `evaluation_only=True`, HF/GitHub pointers, 7 dimensions, snapshot path, evaluator-code path, file manifest, and the "do not convert to SFT" note. Saved as `eval/fintrust_registry.json`.

---

### Cell 45 — [code] `dataset_registry.json` + Phase-3/4 configs + source fingerprints
- `artifact_entry(path, role, dataset, task)` — path, role, dataset, task, **sha256**, and record count.
- **`source_fingerprints.json`** — SHA-256 hashes for FinQA raw files, FiQA canonical JSON, FinBen-FOMC canonical JSON, FinTrust per-file hashes.
- **`dataset_registry.json`**:
  - `training_policy`: allowed = `[FinQA, FiQA]`, forbidden = `[FinBen-FOMC, FinTrust]`.
  - `sampling_policy`: dataset-balanced default `{FinQA: 0.5, FiQA: 0.5}`.
  - `threshold_policy`: calibration on validation only; `never_tune_on` = all protected tests.
  - `artifacts`: train, validation, finqa_test, fiqa_test, finben_fomc_test, fintrust_registry, finqa_gold_execution_audit, near_duplicate_audit.
- **`phase3_data_config.json`**: pointers, allowed datasets, target format for both tasks (`PROGRAM:`/`ANSWER:` for FinQA, `SENTIMENT_SCORE:` for FiQA), task metrics.
- **`phase4_eval_config.json`**: calibration split, protected tests, recommended metrics per task, official evaluator paths, evaluation rules.

---

### Cell 46 — [code] Write `phase2_1_interface.py`
Writes a self-contained module with:
- `_read_jsonl`, `load_registry`, `load_artifact(registry_path, key)`.
- Convenience wrappers: `load_train`, `load_validation`, `load_finqa_test`, `load_fiqa_test`, `load_finben_fomc_test`, `load_fintrust_registry`.
- `filter_records(records, dataset, task)` — subset by dataset/task.
- `dataset_counts(records)` — summary counts.

---

## Part G — Final artifact-level validation

### Cell 47 — [markdown] Part G intro — Artifact invariants
Rationale: the final gates **re-read the exported files**, verifying the actual Phase-3/4 inputs, not just in-memory frames.

---

### Cell 48 — [code] Artifact-level gates (the final audit)
Re-opens all JSONL + the FinTrust registry and checks:

| Gate | Severity | Meaning |
|---|---|---|
| `Exported train count preserved` | ERROR | file count matches in-memory count |
| `Exported validation count preserved` | ERROR | same for validation |
| `FinQA test marked evaluation-only` | ERROR | every record has `evaluation_only=True` |
| `FiQA test marked evaluation-only` | ERROR | same for FiQA |
| `FinBen-FOMC test marked evaluation-only` | ERROR | all FinBen-FOMC records evaluation-only |
| `FinTrust registry marked evaluation-only` | ERROR | registry flag is `True` |
| `No exported protected test IDs in exported training` | ERROR | ID intersection is empty |
| `Export artifact hashes reproduce registry` | ERROR | sha256 of every file matches the registry |

---

### Cell 49 — [markdown] PASS/FAIL semantics
Defines: PASS = every ERROR-severity gate passes; WARNING gates don't block export.

---

### Cell 50 — [code] The final report
- Converts `GATES` to a DataFrame and displays it.
- `hard_failures = [g for g in GATES if severity=="ERROR" and not passed]`.
- `phase2_1_pass = len(hard_failures) == 0`.
- `ready_for_phase3` and `ready_for_phase4` checks.
- Writes **`phase2_1_validation_report.json`**: schema_version, pass/fail, ready flags, hard failures, warnings, full gates list, all artifact paths.
- Prints `PHASE-2.1 STATUS: PASS|FAIL`, hard-failure details, and the report path.

---

### Cell 51 — [markdown] Phase-3/4 handoff contract
- **Phase-3** reads `datasets/train.jsonl`, `datasets/validation.jsonl`, `manifests/phase3_data_config.json`.
  - Training pool: FinQA + FiQA only. Default config proposes dataset-balanced sampling.
  - Recommended experiments: zero-shot → few-shot → FinQA-only QLoRA → FiQA-only QLoRA → joint QLoRA → freeze before protected-test evaluation.
- **Phase-4** uses validation for calibration/threshold selection, protected tests for evaluation.
  - FinQA: official executor metrics.
  - FiQA: continuous regression (MSE, R², MAE, Pearson).
  - FinBen-FOMC: external test-only generalization.
  - FinTrust: official dimension-specific evaluation.
  - Calibration/abstention thresholds on validation only.
  - Near-duplicate candidates inspected and documented, not silently removed.

---

## 6. Complete integrity-gate ledger

| Cell | Gate | Severity |
|---|---|---|
| 15 | FinQA train count / dev count / test count | ERROR |
| 15 | FinQA IDs globally unique | ERROR |
| 15 | FinQA required field present: id, question, exe_ans, program, gold_inds, table | ERROR |
| 15 | FinQA final step matches an official answer representation | WARNING |
| 15 | FinQA program operation vocabulary recognized | ERROR |
| 15 | FinQA no exact full-input leakage across splits | ERROR |
| 18 | FinQA train: official gold programs execute successfully | ERROR |
| 18 | FinQA train: official execution equals qa.exe_ans | ERROR |
| 18 | FinQA dev: official gold programs execute successfully | ERROR |
| 18 | FinQA dev: official execution equals qa.exe_ans | ERROR |
| 18 | FinQA test: official gold programs execute successfully | ERROR |
| 18 | FinQA test: official execution equals qa.exe_ans | ERROR |
| 24 | FiQA train count / valid count / test count | ERROR |
| 24 | FiQA IDs globally unique | WARNING |
| 24 | FiQA sentence non-empty | ERROR |
| 24 | FiQA target entity non-empty | ERROR |
| 24 | FiQA aspect non-empty | ERROR |
| 24 | FiQA score numeric | ERROR |
| 24 | FiQA score bounded in [-1, 1] | ERROR |
| 24 | FiQA no exact task-input leakage across splits | ERROR |
| 29 | FinBen-FOMC source configured | ERROR |
| 29 | FinBen-FOMC test count | ERROR |
| 29 | FinBen-FOMC IDs globally unique | ERROR |
| 29 | FinBen-FOMC text non-empty | ERROR |
| 29 | FinBen-FOMC labels in expected vocabulary | ERROR |
| 29 | FinBen-FOMC choices[gold] consistent with answer | ERROR (WARNING if no gold col) |
| 31 | FinBen-FOMC contributes zero training examples | ERROR |
| 31 | FinBen-FOMC contributes zero validation examples | ERROR |
| 31 | FinBen-FOMC all public records are evaluation-only | ERROR |
| 35 | FinTrust all seven dimensions represented | ERROR |
| 35 | FinTrust data files load without parser errors | ERROR |
| 35 | FinTrust evaluator/postprocess candidates discovered for all dimensions | WARNING |
| 37 | Training datasets exactly FinQA + FiQA | ERROR |
| 37 | No evaluation_only records in training | ERROR |
| 37 | No evaluation_only records in validation | ERROR |
| 37 | No FinTrust records in training/validation | ERROR |
| 37 | No FinBen-FOMC records in training/validation | ERROR |
| 39 | Exact input leakage train_vs_validation | ERROR |
| 39 | Exact input leakage train_vs_finqa_test | ERROR |
| 39 | Exact input leakage train_vs_fiqa_test | ERROR |
| 39 | Exact input leakage train_vs_finben_fomc_test | ERROR |
| 41 | Near-duplicate audit has no candidates >= 0.90 | WARNING |
| 42 | Train record IDs unique | ERROR |
| 42 | Validation record IDs unique | ERROR |
| 42 | Protected test record IDs unique | ERROR |
| 42 | No protected record IDs in training | ERROR |
| 42 | No protected record IDs in validation | ERROR |
| 42 | Train/validation record IDs disjoint | ERROR |
| 48 | Exported train count preserved | ERROR |
| 48 | Exported validation count preserved | ERROR |
| 48 | FinQA test marked evaluation-only | ERROR |
| 48 | FiQA test marked evaluation-only | ERROR |
| 48 | FinBen-FOMC test marked evaluation-only | ERROR |
| 48 | FinTrust registry marked evaluation-only | ERROR |
| 48 | No exported protected test IDs in exported training | ERROR |
| 48 | Export artifact hashes reproduce registry | ERROR |

---

## 7. Key design decisions worth remembering

1. **FiQA replaces FinNLI.** FinNLI is unavailable; FiQA Sentiment provides native continuous sentiment scores in [-1,1] with aspect-level annotations. No discretization is applied — Phase-3 treats the score as a structured numeric target.
2. **Official executor validation.** Phase-2.1 is the first phase to run the official FinQA `evaluate.py` on every gold program. This catches program-annotation mismatches that the old `steps[-1].res` heuristic missed.
3. **FiQA duplicate-ID fix.** The upstream FiQA dataset has duplicate `_id` values within splits. The normalization step appends `_0`, `_1`, ... suffixes to guarantee unique `record_id` values.
4. **Mandatory near-duplicate audit.** Unlike Phase-2 (where the audit was off by default), Phase-2.1 runs MinHash-LSH over word shingles by default. Candidates are flagged for manual review, never silently deleted.
5. **The leakage unit is the model input signature**, not the source ID — so near-duplicate questions with different contexts are correctly *not* flagged, while identical full inputs are.
6. **`evaluation_only` lives in metadata of every record** and is propagated all the way into training gates (cell 37) and artifact gates (cell 48). It is the single flag that keeps protected data out of SFT.
7. **FinBen-FOMC is 100% evaluation.** There is no train/dev, so `finben_records["train"]`/`["dev"]` are hard-coded empty and the registry's task mix is `{FinQA: 0.5, FiQA: 0.5}`.
8. **FinTrust is metadata-normalized, not record-normalized.** This preserves the official per-dimension evaluators rather than forcing one answer column.
9. **Records are complete but gold-safe:** FinQA inputs contain full evidence but gold evidence appears only in `metadata`; the ANSWER target is the executable `exe_ans`, with the human `answer` retained separately.
10. **Source fingerprints.** Every dataset's SHA-256 is recorded in `source_fingerprints.json`, providing an immutable provenance trail from raw source to final export.
11. **Warnings do not fail the run**, but they are retained in the report — designed so an audit can be strict without blocking legitimate variations (e.g., absent gold columns, near-dup candidates, incomplete FinTrust evaluator coverage).

---

## 8. Differences from Phase-2 (`exp.md`)

| Aspect | Phase-2 | Phase-2.1 |
|---|---|---|
| **Training data** | FinQA only | FinQA + FiQA Sentiment |
| **Validation data** | FinQA dev only | FinQA dev + FiQA valid |
| **FinQA test records** | `finqa_test.jsonl` | `finqa_test.jsonl` (unchanged) |
| **FiQA test records** | Not present | `fiqa_test.jsonl` (new) |
| **FinQA gold validation** | `steps[-1].res` heuristic | Official `evaluate.py` executor |
| **Near-duplicate audit** | Optional (default off) | Mandatory (default on) |
| **FiQA ID dedup** | N/A | Suffix `_0`, `_1`, ... for duplicate `_id` values |
| **Schema version** | `trustfin-phase2-v1.0` | `trustfin-phase2.1-v1.0` |
| **Artifact root** | `artifacts/phase2/` | `artifacts/phase2_1/` |
| **Interface module** | `phase2_interface.py` | `phase2_1_interface.py` |
| **FinNLI** | Replaced by FinBen-FOMC in Phase-2 | Fully absent (replaced by FiQA Sentiment) |
| **Source fingerprints** | Not present | `source_fingerprints.json` with SHA-256 for all sources |
| **Sampling policy** | `{FinQA: 1.0}` | `{FinQA: 0.5, FiQA: 0.5}` dataset-balanced |

---

## 9. Failure semantics & troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `WARNING: HF_TOKEN is not set…` | FinBen-FOMC may be gated, token missing | `export HF_TOKEN=hf_...` then re-run |
| Cell 28 throws (network/gated) → many flagged FINBEN gates | `datasets` unavailable, no net, denied access | install deps, set token, verify network |
| `PHASE-2.1 STATUS: FAIL` with `FiQA {split} count` | HF repo schema changed, different split names | check `FIQA_SPLIT_MAP` and `FIQA_EXPECTED` |
| `PHASE-2.1 STATUS: FAIL` with `Record IDs unique` | Upstream duplicate IDs not properly deduped | check the dedup logic in `normalize_fiqa_frame` |
| `FinQA {split}: official execution equals qa.exe_ans` FAIL | Gold program doesn't reproduce `exe_ans` | inspect the specific mismatch in the audit CSV |
| Near-dup audit finds candidates | Legitimate financial boilerplate similarity | inspect manually; document in the paper; do not auto-delete |
| `FinTrust evaluator candidates` WARNING | Some dimensions lack official evaluator scripts | resolve in Phase-4 using the official GitHub repo |
| Stale report shows old `FinNLI` FAILs | Old report from a previous Phase-2 run | fully re-run from Cell 0; clean `artifacts/phase2_1/` |

---

*Generated for the trustfin phase-2.1 pipeline. Schema contract version: `trustfin-phase2.1-v1.0`.*
