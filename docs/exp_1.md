# Deep Explanation — `phase2.ipynb`

**TrustFin-SLM · Phase 2 · Data Integrity, Benchmark Isolation, and Unified Data Contract**

This document explains every cell of `/home/heet18/Downloads/Research Papers/phase2.ipynb` (47 cells, indices 0–46). It covers the purpose, logic, key variables/functions, and the integrity gates added by each cell, plus the artifacts the notebook writes and how the PASS/FAIL contract works.

---

## 1. What the notebook does (30-second version)

Phase-2 turns three benchmark sources into one **frozen, integrity-checked data contract** that downstream phases consume:

| Source | Role | Split policy |
|---|---|---|
| **FinQA** (`czyssrs/FinQA`) | Financial numerical reasoning, program+answer supervision | canonical `train`/`dev`/`test`; test is **protected** |
| **FinBen-FOMC** (`TheFinAI/finben-fomc`) | FOMC stance classification (`dovish`/`hawkish`/`neutral`) | single official `test` split → **entirely evaluation-only** |
| **FinTrust** (`HughieHu/FinTrust`) | 7-dimension trust benchmark | **evaluation-only**, kept as a registry (never flattened into SFT) |

Design philosophy:

1. **Never hand-edit gold.** Training targets are generated directly from official records.
2. **No privileged evidence.** Gold evidence is stored as annotation metadata, never put in the model input.
3. **No chain-of-thought fabrication.** Only the official answer formats are used (`PROGRAM:`/`ANSWER:`, or a label).
4. **No evaluation tuning.** Calibration/threshold selection is restricted to validation; all test + FinBen-FOMC + FinTrust are protected.
5. **Stable interface.** Downstream phases import `phase2_interface.py` and read JSONL/JSON artifacts instead of re-downloading benchmarks.

**Every integrity check is recorded as a `Gate`.** The run passes only if all `ERROR`-severity gates pass (`PHASE-2 STATUS: PASS`).

---

## 2. Data flow overview

```
FinQA (GitHub raw)  ──► raw download (+SHA-256 manifest)  ──► flatten → gates ──► records(train/dev/test) ──┐
FinBen-FOMC (HF)    ──► token-authenticated load          ──► normalize → gates ─► records(test, eval-only) ─┼─► pools → leak audit ─► export artifacts
FinTrust (HF+GitHub)──► snapshot + evaluator clone        ──► manifest + gates ─► registry (eval-only)      ─┘                            │
                                                                                                                                            ▼
        phase3_data_config.json · phase4_eval_config.json · dataset_registry.json · phase2_validation_report.json · phase2_interface.py ◄── gates (cell 45)
```

Output root: `PROJECT_ROOT = $TRUSTFIN_ROOT` or `./trustfin_slm_workspace`, then `artifacts/phase2/`.

---

## 3. Before running

- **Python deps:** `pandas`, `numpy`, `requests`, `datasets`, `huggingface_hub`, `pyarrow`, `tqdm` (cell 2 has the commented install line).
- **Hugging Face token:** FinBen datasets are gated → set `HF_TOKEN` in the environment before running. Cell 17 intentionally warns if it is missing, and the run ends in `FAIL` if the source cannot be loaded.
- **Optional:** `datasketch` for the near-duplicate audit (`RUN_OPTIONAL_NEAR_DUP_AUDIT = False` by default).

---

## 4. Directory layout produced

```
trustfin_slm_workspace/
├── data/raw/
│   ├── finqa/                  # train.json, dev.json, test.json (downloaded, unchanged)
│   ├── fintrust_hf/            # FinTrust HF snapshot (raw files untouched)
│   └── fintrust_eval_code/     # official postprocess_response.py evaluator repo
└── artifacts/phase2/
    ├── datasets/
    │   ├── train.jsonl         # SFT training (FinQA train only)
    │   └── validation.jsonl    # validation + calibration (FinQA dev only)
    ├── eval/
    │   ├── finqa_test.jsonl    # protected FinQA test
    │   ├── finben_fomc_test.jsonl   # protected FinBen-FOMC test
    │   └── fintrust_registry.json   # protected FinTrust registry
    ├── manifests/
    │   ├── dataset_registry.json
    │   ├── phase3_data_config.json
    │   ├── phase4_eval_config.json
    │   ├── phase2_validation_report.json
    │   └── near_duplicate_audit.csv      # only when audit is on
    └── python/
        └── phase2_interface.py # downstream loader helpers
```

---

## 5. Cell-by-cell walkthrough

Notation: **[markdown]** = documentation/explanation · **[code]** = executable Python.

---

### Cell 0 — [markdown] Title, purpose, design rules
- States the research context (Phase-3 → SLM baselines/QLoRA; Phase-4 → TrustFin-SLM: verification, calibration, selective prediction, FinTrust evaluation).
- Enumerates the five **design rules** that drive every later cell:
  1. FinQA canonical splits, full evidence preserved, no hand-edited targets.
  2. FinBen-FOMC as the protected FOMC stance benchmark (496-row single test split).
  3. FinTrust kept evaluation-only, not collapsed into one SFT-shaped table.
  4. Benchmark isolation: test / FinBen-FOMC / FinTrust excluded from SFT and threshold tuning.
  5. Downstream reads artifacts, not the notebook's loading code.
- Lists primary references and a "Change vs. previous notebooks" note (the older exploratory notebook loaded FinBen-FOMC in FinNLI's place and is now codified here).

---

### Cell 1 — [markdown] Environment note
Explains that the install cell is run once in a clean environment and that version pins are intentionally broad.

---

### Cell 2 — [code] (commented) dependency install
A commented `%pip install` line — uncomment only if the environment is missing packages. The required set: `pandas>=2.0`, `numpy>=1.24`, `requests>=2.31`, `datasets>=3.0`, `huggingface_hub>=0.25`, `pyarrow>=15`, `matplotlib>=3.8`, `scikit-learn>=1.4`, `tqdm>=4.66`.

---

### Cell 3 — [code] Imports, seed, environment probe
- `from __future__ import annotations`, standard library imports (`csv`, `hashlib`, `json`, `os`, `platform`, `random`, `re`, `shutil`, `subprocess`, `sys`, `time`, `Counter`, `defaultdict`, `@dataclass`, `Path`, typing helpers).
- Third-party: `numpy`, `pandas`, `requests`, `tqdm`.
- **Seed** `SEED = 42`, applied to `random` and `numpy`.
- Prints Python / platform / pandas versions → reproducibility breadcrumbs for the paper.

---

### Cell 4 — [markdown] Configuration overview
Documents the FinBen-FOMC track (labels, 496 rows, single test split → protected) and the `HF_TOKEN` requirement for gated FinBen datasets.

---

### Cell 5 — [code] Root paths, directories, and all constants
This is the **central configuration cell**. Everything downstream depends on these names:

- **Paths:**
  - `PROJECT_ROOT` — `$TRUSTFIN_ROOT` env var, default `./trustfin_slm_workspace`.
  - `RAW_DIR`, `ARTIFACT_DIR`, `EXPORT_DIR`, `EVAL_DIR`, `MANIFEST_DIR`, `MODULE_DIR` — all `mkdir(parents=True, exist_ok=True)`.
- **FinQA:** `FINQA_BASE_URL` (GitHub raw), `FINQA_EXPECTED = {train:6251, dev:883, test:1147}`.
- **FinBen-FOMC:** `FINBEN_FOMC_HF_REPO = "TheFinAI/finben-fomc"`, `FINBEN_FOMC_SPLIT = "test"`, `FINBEN_FOMC_EXPECTED = 496`, `FINBEN_FOMC_LABELS = ["dovish","hawkish","neutral"]`.
- **Auth:** `HF_TOKEN = os.environ.get("HF_TOKEN", "").strip()`.
- **FinTrust:** `FINTRUST_HF_REPO`, `FINTRUST_GITHUB`, and the seven official dimensions (`trustfulness, robustness, safety, fairness, privacy, transparency, knowledge_discovery`).
- **Integrity knobs:** `DOWNLOAD_TIMEOUT=90`, `FAIL_ON_CROSS_SPLIT_EXACT_DUPLICATE=True`, `RUN_OPTIONAL_NEAR_DUP_AUDIT=False`, `NEAR_DUP_THRESHOLD=0.90`.
- **Contract version:** `SCHEMA_VERSION = "trustfin-phase2-v1.0"` — stamped into every record and manifest.
- Prints the resolved root, repo, split, and whether a token is configured.

---

### Cell 6 — [markdown] Shared utility functions
Foreshadows that the helpers in cell 7 are reused by later phases via the exported module.

---

### Cell 7 — [code] **Serialization, hashing, I/O, and the Gate framework**
The workhorse of the notebook. Notable pieces:

- `json_default(x)` — converts NumPy/Path/NaN types into JSON-safe values.
- `canonical_json(obj)` — deterministic JSON (`sort_keys=True`, compact separators) → same object always serializes identically → safe input to hashing.
- `sha256_text`, `sha256_file` — content hashing for provenance manifests.
- `normalized_text(x)` — lowercase + collapse whitespace; the canonical form for leakage signatures.
- `input_signature(*parts)` — `sha256` of normalized parts joined by a sentinel. **This is the leakage-detection unit**: it fingerprints the *model input* (question+context+table / statement), not the source ID.
- `write_jsonl` / `read_jsonl` — stable record file I/O.
- `save_json`, `load_json_or_jsonl`, `load_tabular_file` — tolerant reading of JSON/JSONL/CSV/TSV/Parquet (used to inventory FinTrust).
- **`Gate` dataclass** (`name, passed, observed, expected, severity, note`) and the **`GATES` ledger list** with `add_gate(...)`. Every check in the notebook funnels through `add_gate`, which prints `[PASS]/[FAIL]/[WARN]` immediately and appends to `GATES`. The final report (cell 45) is built purely from this ledger.

---

### Cell 8 — [markdown] Part A — FinQA download intro
Notes that raw JSON files are kept **unchanged** and SHA-256 hashes are recorded for traceability.

---

### Cell 9 — [code] FinQA download + source manifest
- `FINQA_RAW_DIR` created.
- `download_file(url, path, overwrite=False)` — idempotent HTTP download with `DOWNLOAD_TIMEOUT`; `raise_for_status()` on failure.
- Downloads `train.json`/`dev.json`/`test.json` from the canonical GitHub release; paths stored in `finqa_raw_paths`.
- `finqa_source_manifest` — per split: absolute path, **sha256 of the raw file**, byte size. Displayed as a table.

---

### Cell 10 — [markdown] Parse FinQA without editing gold
States the key rule that will be enforced in cell 15: gold evidence (`gold_inds`) is annotation metadata only, never privileged test-time input.

---

### Cell 11 — [code] FinQA flattening into a DataFrame
- `finqa_table_to_text(table)` — renders the nested list-of-lists table as `row1 | col1 | col2 ...` lines.
- `finqa_gold_evidence_to_text(gold_inds)` — serializes gold evidence (dict/list).
- `flatten_finqa(raw_rows, split)` — unpacks each record: `pre_text`/`post_text`, `table`, `question`, `answer` (human annotation), `exe_ans` (executable numeric result), `program`/`program_re`, `steps`, `gold_inds`, `final_step_result` (last step's `res`), plus derived `context_text` (pre+post joined), `table_text`, `gold_evidence_text`.
- Builds `finqa_splits = {split: DataFrame}` and concatenates into `finqa_df`; prints per-split counts and a 2-row preview.

---

### Cell 12 — [markdown] FinQA integrity gates
Explains the gate list: source counts, ID separation, required gold fields, final-step/result consistency, and exact input leakage. Importantly: duplicate *question strings alone* are **informational** — the leakage signature includes question + context + table.

---

### Cell 13 — [code] FinQA integrity gates + diagnostics
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

- Also computes `input_signature` for every row, prints agreement rates (final-step vs answer / exe_ans) and duplicate-question counts, and **displays the mismatched final-step rows for inspection**.

---

### Cell 14 — [markdown] FinQA record design
Defines the record contract: input = full report text + table + question (no gold); target = structured `PROGRAM: ...\nANSWER: <exe_ans>`; the separate human `answer` is kept in metadata because it is not guaranteed identical to the executable result.

---

### Cell 15 — [code] FinQA canonical records
- `FINQA_SYSTEM` — system prompt for a financial reasoning assistant.
- `make_finqa_record(row)` — builds a **record** dict:
  - `record_id = "finqa::{id}"`, `source_id`, `dataset="FinQA"`, `task="financial_numerical_reasoning"`, `split`.
  - `system`, `instruction`, `input` (evidence text + table + question), `target` (`PROGRAM:`/`ANSWER:`).
  - `target_answer = exe_ans`, `gold_program`, `gold_evidence = gold_inds`.
  - `metadata`: `schema_version`, `has_table`, `annotated_answer` (human), `execution_answer` (executable), `input_signature`, and **`evaluation_only = (split == "test")`** — the first link in the no-leak chain.
- Builds `finqa_records = {split: [records]}`; prints counts and a pretty JSON sample.

---

### Cell 16 — [markdown] Part B — FinBen-FOMC intro
Notes the HF repo, token-authenticated loading, and the `id/query/answer/text/choices/gold` schema.

---

### Cell 17 — [code] FinBen-FOMC loading (requires `HF_TOKEN`)
- `load_finben_fomc_hf(repo_id, split)` — calls `datasets.load_dataset(repo_id, split=split, token=HF_TOKEN or None)` and converts the ArrowDatasetDict to pandas.
- **Warns (non-fatal) if `HF_TOKEN` is unset** — the dataset may be gated.
- Loads `finben_fomc_raw`; prints row count and columns.
- Builds `finben_source_manifest`: `mode="huggingface"`, repo, split, `rows`, and a `sha256` over the **canonical JSON** of all records sorted by `id` — a provenance fingerprint independent of file packaging.

> **Failure mode:** if this cell raises (missing `datasets`, no network, gated/denied), the notebook does *not* crash hard — cell 19 degrades the whole benchmark to `FINBEN_AVAILABLE = False` and the run ends in `FAIL` (see cell 44/45).

---

### Cell 18 — [markdown] Normalization contract
Defines the stable FOMC schema: `text → text`, `answer → label`, `choices`/`gold` preserved as metadata; labels lowercased to the three allowed values; `choices[gold] == answer` checked when gold exists.

---

### Cell 19 — [code] FinBen-FOMC normalization + gates
- `first_existing(columns, candidates)` — column-agnostic resolution, so the same normalize code works if HF renames columns.
- `normalize_fomc_label(x)` — lowercases and **rejects** anything outside the 3-vocabulary (no guessing label order).
- `normalize_fomc_frame(df, split)` — maps columns, coerces IDs to strings (synthesizes `fomc-00000` if absent), strips text, lowercases answers, keeps `query`, parses `choices` (list) and `gold` (nullable `Int64`), sets `label`, computes `input_signature`.
- `FINBEN_AVAILABLE` = raw source present and non-empty.

**When available, gates added (all ERROR unless noted):**

| Gate | Meaning |
|---|---|
| `FinBen-FOMC source configured` | split loaded with N rows |
| `FinBen-FOMC test count` | == 496 |
| `FinBen-FOMC IDs globally unique` | no id collisions |
| `FinBen-FOMC text non-empty` | every statement has text |
| `FinBen-FOMC labels in expected vocabulary` | exactly `{dovish, hawkish, neutral}` |
| `FinBen-FOMC choices[gold] consistent with answer` | `choices[gold] == answer` for all rows; downgraded to WARNING if the source has no gold column |

**When unavailable:** a single hard `FinBen-FOMC source configured = False` gate with the note that Phase-2 cannot pass until the repo is loadable with `HF_TOKEN`.

---

### Cell 20 — [markdown] Diagnostics
Justification for the lightweight stats: label balance + text length catch schema mistakes before protected evaluation.

---

### Cell 21 — [code] FinBen-FOMC diagnostics
- Computes `text_words` mean / 95th percentile, per-split `n`, `label_counts`.
- Displays a crosstab of split × label and a few sample `query` strings.

---

### Cell 22 — [markdown] Record policy
Documents that FOMC is **label-only supervision** and that, because there is a single official `test` split, **every** FinBen-FOMC record carries `evaluation_only = True`.

---

### Cell 23 — [code] FinBen-FOMC canonical records
- `FINBEN_SYSTEM` — FOMC stance classifier prompt.
- `make_finben_record(row)` — record dict with `record_id = "finben-fomc::{id}"`, `dataset="FinBen-FOMC"`, `task="fomc_stance"`, `input` = `Statement:\n{text}`, `target = target_label = label`, `metadata` holding `query`, `choices`, `gold_index`, `input_signature`, and **`evaluation_only = True` unconditionally**.
- `finben_records = {"train": [], "dev": [], "test": [...]}` — train/dev always empty by design; prints `{'train':0, 'dev':0, 'test':496}` when loaded.

---

### Cell 24 — [markdown] Part C — FinTrust intro
Explains why FinTrust is *not* converted to SFT: seven heterogeneous dimensions with task-specific schemas and official evaluators; Phase-2 builds a manifest instead.

---

### Cell 25 — [code] FinTrust snapshot + evaluator clone
- `normalize_dimension_name` — `-`/space → `_`, lowercase.
- `snapshot_fintrust_hf()` — `huggingface_hub.snapshot_download` into `FINTRUST_RAW_DIR`.
- `clone_or_update_fintrust_eval_repo()` — `git pull --ff-only` if already cloned, else `git clone --depth 1` the official GitHub repo; returns the path or `None` on failure (printed as a warning).
- Wraps the snapshot in try/except: on failure `FINTRUST_AVAILABLE = False` and it falls back to the (possibly empty) raw dir, so FinTrust failure never kills the run — it just becomes FAIL gates.

---

### Cell 26 — [markdown] Manifest intent
Only metadata is normalized; raw records stay untouched; Phase-4 pairs a chosen dimension/file with its official `postprocess_response.py`.

---

### Cell 27 — [code] FinTrust dimension-aware manifest
- `Infer dimension`: walks the file's relative path parts; returns the first of the seven dimensions found in path components; special-cases hyphenated `knowledge_discovery`.
- `classify_fintrust_file_role(name, path)`: `generated_or_result` (response/output/result/prediction), `candidate_eval_data` (prompt/test/sample/dataset/question/data), else `unknown`.
- `summarize_schema_from_records` — union of column names (capped at 200 rows).
- `TABULAR_EXT` — accepted extensions.
- Iterates every snapshot file (skipping `.cache`): loads it with `load_tabular_file` (tolerating failures into a `load_error` field), records `dimension`, paths, `role`, `rows`, `columns`, `sha256`, and **`evaluation_only=True`**.
- Displays the manifest table and a per-dimension raw-row total.

---

### Cell 28 — [markdown] FinTrust gate reasoning
Concise explanation of a subtle design choice: the repo claims 7 dimensions / 15,680 answer pairs, but raw file-row totals are **not** forced to equal 15,680 (schemas differ); instead the gates require all 7 dimensions represented and preserve official evaluators.

---

### Cell 29 — [code] FinTrust integrity gates
- `FinTrust all seven dimensions represented` — each of the 7 dimensions appears in the manifest.
- `FinTrust data files load without parser errors` — no `load_error` entries.
- (If snapshot failed) a single hard `FinTrust dataset snapshot available = False`.
- **Evaluator presence (WARNING, not blocking):** for each dimension, checks `postprocess_response.py` exists in `FINTRUST_EVAL_CODE_DIR/{dim or dim-with-dashes}`. The note explicitly forbids substituting a generic metric for the official evaluator.

---

### Cell 30 — [markdown] Part D — Pools
States the pool construction explicitly: train = FinQA train; validation = FinQA dev; protected = FinQA test + FinBen-FOMC + all FinTrust. FinBen-FOMC and FinTrust are **never** appended to the SFT pool.

---

### Cell 31 — [code] Build pools + first no-leak gates
- `train_records = finqa train (finben train = [])`, `validation_records = finqa dev (finben dev = [])`, plus the two protected test lists.
- Prints counts with per-dataset `Counter`.
- **Gates:**
  - `No FinTrust records in training`
  - `No evaluation_only records in training` — every training record must have `metadata.evaluation_only == False`.

---

### Cell 32 — [markdown] Leakage audit explanation
Emphasizes that leakage is measured on **normalized model inputs** (signatures), not source IDs; three intersections: train↔val, train↔FinQA-test, train↔FinBen-FOMC-test. (FinBen-FOMC has no official train, but the audit still guards against accidental inclusion.)

---

### Cell 33 — [code] Exact leakage audit
- `record_input_sig(r)` — prefers the stored `metadata.input_signature`, else recomputes from `r["input"]`.
- `sig_set(records)` — set of signatures.
- Computes the three pairwise intersections and adds an ERROR gate per pair (`Exact input leakage {name}`, expected 0). Returns the `leak_checks` dict for display.

---

### Cell 34 — [markdown] Near-duplicate audit
Explains MinHash-LSH: it flags high-overlap train↔eval inputs but is an **audit only** (no automatic deletion), because financial templates can legitimately resemble each other.

---

### Cell 35 — [code] Optional MinHash near-dup audit (default off)
- `char_ngrams(text, n=5)` — 5-grams of normalized text.
- `run_minhash_audit(train, eval, threshold=.90)` — `datasketch.MinHash` (128 perms) + `MinHashLSH`; for each eval record, queries the LSH for candidate train records, recomputes exact Jaccard on char-5-gram sets, returns a ranked DataFrame. Gracefully emits an install hint if `datasketch` is missing.
- Only executed when `RUN_OPTIONAL_NEAR_DUP_AUDIT = True`; results (if any) are written to a CSV in cell 37.

---

### Cell 36 — [markdown] Export contract
Lists the **only supported downstream interface files** — exactly the artifacts that cells 37/39/41 produce — to minimize split contamination as Phase-3/4 evolves.

---

### Cell 37 — [code] Write eval test files + FinTrust registry
- Writes `train.jsonl`, `validation.jsonl`, `finqa_test.jsonl`, `finben_fomc_test.jsonl` (note: `finben_test_records` → `finben_fomc_test.jsonl`).
- Builds **`fintrust_registry`**: dataset, `evaluation_only=True`, HF/GitHub pointers, the 7 dimensions, snapshot path, evaluator-code path, the full file manifest, and the explicit "do not convert to SFT" note. Saved as `eval/fintrust_registry.json`.
- Persists near-dup CSV if the audit produced rows.
- Prints the exported paths.

---

### Cell 38 — [markdown] Registry + config templates
Explains that the registry carries counts/hashes/tasks/split-roles, that Phase-3 may do dataset-balanced sampling from it, and Phase-4 must keep calibration thresholds on validation only.

---

### Cell 39 — [code] `dataset_registry.json` + Phase-3/4 configs
- `artifact_entry(path, role, dataset, task)` — path, role, dataset, task, **sha256**, and record count (JSONL only).
- **`dataset_registry.json`**:
  - `training_policy`: allowed = `[FinQA, FinBen-FOMC]`, forbidden = `[FinTrust]`, note that FinBen-FOMC is evaluation-only.
  - `threshold_policy`: calibration/abstention on validation only; `never_tune_on` = FinQA test, FinBen-FOMC, FinTrust.
  - `artifacts`: `train`, `validation`, `finqa_test`, `finben_fomc_test` (roles `sft_train` / `validation_and_calibration` / `protected_test`), and `fintrust_registry` (role `protected_trust_evaluation`).
  - `source_manifests`: FinQA (URLs+hashes), FinBen-FOMC (HF hashes), FinTrust (repos + dimensions).
- **`phase3_data_config.json`**: pointers to train/validation artifacts, task mix `{FinQA: 1.0}` with the note that no FOMC SFT data exists, and `target_format` for both datasets.
- **`phase4_eval_config.json`**: calibration split = validation, `protected_tests = [finqa_test, finben_fomc_test, fintrust_registry]`, recommended metrics per task (FinQA consistency metrics; FOMC accuracy/macro-F1/per-class-F1; Calibration ECE/Brier/NLL; Selective-Prediction coverage/selective-accuracy/AURC; FinTrust official metrics; Efficiency trackers), and official evaluator URLs.

---

### Cell 40 — [markdown] Interface export
Tells future notebooks to import `phase2_interface.py` rather than copying Phase-2 loading logic.

---

### Cell 41 — [code] Write `phase2_interface.py`
Writes, as a raw string, a small self-contained module:

- `_read_jsonl`, `load_registry`, `load_artifact(registry_path, key)` — resolves `key` against `registry["artifacts"]`; loads FinTrust as JSON, everything else as JSONL.
- Convenience wrappers: `load_train`, `load_validation`, `load_finqa_test`, `load_finben_fomc_test`, `load_fintrust_registry`.
- Prints the saved path.

---

### Cell 42 — [markdown] Artifact-level invariants
Rationale: the final gates **re-read the exported files**, verifying the actual Phase-3/4 inputs, not just in-memory frames.

---

### Cell 43 — [code] Artifact-level gates (the final audit)
Re-opens all four JSONL + the FinTrust registry and checks:

| Gate | Severity | Meaning |
|---|---|---|
| `Exported train has unique record IDs` | ERROR | no duplicate `record_id` on disk |
| `Exported validation has unique record IDs` | ERROR | same for validation |
| `FinQA test marked evaluation-only` | ERROR | every exported test record has `evaluation_only=True` |
| `FinBen-FOMC test marked evaluation-only` | ERROR | every FinBen-FOMC record is evaluation-only (falls back to `FINBEN_AVAILABLE` if the export is empty) |
| `FinTrust registry marked evaluation-only` | ERROR | registry flag is `True` |
| `No protected test record IDs in training` | ERROR | `train_ids ∩ (finqa_test ∪ finben_fomc_test) == ∅` |

> This cell is where the historical run reported `[FAIL] FinNLI test marked evaluation-only` — it is now the `FinBen-FOMC test marked evaluation-only` gate.

---

### Cell 44 — [markdown] PASS/FAIL semantics
Defines: PASS = every ERROR-severity gate passes; WARNING gates don't block export; if the FinBen-FOMC source is unconfigured (missing token/access), the notebook intentionally ends with `PHASE-2 STATUS: FAIL` so Phase-3 training can't start from an incomplete setup.

---

### Cell 45 — [code] The final report
- Converts `GATES` to a DataFrame and displays it.
- `hard_failures = [g for g in GATES if severity=="ERROR" and not passed]`; `phase2_pass = len(hard_failures)==0`.
- Writes **`phase2_validation_report.json`**: `schema_version`, `phase2_pass`, `hard_failures`, full `gates` list, registry path, interface path.
- Prints `PHASE-2 STATUS: PASS|FAIL`, hard-failure details (name/observed/expected), and the report path.

---

### Cell 46 — [markdown] Phase-3/4 handoff contract
- **Phase-3** reads `datasets/train.jsonl`, `datasets/validation.jsonl`, `manifests/phase3_data_config.json`; suggested experiments 1–4 (zero-shot → few-shot → FinQA QLoRA → joint, only if a curated FOMC subset is added).
- **Phase-4** uses validation for calibration/threshold selection, protected FinQA test for consistency, protected FinBen-FOMC for accuracy/Macro-F1/calibration, and the FinTrust registry + official evaluators.
- Explicit **non-goals**: no FinTrust training, no gold-evidence in test prompts, no fabricated CoT targets, no threshold tuning on test data.

---

## 6. Complete integrity-gate ledger

| Cell | Gate | Severity |
|---|---|---|
| 13 | FinQA train count / dev count / test count | ERROR |
| 13 | FinQA IDs globally unique | ERROR |
| 13 | FinQA required field present: id, question, exe_ans, program, gold_inds, table | ERROR |
| 13 | FinQA final step matches an official answer representation | WARNING |
| 13 | FinQA program operation vocabulary recognized | ERROR |
| 13 | FinQA no exact full-input leakage across splits | ERROR |
| 19 | FinBen-FOMC source configured | ERROR |
| 19 | FinBen-FOMC test count | ERROR |
| 19 | FinBen-FOMC IDs globally unique | ERROR |
| 19 | FinBen-FOMC text non-empty | ERROR |
| 19 | FinBen-FOMC labels in expected vocabulary | ERROR |
| 19 | FinBen-FOMC choices[gold] consistent with answer | ERROR (WARNING if no gold col) |
| 29 | FinTrust all seven dimensions represented | ERROR |
| 29 | FinTrust data files load without parser errors | ERROR |
| 29 | FinTrust official postprocess scripts found for seven dimensions | WARNING |
| 29 | FinTrust dataset snapshot available / evaluator repo available | ERROR/WARNING on failure |
| 31 | No FinTrust records in training | ERROR |
| 31 | No evaluation_only records in training | ERROR |
| 33 | Exact input leakage train_vs_validation | ERROR |
| 33 | Exact input leakage train_vs_finqa_test | ERROR |
| 33 | Exact input leakage train_vs_finben_fomc_test | ERROR |
| 43 | Exported train has unique record IDs | ERROR |
| 43 | Exported validation has unique record IDs | ERROR |
| 43 | FinQA test marked evaluation-only | ERROR |
| 43 | FinBen-FOMC test marked evaluation-only | ERROR |
| 43 | FinTrust registry marked evaluation-only | ERROR |
| 43 | No protected test record IDs in training | ERROR |

---

## 7. Key design decisions worth remembering

1. **The leakage unit is the model input signature**, not the source ID — so near-duplicate questions with different contexts are correctly *not* flagged, while identical full inputs are.
2. **`evaluation_only` lives in metadata of every record** and is propagated all the way into training gates (cell 31) and artifact gates (cell 43). It is the single flag that keeps protected data out of SFT.
3. **FinBen-FOMC is 100% evaluation.** There is no train/dev, so `finben_records["train"]`/`["dev"]` are hard-coded empty and the registry's task mix is `{FinQA: 1.0}`.
4. **FinTrust is metadata-normalized, not record-normalized.** This preserves the official per-dimension evaluators (`postprocess_response.py`) rather than forcing one answer column.
5. **Records are complete but gold-safe:** FinQA inputs contain full evidence but gold evidence appears only in `metadata`; the ANSWER target is the executable `exe_ans`, with the human `answer` retained separately.
6. **Warnings do not fail the run**, but they are retained in the report — designed so an audit can be strict without blocking legitimate variations (e.g., absent gold columns, optional near-dup scan).

---

## 8. Failure semantics & troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `WARNING: HF_TOKEN is not set…` | FinBen-FOMC may be gated, token missing | `export HF_TOKEN=hf_...` then re-run |
| Cell 17 throws (network/gated) → many flagged FINBEN gates | `datasets` unavailable, no net, denied access | install deps, set token, verify network; run from a clean env |
| `PHASE-2 STATUS: FAIL` with `FinBen-FOMC source configured` | The evaluation benchmark was never loaded | whatever cell 17 reported — fix token/access first |
| Optional dup audit prints "Install datasketch…" | `RUN_OPTIONAL_NEAR_DUP_AUDIT=True` but `datasketch` missing | `pip install datasketch` |
| Stale report shows old `FinNLI` FAILs | `phase2_validation_report.json` written by an older run, or an editor saved an old copy of the notebook | fully re-run from Cell 0 → reload notebook in editor first |
| Editor overwriting the converted notebook | A stale tab holds the pre-conversion version | close/reload `phase2.ipynb` in Jupyter/VSCode, then re-run |

---

*Generated for the trustfin phase-2 pipeline. Schema contract version: `trustfin-phase2-v1.0`.*