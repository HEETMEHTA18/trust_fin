# Phase-4B → Phase-4C execution

**Session ID:** ses_e91940613ffe8SK2Rwuv8YaiCn
**Created:** 10/24/2026, 8:28:52 PM
**Updated:** 10/24/2026, 10:20:58 PM

---

## User

execute final experiments in sequence: Phase-4B → Phase-4C. check the flow an run this notebooks '/home/heet18/Downloads/Research Papers/notebooks/Phase4A_TrustFin_Execution_Verification_and_Financial_Grounding.ipynb''/home/heet18/Downloads/Research Papers/notebooks/Phase4C_TrustFin_Ablation_Statistical_Evaluation.ipynb'

---

## Assistant

**Tool: read**

**Input:**
```json
{
  "path": "/home/heet18/Downloads/Research Papers"
}
```

Read directory /home/heet18/Downloads/Research Papers, entries 1-21
.git/
.pytest_cache/
artifacts/
data/
deliverables/
docs/
notebooks/
paper/
papers/
Papers/
phases/
tools/
trust_fin_extra_phase/
trustfin_slm_workspace/
.dockerignore
.gitignore
AGENTS.md
Dockerfile
Makefile
Phase3_TrustFin_SLM_Baselines_and_Failure_Analysis.ipynb
README.md

**Tool: glob**

**Input:**
```json
{
  "pattern": "**/Phase4*.ipynb"
}
```

/home/heet18/Downloads/Research Papers/notebooks/Phase4C_TrustFin_Ablation_Statistical_Evaluation.ipynb
/home/heet18/Downloads/Research Papers/notebooks/Phase4A_TrustFin_Execution_Verification_and_Financial_Grounding.ipynb
/home/heet18/Downloads/Research Papers/notebooks/Phase4A_TrustFin_Execution_Verification_and_Financial_Grounding_executed.ipynb
/home/heet18/Downloads/Research Papers/notebooks/Phase4A_smoke_executed.ipynb
/home/heet18/Downloads/Research Papers/notebooks/Phase4B_TrustFin_Calibration_and_Selective_Prediction.ipynb
/home/heet18/Downloads/Research Papers/trust_fin_extra_phase/notebooks/Phase4_full_CAL.ipynb
/home/heet18/Downloads/Research Papers/trust_fin_extra_phase/notebooks/Phase4_smoke_EVG.ipynb
/home/heet18/Downloads/Research Papers/trust_fin_extra_phase/notebooks/Phase4_2_Evaluation_Improvement.ipynb
/home/heet18/Downloads/Research Papers/trust_fin_extra_phase/notebooks/Phase4_TrustFin_Proposed_Method.ipynb
/home/heet18/Downloads/Research Papers/trust_fin_extra_phase/notebooks/Phase4_smoke_CAL.ipynb
/home/heet18/Downloads/Research Papers/trust_fin_extra_phase/notebooks/Phase4_full_EVG.ipynb
/home/heet18/Downloads/Research Papers/trust_fin_extra_phase/notebooks/Phase4_smoke_E1.ipynb
/home/heet18/Downloads/Research Papers/trust_fin_extra_phase/notebooks/Phase4_full_TEST.ipynb

**Tool: glob**

**Input:**
```json
{
  "pattern": "**/phase4*.py"
}
```

/home/heet18/Downloads/Research Papers/tools/phase4a_configure.py
/home/heet18/Downloads/Research Papers/trust_fin_extra_phase/tools/phase4_shard.py

---

## Assistant

**Tool: read**

**Input:**
```json
{
  "path": "/home/heet18/Downloads/Research Papers/notebooks/Phase4B_TrustFin_Calibration_and_Selective_Prediction.ipynb"
}
```

Read file /home/heet18/Downloads/Research Papers/notebooks/Phase4B_TrustFin_Calibration_and_Selective_Prediction.ipynb, lines 1-538
1: {
2:  "cells": [
3:   {
4:    "cell_type": "markdown",
5:    "id": "f76b8000",
6:    "metadata": {},
7:    "source": [
8:     "# Phase-4B — TrustFin-SLM Verifier-Aware Confidence Calibration and Selective Prediction\n",
9:     "\n",
10:     "**Research objective:** The frozen E5 Qwen3-1.7B joint QLoRA model can generate executable FinQA programs, but executable programs can still be wrong. Phase-4A showed that correcting the answer by executing a program improves FinQA validation answer accuracy from **37.6% to 67.5%**. Raw model likelihood has only weak discrimination of correctness (~0.585 ROC-AUC on the designated calibration-fit records). We must test whether a modest **verifier-aware risk estimator** can better predict *when execution is trustworthy* and selectively abstain.\n",
11:     "\n",
12:     "**Important:** This is *validation-development work*, not a new SLM fine-tuning run or protected test evaluation. All comparisons in this notebook are explicitly validation-only. FinTrust scoring is outside Phase-4B.\n",
13:     "\n",
14:     "| ID | Experiment | Output |\n",
15:     "|---|---|---|\n",
16:     "| B1 | Raw generated-token likelihood | Confidence baseline and risk–coverage curve |\n",
17:     "| B2 | Likelihood-only Platt/logistic model | Calibration improvement versus B1 |\n",
18:     "| B3 | Verifier-aware logistic model | Likelihood + prompt length/truncation + syntactic/executable validity |\n",
19:     "| B4 | Grounding-aware logistic model | B3 + recomputed numerical grounding features |\n",
20:     "| B5 | Selective prediction | Fit-split coverage thresholds and holdout risk/coverage |\n",
21:     "| B6 | FiQA regression error/uncertainty | Predict *absolute error*, selective MAE, empirical intervals |\n",
22:     "| B7 | Reliability / AUC / Brier / ECE / AURC | Fit-OOF and holdout diagnostics |\n",
23:     "| B8 | Phase-4B freeze | Fit-only model selection, immutable feature policy and test-safe handoff |\n",
24:     "\n",
25:     "**Research safeguards:** The model's permitted predictors are explicitly whitelisted. `record_id` and `calibration_fold` serve only for alignment; gold answers, correctness labels, and errors are NEVER predictors. Hyperparameters and the winning policy are selected **from fit-split out-of-fold predictions**, before any holdout metrics are displayed. Holdout results are descriptive only; never reselect after inspecting them."
26:    ]
27:   },
28:   {
29:    "cell_type": "markdown",
30:    "id": "1f00874a",
31:    "metadata": {},
32:    "source": [
33:     "## How to run (students)\n",
34:     "\n",
35:     "1. Obtain the successful Phase-4A package `Phase4A_Review_Package(1).zip` from the previous phase, or preserve its original `phase4b_interface/` and `labels/` folders.\n",
36:     "2. Put the ZIP **beside this notebook**, or set `PHASE4A_ZIP` / `PHASE4A_DIR` in the configuration cell. The notebook automatically reads the original Phase-4A schema and checks IDs, folds, counts, feature leakage and source fingerprints.\n",
37:     "3. Python 3.10+, `numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib`, `joblib`, `nbformat` are sufficient. **No GPU required.** Run the optional installation cell only if dependencies are missing.\n",
38:     "4. First set `RUN_MODE=\"smoke\"` and execute **Kernel → Restart & Run All**. Smoke mode runs B1–B8 with fewer calibration-fit rows but **does not use the holdout** and **cannot freeze** the model.\n",
39:     "5. Change to `RUN_MODE=\"full\"` and restart/run all. The complete run uses 698 FinQA fit / 185 holdout and 92 FiQA fit / 25 holdout, with 5-fold (FinQA) and 4-fold (FiQA) OOF diagnostics. The defaults are CPU-friendly.\n",
40:     "6. The notebook writes `artifacts/phase4b_full/`, including a `Phase4B_Review_Package.zip`. Do not manually edit `phase4b_freeze.json`, fit features or the holdout results.\n",
41:     "7. **Run Phase-4C only after full-mode Phase-4B passes.** Phase-4C consumes the frozen B models/predictions; it does not tune on holdout.\n",
42:     "\n",
43:     "**What to share:** Executed notebook, `Phase4B_Review_Package.zip`, `phase4b_freeze.json`, `phase4b_readiness.json`, `B7_model_diagnostics.csv`, `B5_risk_coverage.csv`, `B6_fiqa_uncertainty.csv`. Do **not** upload private financial source documents or checkpoint weights unless specifically necessary."
44:    ]
45:   },
46:   {
47:    "cell_type": "code",
48:    "execution_count": null,
49:    "id": "1b6b8ba8",
50:    "metadata": {},
51:    "outputs": [],
52:    "source": [
53:     "INSTALL_DEPS = False\n",
54:     "if INSTALL_DEPS:\n",
55:     "    import sys, subprocess\n",
56:     "    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-U',\n",
57:     "        'numpy', 'pandas', 'scipy', 'scikit-learn', 'matplotlib', 'joblib', 'nbformat'])\n",
58:     "    print('Restart the kernel once after installation.')\n",
59:     "else:\n",
60:     "    print('Using the current Python environment.')"
61:    ]
62:   },
63:   {
64:    "cell_type": "markdown",
65:    "id": "5bf8aa4e",
66:    "metadata": {},
67:    "source": [
68:     "## 1. Configuration, imports, deterministic execution"
69:    ]
70:   },
71:   {
72:    "cell_type": "code",
73:    "execution_count": null,
74:    "id": "3f9b8a48",
75:    "metadata": {},
76:    "outputs": [],
77:    "source": [
78:     "import os, io, json, zipfile, hashlib, time, platform, warnings\n",
79:     "from pathlib import Path\n",
80:     "from collections import Counter\n",
81:     "import numpy as np, pandas as pd, matplotlib.pyplot as plt\n",
82:     "from scipy.stats import spearmanr\n",
83:     "from sklearn.base import clone\n",
84:     "from sklearn.pipeline import Pipeline\n",
85:     "from sklearn.impute import SimpleImputer\n",
86:     "from sklearn.preprocessing import StandardScaler\n",
87:     "from sklearn.linear_model import LogisticRegression, Ridge\n",
88:     "from sklearn.model_selection import StratifiedKFold, KFold, cross_val_predict\n",
89:     "from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss, mean_absolute_error, mean_squared_error\n",
90:     "import joblib\n",
91:     "\n",
92:     "RUN_MODE = os.environ.get('TRUSTFIN_RUN_MODE','full').lower()  # 'smoke' or 'full'\n",
93:     "SEED = 42\n",
94:     "WORKSPACE_ROOT = Path(os.environ.get('TRUSTFIN_WORKSPACE',str(Path.cwd()))).expanduser().resolve()\n",
95:     "PHASE4A_ZIP = os.environ.get('TRUSTFIN_PHASE4A_ZIP','')\n",
96:     "PHASE4A_DIR = os.environ.get('TRUSTFIN_PHASE4A_DIR','')\n",
97:     "OUTPUT_ROOT = WORKSPACE_ROOT/'artifacts'/('phase4b_full' if RUN_MODE=='full' else 'phase4b_smoke')\n",
98:     "for child in ['tables','predictions','models','plots','manifests','reviews']:\n",
99:     "    (OUTPUT_ROOT/child).mkdir(parents=True,exist_ok=True)\n",
100:     "assert RUN_MODE in {'smoke','full'}\n",
101:     "rng=np.random.default_rng(SEED)\n",
102:     "print('Phase-4B', RUN_MODE, 'Workspace:', WORKSPACE_ROOT)"
103:    ]
104:   },
105:   {
106:    "cell_type": "markdown",
107:    "id": "d8cc83fd",
108:    "metadata": {},
109:    "source": [
110:     "## 2. Read the exact Phase-4A data contract (ZIP or extracted folder)"
111:    ]
112:   },
113:   {
114:    "cell_type": "code",
115:    "execution_count": null,
116:    "id": "7f8cc8a9",
117:    "metadata": {},
118:    "outputs": [],
119:    "source": [
120:     "REQUIRED = {\n",
121:     "    'finqa_features':'phase4b_interface/phase4b_finqa_features.jsonl',\n",
122:     "    'fiqa_features':'phase4b_interface/phase4b_fiqa_features.jsonl',\n",
123:     "    'finqa_labels':'labels/phase4b_finqa_labels.jsonl',\n",
124:     "    'fiqa_labels':'labels/phase4b_fiqa_labels.jsonl',\n",
125:     "    'policy':'phase4b_interface/phase4b_feature_policy.json',\n",
126:     "    'readiness':'manifests/phase4a_readiness.json',\n",
127:     "}\n",
128:     "def sha(raw):return hashlib.sha256(raw).hexdigest()\n",
129:     "def locate_source():\n",
130:     "    if PHASE4A_DIR:\n",
131:     "        p=Path(PHASE4A_DIR).expanduser()\n",
132:     "        if all((p/v).is_file() for v in REQUIRED.values()):return 'directory',p\n",
133:     "    if PHASE4A_ZIP:\n",
134:     "        p=Path(PHASE4A_ZIP).expanduser()\n",
135:     "        if p.is_file():return 'zip',p\n",
136:     "    for p in [WORKSPACE_ROOT/'Phase4A_Review_Package(1).zip',WORKSPACE_ROOT/'Phase4A_Review_Package.zip',\n",
137:     "              Path.cwd()/'Phase4A_Review_Package(1).zip',Path.cwd()/'Phase4A_Review_Package.zip',\n",
138:     "              Path('/mnt/data/Phase4A_Review_Package(1).zip')]:\n",
139:     "        if p.is_file():return 'zip',p\n",
140:     "    for p in [WORKSPACE_ROOT/'artifacts/phase4a',WORKSPACE_ROOT/'phase4a_user_review',Path('/mnt/data/phase4a_user_review')]:\n",
141:     "        if all((p/v).is_file() for v in REQUIRED.values()):return 'directory',p\n",
142:     "    raise FileNotFoundError('Phase-4A inputs missing: set TRUSTFIN_PHASE4A_ZIP or TRUSTFIN_PHASE4A_DIR')\n",
143:     "SOURCE_KIND,SOURCE_PATH=locate_source()\n",
144:     "SOURCE_HASHES={}\n",
145:     "def read_source(key):\n",
146:     "    rel=REQUIRED[key]\n",
147:     "    if SOURCE_KIND=='directory': raw=(SOURCE_PATH/rel).read_bytes()\n",
148:     "    else:\n",
149:     "        with zipfile.ZipFile(SOURCE_PATH) as z:\n",
150:     "            matches=[n for n in z.namelist() if n==rel or n.endswith('/'+rel)]\n",
151:     "            if len(matches)!=1:raise ValueError(f'Exactly one {rel} is required; found {matches}')\n",
152:     "            raw=z.read(matches[0])\n",
153:     "    SOURCE_HASHES[key]=sha(raw)\n",
154:     "    return raw\n",
155:     "def load_jsonl(k):return pd.DataFrame([json.loads(line) for line in read_source(k).decode().splitlines() if line.strip()])\n",
156:     "def save_json(name,obj):\n",
157:     "    p=OUTPUT_ROOT/'manifests'/name\n",
158:     "    p.write_text(json.dumps(obj,indent=2,default=str),encoding='utf-8');return p\n",
159:     "policy=json.loads(read_source('policy'))\n",
160:     "phase4a_readiness=json.loads(read_source('readiness'))\n",
161:     "print('Source:',SOURCE_KIND,SOURCE_PATH)\n",
162:     "print('Phase4A readiness:', phase4a_readiness.get('core_phase4a_status'),\n",
163:     "      phase4a_readiness.get('research_full_grounding_status'))\n",
164:     "assert phase4a_readiness.get('core_phase4a_status')=='PASS'"
165:    ]
166:   },
167:   {
168:    "cell_type": "markdown",
169:    "id": "47a19138",
170:    "metadata": {},
171:    "source": [
172:     "## 3. Split integrity, predictors whitelist, and label-leakage firewall"
173:    ]
174:   },
175:   {
176:    "cell_type": "code",
177:    "execution_count": null,
178:    "id": "9f373142",
179:    "metadata": {},
180:    "outputs": [],
181:    "source": [
182:     "F=load_jsonl('finqa_features'); FL=load_jsonl('finqa_labels')\n",
183:     "Q=load_jsonl('fiqa_features'); QL=load_jsonl('fiqa_labels')\n",
184:     "def validate_pair(X,Y,dataset,n):\n",
185:     "    assert len(X)==len(Y)==n, (dataset,len(X),len(Y))\n",
186:     "    assert X.record_id.is_unique and Y.record_id.is_unique\n",
187:     "    assert set(X.record_id)==set(Y.record_id)\n",
188:     "    assert set(X.calibration_fold)==set(Y.calibration_fold)=={'calibration_fit','calibration_holdout'}\n",
189:     "    # One-to-one joins only; do not bring gold labels into the predictor frame.\n",
190:     "    cross=X[['record_id','calibration_fold']].merge(Y[['record_id','calibration_fold']],on='record_id',validate='one_to_one',suffixes=('_X','_Y'))\n",
191:     "    assert (cross.calibration_fold_X==cross.calibration_fold_Y).all()\n",
192:     "    assert (X.dataset==dataset).all()\n",
193:     "validate_pair(F,FL,'FinQA',883);validate_pair(Q,QL,'FiQA',117)\n",
194:     "assert Counter(F.calibration_fold)=={'calibration_fit':698,'calibration_holdout':185}\n",
195:     "assert Counter(Q.calibration_fold)=={'calibration_fit':92,'calibration_holdout':25}\n",
196:     "forbidden=set(policy['forbidden_feature_names']) | {'dataset','calibration_fold','record_id', 'mean_token_logprob'}\n",
197:     "for field in ['execution_correct','written_correct','corrected_correct','gold_sentiment','absolute_error']:\n",
198:     "    assert field not in F.columns and field not in Q.columns\n",
199:     "FINQA_SETS={\n",
200:     "   'B1_raw':['source_likelihood'],\n",
201:     "   'B2_likelihood':['source_likelihood'],\n",
202:     "   'B3_verifier':['source_likelihood','prompt_tokens','prompt_truncation_observed','official_executable_feature','program_structural_valid'],\n",
203:     "   'B4_grounding':['source_likelihood','prompt_tokens','prompt_truncation_observed','official_executable_feature','program_structural_valid',\n",
204:     "                   'any_literal_unmatched','unmatched_literal_count','literal_presence_fraction','max_question_local_overlap'],\n",
205:     "}\n",
206:     "FIQA_FEATURES=['source_likelihood','generated_tokens','prompt_tokens','sentiment_pred_clipped','score_near_boundary','parse_ok']\n",
207:     "for name, features in [*FINQA_SETS.items(),('FiQA',FIQA_FEATURES)]:\n",
208:     "    df=Q if name=='FiQA' else F\n",
209:     "    assert set(features).issubset(df.columns), (name,set(features)-set(df.columns))\n",
210:     "    assert not (set(features)&forbidden), (name,set(features)&forbidden)\n",
211:     "    assert all(pd.api.types.is_numeric_dtype(df[f]) for f in features)\n",
212:     "assert phase4a_readiness.get('research_full_grounding_status')=='PASS', 'Grounding must be recomputed by Phase-4A before B4.'\n",
213:     "print('PASS: feature/label separation; 698/185 and 92/25 protected calibration splits.')"
214:    ]
215:   },
216:   {
217:    "cell_type": "markdown",
218:    "id": "aed0bd65",
219:    "metadata": {},
220:    "source": [
221:     "## 4. Sampling and evaluation utilities (only fit split may select models)"
222:    ]
223:   },
224:   {
225:    "cell_type": "code",
226:    "execution_count": null,
227:    "id": "997a2396",
228:    "metadata": {},
229:    "outputs": [],
230:    "source": [
231:     "Xf=F.set_index('record_id',drop=False); Yf=FL.set_index('record_id',drop=False)\n",
232:     "Xq=Q.set_index('record_id',drop=False); Yq=QL.set_index('record_id',drop=False)\n",
233:     "yf=Yf.reindex(Xf.index)['execution_correct'].astype(int)\n",
234:     "yq=Yq.reindex(Xq.index)['absolute_error'].astype(float)\n",
235:     "f_fit=(Xf.calibration_fold=='calibration_fit'); f_hold=~f_fit\n",
236:     "q_fit=(Xq.calibration_fold=='calibration_fit'); q_hold=~q_fit\n",
237:     "if RUN_MODE=='smoke':\n",
238:     "    # Smoke only uses calibration_fit, never the real calibration_holdout.\n",
239:     "    original_f_ids=Xf[f_fit].sample(frac=1,random_state=SEED).index\n",
240:     "    original_q_ids=Xq[q_fit].sample(frac=1,random_state=SEED).index\n",
241:     "    f_fit=Xf.index.isin(original_f_ids[:120])\n",
242:     "    f_hold=Xf.index.isin(original_f_ids[120:150])\n",
243:     "    q_fit=Xq.index.isin(original_q_ids[:48])\n",
244:     "    q_hold=Xq.index.isin(original_q_ids[48:60])\n",
245:     "def logistic(features):\n",
246:     "    return Pipeline([('impute',SimpleImputer(strategy='median',add_indicator=False)),\n",
247:     "        ('scale',StandardScaler()),('model',LogisticRegression(C=1.0,max_iter=1200,random_state=SEED))])\n",
248:     "def regression(features):\n",
249:     "    return Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler()),('model',Ridge(alpha=10.0))])\n",
250:     "def roc(y,p):return float(roc_auc_score(y,p)) if len(np.unique(y))>1 else float('nan')\n",
251:     "def ece(y,p,bins=10):\n",
252:     "    y=np.asarray(y);p=np.clip(np.asarray(p),0,1)\n",
253:     "    edges=np.linspace(0,1,bins+1);err=0.\n",
254:     "    for i in range(bins):\n",
255:     "        sel=(p>=edges[i]) & ((p<edges[i+1]) if i<bins-1 else (p<=edges[i+1]))\n",
256:     "        if sel.any(): err+=sel.mean()*abs(y[sel].mean()-p[sel].mean())\n",
257:     "    return float(err)\n",
258:     "def rc_curve(correct,score):\n",
259:     "    correct=np.asarray(correct,dtype=float);score=np.asarray(score,dtype=float)\n",
260:     "    score=np.nan_to_num(score,nan=-1e10); order=np.argsort(-score,kind='stable')\n",
261:     "    err=1-correct[order]; cover=np.arange(1,len(err)+1)/len(err)\n",
262:     "    risk=np.cumsum(err)/np.arange(1,len(err)+1)\n",
263:     "    return cover,risk\n",
264:     "def aurc(correct,score):\n",
265:     "    c,r=rc_curve(correct,score)\n",
266:     "    return float(np.mean(r)) # discrete coverage-integrated risk\n",
267:     "def classification_metrics(y,p):\n",
268:     "    y=np.asarray(y).astype(int);p=np.clip(np.asarray(p,dtype=float),1e-6,1-1e-6)\n",
269:     "    return {'n':len(y),'roc_auc':roc(y,p),'brier':float(brier_score_loss(y,p)),\n",
270:     "            'ece10':ece(y,p),'nll':float(log_loss(y,p,labels=[0,1])),\n",
271:     "            'aurc':aurc(y,p)}\n",
272:     "def fit_threshold(scores,goal):\n",
273:     "    scores=np.asarray(scores,dtype=float)\n",
274:     "    # Conservative cutoff computed from fit predictions only. Ties can alter actual coverage.\n",
275:     "    return float(np.quantile(scores,1-goal))\n",
276:     "def selective_stats(correct,score,threshold,valid_mask=None):\n",
277:     "    correct=np.asarray(correct,dtype=int);score=np.asarray(score,dtype=float)\n",
278:     "    accept=np.isfinite(score)&(score>=threshold)\n",
279:     "    if valid_mask is not None: accept &= np.asarray(valid_mask,dtype=bool)\n",
280:     "    return {'n':len(correct),'accepted':int(accept.sum()),'coverage':float(accept.mean()),\n",
281:     "            'selective_risk':float(np.mean(1-correct[accept])) if accept.any() else float('nan'),\n",
282:     "            'selective_accuracy':float(np.mean(correct[accept])) if accept.any() else float('nan')},accept\n",
283:     "print('FinQA fit',int(f_fit.sum()),'holdout',int(f_hold.sum()),\n",
284:     "      '| FiQA fit',int(q_fit.sum()),'holdout',int(q_hold.sum()))"
285:    ]
286:   },
287:   {
288:    "cell_type": "markdown",
289:    "id": "36dc43cb",
290:    "metadata": {},
291:    "source": [
292:     "## 5. B1–B4: Out-of-fold development; **freeze selection before holdout**"
293:    ]
294:   },
295:   {
296:    "cell_type": "code",
297:    "execution_count": null,
298:    "id": "5761083f",
299:    "metadata": {},
300:    "outputs": [],
301:    "source": [
302:     "fit_X=Xf.loc[f_fit];fit_y=yf.loc[f_fit].to_numpy()\n",
303:     "folds=StratifiedKFold(n_splits=5 if RUN_MODE=='full' else 3,shuffle=True,random_state=SEED)\n",
304:     "models={};fit_scores={};dev_rows=[]\n",
305:     "for name,feat in FINQA_SETS.items():\n",
306:     "    if name=='B1_raw':\n",
307:     "        prob=np.clip(fit_X['source_likelihood'].to_numpy(dtype=float),1e-6,1-1e-6)\n",
308:     "        models[name]=None\n",
309:     "    else:\n",
310:     "        mdl=logistic(feat)\n",
311:     "        prob=cross_val_predict(mdl,fit_X[feat],fit_y,cv=folds,method='predict_proba')[:,1]\n",
312:     "        mdl.fit(fit_X[feat],fit_y)\n",
313:     "        models[name]=mdl\n",
314:     "    fit_scores[name]=prob\n",
315:     "    m=classification_metrics(fit_y,prob)\n",
316:     "    dev_rows.append({'method':name,'partition':'fit_oof',**m,'n_features':len(feat)})\n",
317:     "dev=pd.DataFrame(dev_rows).sort_values(['aurc','brier','method'])\n",
318:     "dev.to_csv(OUTPUT_ROOT/'tables/B1_B4_fit_oof_comparison.csv',index=False)\n",
319:     "display(dev)\n",
320:     "# PRE-COMMITTED selection -- strictly fit-OOF AURC; no holdout results seen.\n",
321:     "WINNER=str(dev.iloc[0]['method'])\n",
322:     "TARGET_COVERAGE=0.80\n",
323:     "thresholds={name:fit_threshold(p,TARGET_COVERAGE) for name,p in fit_scores.items()}\n",
324:     "FROZEN_SELECTION={'winner':WINNER,'criteria':'minimum fit-OOF AURC, Brier as tie-breaker',\n",
325:     "    'target_coverage':TARGET_COVERAGE,'thresholds_from_fit_oof':thresholds,\n",
326:     "    'feature_sets':FINQA_SETS,'no_label_features':True,'fit_sample_count':len(fit_y),\n",
327:     "    'seed':SEED,'run_mode':RUN_MODE,'holdout_not_used_for_selection':True,\n",
328:     "    'grounding_source':phase4a_readiness.get('research_full_grounding_status')}\n",
329:     "save_json('B8_fit_only_selection.json',FROZEN_SELECTION)\n",
330:     "for name,mdl in models.items():\n",
331:     "    if mdl is not None:joblib.dump(mdl,OUTPUT_ROOT/'models'/f'{name}.joblib')\n",
332:     "print('FROZEN FROM FIT OOF BEFORE HOLDOUT:',WINNER, 'target coverage:',TARGET_COVERAGE)"
333:    ]
334:   },
335:   {
336:    "cell_type": "markdown",
337:    "id": "7bb34273",
338:    "metadata": {},
339:    "source": [
340:     "## 6. B5, B7: One-time FinQA holdout evaluation, reliability and selective curves"
341:    ]
342:   },
343:   {
344:    "cell_type": "code",
345:    "execution_count": null,
346:    "id": "455a4fe5",
347:    "metadata": {},
348:    "outputs": [],
349:    "source": [
350:     "hold_X=Xf.loc[f_hold];hold_y=yf.loc[f_hold].to_numpy()\n",
351:     "hold_rows=[]; pred_rows=[];curve_rows=[]\n",
352:     "for name,feat in FINQA_SETS.items():\n",
353:     "    if name=='B1_raw': hp=np.clip(hold_X.source_likelihood.to_numpy(dtype=float),1e-6,1-1e-6)\n",
354:     "    else: hp=models[name].predict_proba(hold_X[feat])[:,1]\n",
355:     "    pred_rows.extend([{'record_id':rid,'dataset':'FinQA','calibration_fold':'calibration_holdout',\n",
356:     "         'method':name,'p_correct':float(p)} for rid,p in zip(hold_X.index,hp)])\n",
357:     "    hold_rows.append({'method':name,'partition':'holdout',**classification_metrics(hold_y,hp)})\n",
358:     "    # Fixed fit-side thresholds at target coverages; empirical holdout coverages may vary.\n",
359:     "    for coverage in [0.50,0.70,0.80,0.90,1.00]:\n",
360:     "        cutoff=fit_threshold(fit_scores[name],coverage) if coverage<1 else float('-inf')\n",
361:     "        hard_gate=(hold_X['official_executable_feature'].astype(bool)&\n",
362:     "                   hold_X['program_structural_valid'].astype(bool)).to_numpy()\n",
363:     "        stats,accepted=selective_stats(hold_y,hp,cutoff,hard_gate)\n",
364:     "        curve_rows.append({'method':name,'partition':'holdout','requested_coverage':coverage,\n",
365:     "                           'frozen_fit_threshold':cutoff,**stats})\n",
366:     "B7=pd.concat([dev,pd.DataFrame(hold_rows)],ignore_index=True,sort=False)\n",
367:     "B7.to_csv(OUTPUT_ROOT/'tables/B7_model_diagnostics.csv',index=False)\n",
368:     "PD=pd.DataFrame(pred_rows);PD.to_csv(OUTPUT_ROOT/'predictions/B1_B4_finqa_holdout_scores.csv',index=False)\n",
369:     "B5=pd.DataFrame(curve_rows)\n",
370:     "B5.to_csv(OUTPUT_ROOT/'tables/B5_risk_coverage.csv',index=False)\n",
371:     "display(B7);display(B5.query('requested_coverage==0.8'))\n",
372:     "fig,ax=plt.subplots(figsize=(8,5))\n",
373:     "for name,g in B5.groupby('method'):\n",
374:     "    ax.plot(g.coverage,g.selective_risk,marker='o',label=name)\n",
375:     "ax.set(xlabel='Actual accepted fraction on holdout',ylabel='Error among accepted',\n",
376:     "       title='FinQA selective risk, thresholds frozen on fit OOF')\n",
377:     "ax.legend(fontsize=8);fig.tight_layout();fig.savefig(OUTPUT_ROOT/'plots/B5_finqa_risk_coverage.png',dpi=180);plt.show()"
378:    ]
379:   },
380:   {
381:    "cell_type": "markdown",
382:    "id": "779266e1",
383:    "metadata": {},
384:    "source": [
385:     "## 7. B6: FiQA-specific error-risk estimation and *empirical* uncertainty"
386:    ]
387:   },
388:   {
389:    "cell_type": "code",
390:    "execution_count": null,
391:    "id": "d4e4f218",
392:    "metadata": {},
393:    "outputs": [],
394:    "source": [
395:     "qX=Xq.loc[q_fit]; qy=yq.loc[q_fit].to_numpy(dtype=float)\n",
396:     "q_cv=KFold(n_splits=4 if RUN_MODE=='full' else 3,shuffle=True,random_state=SEED)\n",
397:     "fiqa_model=regression(FIQA_FEATURES)\n",
398:     "q_oof=cross_val_predict(fiqa_model,qX[FIQA_FEATURES],qy,cv=q_cv,method='predict')\n",
399:     "q_oof=np.maximum(q_oof,0)\n",
400:     "fiqa_model.fit(qX[FIQA_FEATURES],qy)\n",
401:     "qH=Xq.loc[q_hold]; qY=yq.loc[q_hold].to_numpy(dtype=float)\n",
402:     "q_hat=np.maximum(fiqa_model.predict(qH[FIQA_FEATURES]),0)\n",
403:     "q_base=np.repeat(float(np.mean(qy)),len(qY))\n",
404:     "q_likelihood=qH['source_likelihood'].to_numpy(dtype=float)\n",
405:     "q_methods={'B6_constant':-q_base,'B6_likelihood':q_likelihood,'B6_ridge_error':-q_hat}\n",
406:     "q_goal=0.8\n",
407:     "q_risks=[];q_pred=[]\n",
408:     "fit_risk={'B6_constant':-np.repeat(float(np.mean(qy)),len(qy)),\n",
409:     "          'B6_likelihood':qX['source_likelihood'].to_numpy(dtype=float),\n",
410:     "          'B6_ridge_error':-q_oof}\n",
411:     "for name,score in q_methods.items():\n",
412:     "    thresh=fit_threshold(fit_risk[name],q_goal)\n",
413:     "    ok=np.isfinite(score)&(score>=thresh)\n",
414:     "    ok &= qH['parse_ok'].to_numpy(dtype=int)==1\n",
415:     "    q_risks.append({'method':name,'partition':'holdout','requested_coverage':q_goal,\n",
416:     "                    'coverage':float(ok.mean()),'accepted':int(ok.sum()),\n",
417:     "                    'selective_mae':float(np.mean(qY[ok])) if ok.any() else float('nan'),\n",
418:     "                    'full_mae':float(np.mean(qY)),'spearman_score_negative_error':float(spearmanr(score,-qY).statistic)})\n",
419:     "    for rid,s,pred_e,accept in zip(qH.index,score,q_hat,ok):\n",
420:     "        q_pred.append({'record_id':rid,'method':name,'uncertainty_score':float(s),\n",
421:     "                       'expected_absolute_error':float(pred_e),'accept_80pct':bool(accept)})\n",
422:     "# Descriptive error interval: quantile of fit-side OOF absolute errors. This is NOT formal conformal coverage.\n",
423:     "q_radius=float(np.quantile(qy,0.90))\n",
424:     "q_predscore=qH['sentiment_pred_clipped'].to_numpy(dtype=float)\n",
425:     "ygold=qH['sentiment_pred_clipped'].to_numpy(dtype=float) - 0 # true score stays in labels\n",
426:     "q_gold=QL.set_index('record_id').loc[qH.index,'gold_sentiment'].to_numpy(dtype=float)\n",
427:     "interval_lo=np.clip(q_predscore-q_radius,-1,1);interval_hi=np.clip(q_predscore+q_radius,-1,1)\n",
428:     "empirical_coverage=float(np.mean((q_gold>=interval_lo)&(q_gold<=interval_hi)))\n",
429:     "B6=pd.DataFrame(q_risks);B6.to_csv(OUTPUT_ROOT/'tables/B6_fiqa_uncertainty.csv',index=False)\n",
430:     "pd.DataFrame(q_pred).to_csv(OUTPUT_ROOT/'predictions/B6_fiqa_holdout_scores.csv',index=False)\n",
431:     "joblib.dump(fiqa_model,OUTPUT_ROOT/'models/B6_fiqa_error_ridge.joblib')\n",
432:     "FIQA_FREEZE={'features':FIQA_FEATURES,'alpha':10.0,'target':'absolute_error',\n",
433:     "             'fit_oof_90th_percentile_absolute_error':q_radius,\n",
434:     "             'holdout_empirical_interval_coverage':empirical_coverage,\n",
435:     "             'interval_guarantee':'none; descriptive fit OOF quantile, not formal conformal prediction',\n",
436:     "             'target_coverage':q_goal,'fit_thresholds':{k:fit_threshold(v,q_goal) for k,v in fit_risk.items()}}\n",
437:     "save_json('B6_fiqa_policy.json',FIQA_FREEZE)\n",
438:     "display(B6)\n",
439:     "print('FiQA empirical interval coverage (diagnostic only):',round(empirical_coverage,3))"
440:    ]
441:   },
442:   {
443:    "cell_type": "markdown",
444:    "id": "c76f97e8",
445:    "metadata": {},
446:    "source": [
447:     "## 8. B8 freeze, regression tests and downstream Phase-4C interface"
448:    ]
449:   },
450:   {
451:    "cell_type": "code",
452:    "execution_count": null,
453:    "id": "e03d97a4",
454:    "metadata": {},
455:    "outputs": [],
456:    "source": [
457:     "GATES={\n",
458:     "   'phase4a_core_pass':phase4a_readiness.get('core_phase4a_status')=='PASS',\n",
459:     "   'phase4a_grounding_recomputed':phase4a_readiness.get('research_full_grounding_status')=='PASS',\n",
460:     "   'finqa_record_alignment':len(F)==len(FL)==883 and Xf.index.is_unique,\n",
461:     "   'fiqa_record_alignment':len(Q)==len(QL)==117 and Xq.index.is_unique,\n",
462:     "   'fit_holdout_fixed':Counter(F.calibration_fold)=={'calibration_fit':698,'calibration_holdout':185}\n",
463:     "                          and Counter(Q.calibration_fold)=={'calibration_fit':92,'calibration_holdout':25},\n",
464:     "   'fit_only_winner_frozen':(OUTPUT_ROOT/'manifests/B8_fit_only_selection.json').exists(),\n",
465:     "   'no_gold_in_features':all(not (set(x)&forbidden) for x in FINQA_SETS.values())\n",
466:     "                          and not(set(FIQA_FEATURES)&forbidden),\n",
467:     "   'all_B1_to_B8_outputs_present':all((OUTPUT_ROOT/sub/name).exists() for sub,name in [\n",
468:     "      ('tables','B1_B4_fit_oof_comparison.csv'),('tables','B7_model_diagnostics.csv'),\n",
469:     "      ('tables','B5_risk_coverage.csv'),('tables','B6_fiqa_uncertainty.csv'),\n",
470:     "      ('predictions','B1_B4_finqa_holdout_scores.csv'),('predictions','B6_fiqa_holdout_scores.csv'),\n",
471:     "      ('manifests','B6_fiqa_policy.json'),('manifests','B8_fit_only_selection.json')]),\n",
472:     "}\n",
473:     "passed=bool(all(GATES.values()))\n",
474:     "readiness={'phase':'Phase-4B','run_mode':RUN_MODE,\n",
475:     "    'status':'PASS_FROZEN' if passed and RUN_MODE=='full' else 'SMOKE_NOT_FROZEN' if passed else 'FAIL',\n",
476:     "    'gates':GATES,'fit_selection_winner':WINNER,\n",
477:     "    'holdout_never_used_to_select_winner':True,\n",
478:     "    'protected_test_sets_used':False,\n",
479:     "    'limitations':['The 185/25 holdouts are internal validation, not protected test sets.',\n",
480:     "                   'FiQA interval is descriptive; do not claim formal conformal guarantee.',\n",
481:     "                   'Source-based numerical presence is not semantic grounding.',\n",
482:     "                   'Do not retune based on these holdout outputs.']}\n",
483:     "save_json('phase4b_readiness.json',readiness)\n",
484:     "freeze={'schema':'trustfin.phase4b.v1','status':readiness['status'],\n",
485:     "    'source_phase4a':str(SOURCE_PATH),'source_sha256':sha(SOURCE_PATH.read_bytes()) if SOURCE_KIND=='zip' else None,\n",
486:     "    'source_component_hashes':SOURCE_HASHES,'winner_from_fit_oof':WINNER,\n",
487:     "    'fit_only_selection':FROZEN_SELECTION,'fiqa_uncertainty_policy':FIQA_FREEZE,\n",
488:     "    'seed':SEED,'models':{k:str((OUTPUT_ROOT/'models'/f'{k}.joblib').resolve()) if k!='B1_raw' else None for k in FINQA_SETS},\n",
489:     "    'reports_relative_to_output_root':{'holdout_scores':'predictions/B1_B4_finqa_holdout_scores.csv',\n",
490:     "                                     'fiqa_scores':'predictions/B6_fiqa_holdout_scores.csv',\n",
491:     "                                     'holdout_metrics':'tables/B7_model_diagnostics.csv',\n",
492:     "                                     'risk_coverage':'tables/B5_risk_coverage.csv'},\n",
493:     "    'holdout_was_not_used_for_model_or_threshold_selection':True}\n",
494:     "save_json('phase4b_freeze.json',freeze)\n",
495:     "# Data-agnostic production feature contract: no gold answers or validation-only correctness labels.\n",
496:     "save_json('phase4c_handoff.json',{'schema':'trustfin.phase4c.v1','phase4b_root':str(OUTPUT_ROOT),\n",
497:     "     'phase4b_freeze':'manifests/phase4b_freeze.json',\n",
498:     "     'phase4a_source':str(SOURCE_PATH),'source_kind':SOURCE_KIND,\n",
499:     "     'finqa_feature_sets':FINQA_SETS,'fiqa_features':FIQA_FEATURES,\n",
500:     "     'no_protected_test':True})\n",
501:     "import zipfile\n",
502:     "pkg=OUTPUT_ROOT/'Phase4B_Review_Package.zip'\n",
503:     "with zipfile.ZipFile(pkg,'w',zipfile.ZIP_DEFLATED) as z:\n",
504:     "    for file in sorted(OUTPUT_ROOT.rglob('*')):\n",
505:     "        if file.is_file() and file!=pkg and file.suffix!='.joblib':\n",
506:     "            z.write(file,file.relative_to(OUTPUT_ROOT))\n",
507:     "print('STATUS:',readiness['status'],'Winner:',WINNER,'Package:',pkg)\n",
508:     "assert passed, GATES"
509:    ]
510:   },
511:   {
512:    "cell_type": "markdown",
513:    "id": "d10ea332",
514:    "metadata": {},
515:    "source": [
516:     "## Phase-4B: outputs to share before Phase-4C research review\n",
517:     "\n",
518:     "**Mandatory:** executed `.ipynb`, `Phase4B_Review_Package.zip`, `manifests/phase4b_readiness.json`, `manifests/phase4b_freeze.json`, `tables/B1_B4_fit_oof_comparison.csv`, `tables/B7_model_diagnostics.csv`, `tables/B5_risk_coverage.csv`, and `tables/B6_fiqa_uncertainty.csv`.\n",
519:     "\n",
520:     "**Keep locally for future protected-test inference:** `models/B2_likelihood.joblib`, `models/B3_verifier.joblib`, `models/B4_grounding.joblib`, `models/B6_fiqa_error_ridge.joblib`, plus the frozen Phase-3 E5 checkpoint. Only load serialized models from trusted sources.\n",
521:     "\n",
522:     "**Research decision:** A Phase-4B `PASS_FROZEN` status means the calibration protocol executed and selection was fit-only. It **does not** guarantee the chosen risk model improved on likelihood-only. Interpret results before claiming added novelty."
523:    ]
524:   }
525:  ],
526:  "metadata": {
527:   "kernelspec": {
528:    "display_name": "Python 3",
529:    "language": "python",
530:    "name": "python3"
531:   },
532:   "language_info": {
533:    "name": "python"
534:   }
535:  },
536:  "nbformat": 4,
537:  "nbformat_minor": 5
538: }

**Tool: read**

**Input:**
```json
{
  "path": "/home/heet18/Downloads/Research Papers/notebooks/Phase4C_TrustFin_Ablation_Statistical_Evaluation.ipynb"
}
```

Read file /home/heet18/Downloads/Research Papers/notebooks/Phase4C_TrustFin_Ablation_Statistical_Evaluation.ipynb, lines 1-475
1: {
2:  "cells": [
3:   {
4:    "cell_type": "markdown",
5:    "id": "aacc84e7",
6:    "metadata": {},
7:    "source": [
8:     "# Phase-4C — TrustFin-SLM Ablations, Statistical Evaluation and Methodology Freeze\n",
9:     "\n",
10:     "**Why this phase?** Phase-4A established that deterministic execution beats copying the SLM's written answer, but validity and number-presence do not prove correct financial reasoning. Phase-4B fits a risk estimator on calibration-fit examples, freezing model/threshold selection before evaluating on a held-out subset of the Phase-2.1 validation data. Phase-4C asks: **Does the full system actually improve *selective reliability* over execution-only and simple likelihood selection?**\n",
11:     "\n",
12:     "**Study limitations:** Phase-4C is a **validation-level study** on the same 185/25 calibration holdouts already summarized by Phase-4B; it does not create a second independent test set. Do not change features or thresholds after viewing holdout outcomes. Protected FinQA/FiQA tests and FinTrust dimensions are not accessed here.\n",
13:     "\n",
14:     "| Experiment | Analysis |\n",
15:     "|---|---|\n",
16:     "| C1 | Frozen E5 generated-answer baseline |\n",
17:     "| C2 | Execution-only corrected-answer baseline |\n",
18:     "| C3 | Hard program/evidence-grounding checks |\n",
19:     "| C4 | Likelihood-only abstention (B1/B2) |\n",
20:     "| C5 | Frozen verifier-aware abstention (B3/B4 and fit-only winner) |\n",
21:     "| C6 | Full TrustFin-SLM selective answering policy |\n",
22:     "| C7 | Grounding/verification/calibration component ablations |\n",
23:     "| C8 | Paired significance, bootstrap CIs, coverage-standardized comparisons |\n",
24:     "| C9 | Efficiency and honest offline-overhead reporting |\n",
25:     "| C10 | Methodology freeze and paper-ready exports |"
26:    ]
27:   },
28:   {
29:    "cell_type": "markdown",
30:    "id": "c8388917",
31:    "metadata": {},
32:    "source": [
33:     "## How to run\n",
34:     "\n",
35:     "1. Complete Phase-4B in **full mode** and confirm `phase4b_readiness.json` = `PASS_FROZEN`. Never run Phase-4C with a smoke-stage B freeze.\n",
36:     "2. Keep `Phase4A_Review_Package(1).zip` and **Phase-3 deliverable ZIP** available for original predictions and latency metadata. No GPU or training required.\n",
37:     "3. Place this notebook next to Phase-4B's notebook. By default it reads `artifacts/phase4b_full/` and auto-locates the two ZIPs. Otherwise set `TRUSTFIN_PHASE4B_ROOT`, `TRUSTFIN_PHASE4A_ZIP`, `TRUSTFIN_PHASE3_ZIP`, and optionally `TRUSTFIN_WORKSPACE`.\n",
38:     "4. Run **Kernel → Restart & Run All**. The C1–C10 analyses use only fixed Phase-4B holdout scores and the original E5 saved predictions; all thresholds are imported from fit-only Phase-4B metadata.\n",
39:     "5. Review the final `phase4c_readiness.json`. A methodological **PASS** does not prove the proposed method is better; report actual effects and CIs even when null or unfavorable.\n",
40:     "6. Share the **executed notebook**, `Phase4C_Review_Package.zip`, `phase4c_readiness.json`, `phase4_methodology_freeze.json`, and core C1–C9 CSV reports. Preserve E5/Phase-4B models locally.\n",
41:     "\n",
42:     "**Avoid:** using validation holdout labels to change the chosen risk estimator, setting new thresholds on holdout, or describing validation-only effects as test-set results."
43:    ]
44:   },
45:   {
46:    "cell_type": "code",
47:    "execution_count": null,
48:    "id": "3506f2c7",
49:    "metadata": {},
50:    "outputs": [],
51:    "source": [
52:     "import os, json, hashlib, zipfile, io, platform, time\n",
53:     "from pathlib import Path\n",
54:     "import numpy as np, pandas as pd, matplotlib.pyplot as plt\n",
55:     "from scipy.stats import binomtest\n",
56:     "from sklearn.metrics import roc_auc_score\n",
57:     "SEED=42\n",
58:     "WORKSPACE_ROOT=Path(os.environ.get('TRUSTFIN_WORKSPACE',str(Path.cwd()))).expanduser().resolve()\n",
59:     "B_ROOT=Path(os.environ.get('TRUSTFIN_PHASE4B_ROOT',str(WORKSPACE_ROOT/'artifacts/phase4b_full')))\n",
60:     "OUT=WORKSPACE_ROOT/'artifacts/phase4c_full'\n",
61:     "for sub in ['tables','plots','manifests','predictions']:(OUT/sub).mkdir(parents=True,exist_ok=True)\n",
62:     "assert (B_ROOT/'manifests/phase4b_readiness.json').is_file(),f'Phase-4B full output missing: {B_ROOT}'\n",
63:     "def load(path):return json.loads(Path(path).read_text(encoding='utf-8'))\n",
64:     "def save(name,obj):\n",
65:     "    p=OUT/'manifests'/name\n",
66:     "    p.write_text(json.dumps(obj,indent=2,default=str),encoding='utf-8');return p\n",
67:     "B_READY=load(B_ROOT/'manifests/phase4b_readiness.json')\n",
68:     "B_FREEZE=load(B_ROOT/'manifests/phase4b_freeze.json')\n",
69:     "assert B_READY['status']=='PASS_FROZEN' and B_FREEZE['status']=='PASS_FROZEN'\n",
70:     "print('Loaded frozen Phase-4B winner:',B_FREEZE['winner_from_fit_oof'])"
71:    ]
72:   },
73:   {
74:    "cell_type": "markdown",
75:    "id": "57abee41",
76:    "metadata": {},
77:    "source": [
78:     "## 1. Read Phase-4A/Phase-3 inputs and check record alignment"
79:    ]
80:   },
81:   {
82:    "cell_type": "code",
83:    "execution_count": null,
84:    "id": "2e04b6de",
85:    "metadata": {},
86:    "outputs": [],
87:    "source": [
88:     "def locate_zip(env,variants):\n",
89:     "    q=os.environ.get(env,'')\n",
90:     "    if q and Path(q).is_file():return Path(q)\n",
91:     "    for name in variants:\n",
92:     "        for root in [WORKSPACE_ROOT,Path.cwd(),Path('/mnt/data')]:\n",
93:     "            p=root/name\n",
94:     "            if p.is_file():return p\n",
95:     "    raise FileNotFoundError(f'Set {env}: expected one of {variants}')\n",
96:     "A_ZIP=locate_zip('TRUSTFIN_PHASE4A_ZIP',['Phase4A_Review_Package(1).zip','Phase4A_Review_Package.zip'])\n",
97:     "if B_FREEZE.get('source_sha256'):\n",
98:     "    assert hashlib.sha256(A_ZIP.read_bytes()).hexdigest()==B_FREEZE['source_sha256'], 'Phase-4A ZIP differs from the Phase-4B frozen source' \n",
99:     "P3_ZIP=locate_zip('TRUSTFIN_PHASE3_ZIP',['Phase3_TrustFin_Deliverable.zip'])\n",
100:     "def read_zip_bytes(zp,suffix):\n",
101:     "    with zipfile.ZipFile(zp) as z:\n",
102:     "        found=[n for n in z.namelist() if n==suffix or n.endswith('/'+suffix)]\n",
103:     "        if len(found)!=1:raise ValueError((zp,suffix,found))\n",
104:     "        return z.read(found[0])\n",
105:     "def zip_csv(zp,suffix):return pd.read_csv(io.BytesIO(read_zip_bytes(zp,suffix)))\n",
106:     "def zip_jsonl(zp,suffix):\n",
107:     "    return pd.DataFrame([json.loads(x) for x in read_zip_bytes(zp,suffix).decode().splitlines() if x.strip()])\n",
108:     "E5=zip_csv(P3_ZIP,'predictions/E5.csv')\n",
109:     "F=zip_jsonl(A_ZIP,'phase4b_interface/phase4b_finqa_features.jsonl')\n",
110:     "FL=zip_jsonl(A_ZIP,'labels/phase4b_finqa_labels.jsonl')\n",
111:     "Q=zip_jsonl(A_ZIP,'phase4b_interface/phase4b_fiqa_features.jsonl')\n",
112:     "QL=zip_jsonl(A_ZIP,'labels/phase4b_fiqa_labels.jsonl')\n",
113:     "Bscores=pd.read_csv(B_ROOT/'predictions/B1_B4_finqa_holdout_scores.csv')\n",
114:     "Bfiqa=pd.read_csv(B_ROOT/'predictions/B6_fiqa_holdout_scores.csv')\n",
115:     "fitmeta=load(B_ROOT/'manifests/B8_fit_only_selection.json')\n",
116:     "assert fitmeta['holdout_not_used_for_selection']\n",
117:     "assert E5.record_id.is_unique and len(E5)==1000\n",
118:     "assert FL.record_id.is_unique and F.record_id.is_unique\n",
119:     "assert set(FL.record_id)==set(F.record_id)\n",
120:     "assert set(QL.record_id)==set(Q.record_id)\n",
121:     "Fin=F.merge(FL,on=['record_id','calibration_fold'],validate='one_to_one')\n",
122:     "Fin=Fin.merge(E5[E5.dataset=='FinQA'][['record_id','answer_correct','execution_correct','program_executable','latency_seconds']],\n",
123:     "              on='record_id',validate='one_to_one',suffixes=('','_phase3'))\n",
124:     "assert len(Fin)==883\n",
125:     "Hold=Fin[Fin.calibration_fold=='calibration_holdout'].copy()\n",
126:     "assert len(Hold)==185\n",
127:     "Fi=Q.merge(QL,on=['record_id','calibration_fold'],validate='one_to_one')\n",
128:     "assert len(Fi)==117 and len(Fi[Fi.calibration_fold=='calibration_holdout'])==25\n",
129:     "assert set(Bscores.record_id)==set(Hold.record_id)\n",
130:     "assert Bscores.groupby('method').record_id.nunique().eq(185).all()\n",
131:     "print('Aligned E5 all 1000; FinQA holdout 185; FiQA holdout 25.')"
132:    ]
133:   },
134:   {
135:    "cell_type": "markdown",
136:    "id": "70ebd46b",
137:    "metadata": {},
138:    "source": [
139:     "## 2. C1/C2 — Raw text answer versus deterministic executed answer"
140:    ]
141:   },
142:   {
143:    "cell_type": "code",
144:    "execution_count": null,
145:    "id": "1d3940f2",
146:    "metadata": {},
147:    "outputs": [],
148:    "source": [
149:     "def boolcol(s):return s.replace({'True':True,'False':False}).fillna(False).astype(bool)\n",
150:     "# Phase4A independently generated gold correctness labels; results are validation-only.\n",
151:     "raw=boolcol(Fin.written_correct); executed=boolcol(Fin.execution_correct)\n",
152:     "cont=pd.crosstab(raw,executed)\n",
153:     "all_raw=raw.mean();all_exec=executed.mean()\n",
154:     "C12=pd.DataFrame([\n",
155:     "    {'method':'C1_written','n':len(Fin),'accuracy':float(all_raw),'error_rate':1-float(all_raw)},\n",
156:     "    {'method':'C2_executed','n':len(Fin),'accuracy':float(all_exec),'error_rate':1-float(all_exec)},\n",
157:     "    {'method':'C1_written_holdout','n':len(Hold),'accuracy':float(boolcol(Hold.written_correct).mean())},\n",
158:     "    {'method':'C2_executed_holdout','n':len(Hold),'accuracy':float(boolcol(Hold.execution_correct).mean())},\n",
159:     "])\n",
160:     "C12.to_csv(OUT/'tables/C1_C2_answer_reconciliation.csv',index=False)\n",
161:     "display(C12);display(cont)"
162:    ]
163:   },
164:   {
165:    "cell_type": "markdown",
166:    "id": "51a2d613",
167:    "metadata": {},
168:    "source": [
169:     "## 3. C3–C6 — Apply **frozen** risk thresholds, no holdout selection"
170:    ]
171:   },
172:   {
173:    "cell_type": "code",
174:    "execution_count": null,
175:    "id": "ba4f5f9c",
176:    "metadata": {},
177:    "outputs": [],
178:    "source": [
179:     "FIN_METHODS=list(fitmeta['thresholds_from_fit_oof'])\n",
180:     "win=B_FREEZE['winner_from_fit_oof'];goal=float(fitmeta['target_coverage'])\n",
181:     "hs=Hold.set_index('record_id');htruth=hs.execution_correct.astype(int).to_numpy()\n",
182:     "validity=(hs.official_executable_feature.astype(bool)&hs.program_structural_valid.astype(bool))\n",
183:     "grounding=~hs.any_literal_unmatched.astype(bool)\n",
184:     "report=[]; decisions=[]\n",
185:     "def record_method(name,mask,truth=htruth,group='FinQA'):\n",
186:     "    mask=np.asarray(mask,dtype=bool)\n",
187:     "    acc=float(np.mean(truth[mask])) if mask.any() else float('nan')\n",
188:     "    report.append({'method':name,'dataset':group,'n':len(truth),'accepted':int(mask.sum()),\n",
189:     "                   'coverage':float(mask.mean()),'selective_accuracy':acc,\n",
190:     "                   'selective_risk':1-acc if np.isfinite(acc) else float('nan'),\n",
191:     "                   'selection_note':'Phase-4B fit-only threshold; Phase-4C holdout evaluated once'})\n",
192:     "    decisions.extend({'record_id':rid,'method':name,'accepted':bool(a)} for rid,a in zip(hs.index,mask))\n",
193:     "record_method('C2_execution_only',np.ones(len(hs),dtype=bool))\n",
194:     "record_method('C3_structural_gate',validity.to_numpy())\n",
195:     "record_method('C3_grounding_gate',(validity&grounding).to_numpy())\n",
196:     "for name in FIN_METHODS:\n",
197:     "    subset=Bscores[Bscores.method==name].set_index('record_id').loc[hs.index]\n",
198:     "    cut=float(fitmeta['thresholds_from_fit_oof'][name]);select=subset.p_correct.to_numpy(dtype=float)>=cut\n",
199:     "    record_method('C4_likelihood_'+name, select&validity.to_numpy()) if name in ['B1_raw','B2_likelihood'] else None\n",
200:     "    record_method('C5_verifier_'+name, select&validity.to_numpy()) if name in ['B3_verifier','B4_grounding'] else None\n",
201:     "    if name==win:\n",
202:     "        record_method('C6_full_frozen',select&validity.to_numpy())\n",
203:     "C6=pd.DataFrame(report);C6.to_csv(OUT/'tables/C3_C6_ablation_holdout.csv',index=False)\n",
204:     "pd.DataFrame(decisions).to_csv(OUT/'predictions/C3_C6_acceptance_masks.csv',index=False)\n",
205:     "display(C6)"
206:    ]
207:   },
208:   {
209:    "cell_type": "markdown",
210:    "id": "19ae90cd",
211:    "metadata": {},
212:    "source": [
213:     "## 4. C7 — Component ablations and error attribution"
214:    ]
215:   },
216:   {
217:    "cell_type": "code",
218:    "execution_count": null,
219:    "id": "6bb66fe9",
220:    "metadata": {},
221:    "outputs": [],
222:    "source": [
223:     "details=[]\n",
224:     "for name in ['B1_raw','B2_likelihood','B3_verifier','B4_grounding']:\n",
225:     "    row=C6[C6.method.str.endswith(name)]\n",
226:     "    if len(row):\n",
227:     "        r=row.iloc[0]; details.append({'risk_model':name,'coverage':r.coverage,\n",
228:     "            'selective_risk':r.selective_risk,'difference_from_execution_only':r.selective_risk - (1-htruth.mean())})\n",
229:     "C7=pd.DataFrame(details);C7.to_csv(OUT/'tables/C7_component_ablation.csv',index=False)\n",
230:     "display(C7)\n",
231:     "# Literal mismatch subgroup is NOT a proof of semantic grounding.\n",
232:     "subgroup=Hold.groupby('any_literal_unmatched')['execution_correct'].agg(['size','sum','mean'])\n",
233:     "subgroup.to_csv(OUT/'tables/C7_grounding_error_subgroups.csv')\n",
234:     "display(subgroup)"
235:    ]
236:   },
237:   {
238:    "cell_type": "markdown",
239:    "id": "13ea18fe",
240:    "metadata": {},
241:    "source": [
242:     "## 5. C8 — Paired significance and bootstrap confidence intervals"
243:    ]
244:   },
245:   {
246:    "cell_type": "code",
247:    "execution_count": null,
248:    "id": "b9d9ca40",
249:    "metadata": {},
250:    "outputs": [],
251:    "source": [
252:     "def paired_boot_diff(y1,y2,iterations=3000,seed=SEED):\n",
253:     "    y1=np.asarray(y1,float); y2=np.asarray(y2,float)\n",
254:     "    r=np.random.default_rng(seed);n=len(y1)\n",
255:     "    diffs=np.empty(iterations)\n",
256:     "    for t in range(iterations):\n",
257:     "        i=r.integers(0,n,n)\n",
258:     "        diffs[t]=y2[i].mean()-y1[i].mean()\n",
259:     "    return [float(np.percentile(diffs,q)) for q in [2.5,97.5]]\n",
260:     "r0=boolcol(Fin.written_correct).astype(int).to_numpy()\n",
261:     "r1=boolcol(Fin.execution_correct).astype(int).to_numpy()\n",
262:     "discord01=int(((r0==0)&(r1==1)).sum())\n",
263:     "discord10=int(((r0==1)&(r1==0)).sum())\n",
264:     "pval=float(binomtest(min(discord01,discord10),discord01+discord10,0.5).pvalue)\n",
265:     "ci=paired_boot_diff(r0,r1)\n",
266:     "C8A=pd.DataFrame([{'comparison':'C2_execution versus C1_written','n':len(Fin),\n",
267:     "    'absolute_accuracy_difference':float(r1.mean()-r0.mean()),\n",
268:     "    'bootstrap_95ci_low':ci[0],'bootstrap_95ci_high':ci[1],\n",
269:     "    'discordant_written_wrong_execution_right':discord01,\n",
270:     "    'discordant_written_right_execution_wrong':discord10,\n",
271:     "    'exact_mcnemar_p':pval}])\n",
272:     "C8A.to_csv(OUT/'tables/C8_paired_significance.csv',index=False)\n",
273:     "display(C8A)\n",
274:     "# Coverage-matched ranking evaluation: identical count, different accepted subsets.\n",
275:     "Bm=Bscores.copy()\n",
276:     "fixed_k=max(1,int(round(goal*len(hs))))\n",
277:     "comp=[]\n",
278:     "pmap={name:g.set_index('record_id').loc[hs.index].p_correct.to_numpy(dtype=float)\n",
279:     "      for name,g in Bm.groupby('method')}\n",
280:     "# For comparison, rank all cases, but invalid cases must go to the bottom.\n",
281:     "v=validity.to_numpy(dtype=bool)\n",
282:     "for name,p in pmap.items():\n",
283:     "    effective=p.copy();effective[~v]=-np.inf\n",
284:     "    order=np.argsort(-effective,kind='stable')\n",
285:     "    acc=float(htruth[order[:fixed_k]].mean())\n",
286:     "    comp.append({'method':name,'n':len(hs),'accepted':fixed_k,'coverage':fixed_k/len(hs),\n",
287:     "                 'matched_count_selective_risk':1-acc,\n",
288:     "                 'warning':'If all valid < target k, some invalid enter rank tail; see valid count.'})\n",
289:     "C8B=pd.DataFrame(comp).sort_values('matched_count_selective_risk')\n",
290:     "C8B.to_csv(OUT/'tables/C8_coverage_matched_comparison.csv',index=False)\n",
291:     "display(C8B)"
292:    ]
293:   },
294:   {
295:    "cell_type": "markdown",
296:    "id": "bdf9cb41",
297:    "metadata": {},
298:    "source": [
299:     "## 6. C8 — Bootstrap paired risk difference at fixed count (exploratory)"
300:    ]
301:   },
302:   {
303:    "cell_type": "code",
304:    "execution_count": null,
305:    "id": "dde20e81",
306:    "metadata": {},
307:    "outputs": [],
308:    "source": [
309:     "if 'B1_raw' in pmap and win in pmap:\n",
310:     "    def risk_at_k(y,p,k,valid):\n",
311:     "        score=p.copy(); score[~valid]=-np.inf\n",
312:     "        idx=np.argsort(-score,kind='stable')[:k]\n",
313:     "        return float(1-y[idx].mean())\n",
314:     "    baseline=pmap['B1_raw']; proposed=pmap[win];r=np.random.default_rng(SEED+2)\n",
315:     "    diffs=[]\n",
316:     "    for _ in range(2000):\n",
317:     "        ix=r.integers(0,len(htruth),len(htruth));k=max(1,round(goal*len(ix)))\n",
318:     "        diffs.append(risk_at_k(htruth[ix],proposed[ix],k,v[ix])-\n",
319:     "                     risk_at_k(htruth[ix],baseline[ix],k,v[ix]))\n",
320:     "    res={'comparison':f'{win} minus raw likelihood, at equal selected count',\n",
321:     "         'diff_negative_is_better':float(np.mean(diffs)),\n",
322:     "         '95_ci':[float(np.percentile(diffs,q)) for q in (2.5,97.5)],\n",
323:     "         'n_bootstrap':len(diffs),'selection_condition':'fit-only winner frozen'}\n",
324:     "    save('C8_paired_bootstrap_risk_difference.json',res)\n",
325:     "    print(res)"
326:    ]
327:   },
328:   {
329:    "cell_type": "markdown",
330:    "id": "88689be4",
331:    "metadata": {},
332:    "source": [
333:     "## 7. C8 — FiQA selective regression (independent task semantics)"
334:    ]
335:   },
336:   {
337:    "cell_type": "code",
338:    "execution_count": null,
339:    "id": "f97f1280",
340:    "metadata": {},
341:    "outputs": [],
342:    "source": [
343:     "qH=Fi[Fi.calibration_fold=='calibration_holdout'].set_index('record_id')\n",
344:     "fi_rows=[]\n",
345:     "for method,grp in Bfiqa.groupby('method'):\n",
346:     "    sub=grp.set_index('record_id').loc[qH.index]\n",
347:     "    accepted=sub.accept_80pct.astype(bool).to_numpy()\n",
348:     "    errors=qH.absolute_error.to_numpy(dtype=float)\n",
349:     "    fi_rows.append({'method':method,'n':len(qH),'accepted':int(accepted.sum()),\n",
350:     "                    'coverage':float(accepted.mean()),\n",
351:     "                    'selective_mae':float(errors[accepted].mean()) if accepted.any() else np.nan,\n",
352:     "                    'full_mae':float(errors.mean())})\n",
353:     "C8Q=pd.DataFrame(fi_rows)\n",
354:     "C8Q.to_csv(OUT/'tables/C8_fiqa_selective_mae.csv',index=False)\n",
355:     "display(C8Q)"
356:    ]
357:   },
358:   {
359:    "cell_type": "markdown",
360:    "id": "5b60c5c2",
361:    "metadata": {},
362:    "source": [
363:     "## 8. C9 — Efficiency (separate measured generation and post-processing)"
364:    ]
365:   },
366:   {
367:    "cell_type": "code",
368:    "execution_count": null,
369:    "id": "3e8329d1",
370:    "metadata": {},
371:    "outputs": [],
372:    "source": [
373:     "# E5 latency was measured during model generation in Phase-3 (GPU), not during\n",
374:     "# this CPU-only post-processing pass. Verification/grounding latency must be\n",
375:     "# measured in deployed Phase-4 application, and is deliberately NOT invented.\n",
376:     "p3=E5.copy();p3.latency_seconds=pd.to_numeric(p3.latency_seconds,errors='coerce')\n",
377:     "EFF=pd.DataFrame([{\n",
378:     "    'component':'Frozen E5 generation (Phase-3 observed)',\n",
379:     "    'sample_count':int(len(p3)),\n",
380:     "    'mean_latency_seconds':float(p3.latency_seconds.mean()),\n",
381:     "    'latency_source':'Phase-3 original inference logs',\n",
382:     "    'verifier_additional_latency_seconds':'NOT_MEASURED',\n",
383:     "    'calibrator_additional_latency_seconds':'NOT_MEASURED',\n",
384:     "}])\n",
385:     "EFF.to_csv(OUT/'tables/C9_efficiency.csv',index=False)\n",
386:     "display(EFF)"
387:    ]
388:   },
389:   {
390:    "cell_type": "markdown",
391:    "id": "39cbc633",
392:    "metadata": {},
393:    "source": [
394:     "## 9. Research figures and C10 methodology freeze"
395:    ]
396:   },
397:   {
398:    "cell_type": "code",
399:    "execution_count": null,
400:    "id": "b5e24baf",
401:    "metadata": {},
402:    "outputs": [],
403:    "source": [
404:     "fig,ax=plt.subplots(figsize=(9,5))\n",
405:     "s=C6.sort_values('coverage')\n",
406:     "for name,row in s.groupby('method'):\n",
407:     "    if name in ['C6_full_frozen','C2_execution_only','C3_grounding_gate'] or 'C4_likelihood_B1_raw' in name:\n",
408:     "        ax.scatter(row.coverage,row.selective_risk,label=name,s=55)\n",
409:     "ax.set(xlabel='Actual FinQA holdout coverage',ylabel='Error among accepted',\n",
410:     "       title='Phase-4C validation-holdout selective risk')\n",
411:     "ax.legend(fontsize=8);fig.tight_layout();fig.savefig(OUT/'plots/C3_C6_selective_risk.png',dpi=180);plt.show()\n",
412:     "missing=[x for x in ['C1_C2_answer_reconciliation.csv','C3_C6_ablation_holdout.csv',\n",
413:     "        'C7_component_ablation.csv','C8_paired_significance.csv','C8_coverage_matched_comparison.csv',\n",
414:     "        'C8_fiqa_selective_mae.csv','C9_efficiency.csv'] if not (OUT/'tables'/x).exists()]\n",
415:     "GATES={'phase4b_full_frozen':True,'e5_baseline_1000':len(E5)==1000,\n",
416:     "       'finqa_holdout_185':len(Hold)==185,'fiqa_holdout_25':len(qH)==25,\n",
417:     "       'no_holdout_retraining':True,'no_threshold_refitting':True,\n",
418:     "       'all_C1_to_C10_tables_written':not missing}\n",
419:     "state='PASS_METHODOLOGY_FROZEN' if all(GATES.values()) else 'FAIL'\n",
420:     "readiness={'phase':'Phase-4C','status':state,'gates':GATES,\n",
421:     "    'winner_risk_policy':win,'fixed_coverage':goal,'n_finqa_holdout':185,'n_fiqa_holdout':25,\n",
422:     "    'protected_test_sets_used':False,'fintrust_final_protocol':'NOT_VERIFIED',\n",
423:     "    'no_claim_of_superiority_without_effect_estimate':True,\n",
424:     "    'verifier_aware_policy_selected':win in ['B3_verifier','B4_grounding'],\n",
425:     "    'novelty_status':('NOT_SUPPORTED_BY_FIT_ONLY_SELECTION' if win in ['B1_raw','B2_likelihood'] else 'REQUIRES_TEST_CONFIRMATION'),\n",
426:     "    'cautions':['This is validation-holdout evidence, not a protected test result.',\n",
427:     "        'Phase-4B already reported holdout; no new independent holdout exists.',\n",
428:     "        'Full TrustFin-SLM may not beat execution-only or likelihood baseline.',\n",
429:     "        'Grounding is numerical presence, not complete semantic grounding.']}\n",
430:     "save('phase4c_readiness.json',readiness)\n",
431:     "frozen={'phase':'Phase-4C','status':state,'base_model':'Qwen/Qwen3-1.7B',\n",
432:     "    'frozen_sft_experiment':'E5','risk_policy':win,\n",
433:     "    'selected_on':'Phase-4B calibration-fit OOF only',\n",
434:     "    'threshold':fitmeta['thresholds_from_fit_oof'][win],\n",
435:     "    'target_coverage':goal,'deterministic_gate':['official_executable_feature','program_structural_valid'],\n",
436:     "    'FiQA_policy_file':'Phase-4B B6_fiqa_policy.json',\n",
437:     "    'protect_test':'Do not access protected FinQA/FiQA/FOMC/FinTrust tests until methodology freeze is reviewed.',\n",
438:     "    'limitation':'Any claim of TrustFin-SLM superiority must come from observed, verified test estimates.'}\n",
439:     "save('phase4_methodology_freeze.json',frozen)\n",
440:     "pkg=OUT/'Phase4C_Review_Package.zip'\n",
441:     "with zipfile.ZipFile(pkg,'w',zipfile.ZIP_DEFLATED) as z:\n",
442:     "    for file in sorted(OUT.rglob('*')):\n",
443:     "        if file.is_file() and file!=pkg:z.write(file,file.relative_to(OUT))\n",
444:     "print('Phase-4C:',state,'Package:',pkg)\n",
445:     "if win not in ['B3_verifier','B4_grounding']:\n",
446:     "    print('IMPORTANT: Fit-OOF selection chose likelihood-only; C6 is NOT evidence of a novel verifier-aware improvement.')\n",
447:     "assert not missing, missing"
448:    ]
449:   },
450:   {
451:    "cell_type": "markdown",
452:    "id": "248a4d4f",
453:    "metadata": {},
454:    "source": [
455:     "## Output checklist for manuscript review\n",
456:     "\n",
457:     "Share the executed notebook and `artifacts/phase4c_full/Phase4C_Review_Package.zip` (includes `phase4_methodology_freeze.json` and `phase4c_readiness.json`). Essential tables are `C1_C2_answer_reconciliation.csv`, `C3_C6_ablation_holdout.csv`, `C7_component_ablation.csv`, `C8_paired_significance.csv`, `C8_coverage_matched_comparison.csv`, `C8_fiqa_selective_mae.csv`, and `C9_efficiency.csv`.\n",
458:     "\n",
459:     "**Next phase:** Review whether verifier-aware selective risk truly improves over execution-only and likelihood-only. Preserve the already frozen E5, risk features, thresholds and prompts. Resolve FinTrust dimension-specific scoring and then perform a **single protected-test evaluation** without altering the methodology."
460:    ]
461:   }
462:  ],
463:  "metadata": {
464:   "kernelspec": {
465:    "display_name": "Python 3",
466:    "language": "python",
467:    "name": "python3"
468:   },
469:   "language_info": {
470:    "name": "python"
471:   }
472:  },
473:  "nbformat": 4,
474:  "nbformat_minor": 5
475: }

---

## Assistant

**Tool: glob**

**Input:**
```json
{
  "pattern": "**/Phase4A*Review*zip"
}
```

/home/heet18/Downloads/Research Papers/artifacts/phase4a_smoke/Phase4A_Review_Package.zip
/home/heet18/Downloads/Research Papers/artifacts/phase4a/Phase4A_Review_Package.zip

---

## Assistant

**Tool: read**

**Input:**
```json
{
  "path": "/home/heet18/Downloads/Research Papers/notebooks/Phase4A_TrustFin_Execution_Verification_and_Financial_Grounding.ipynb | head -200"
}
```

File not found: /home/heet18/Downloads/Research Papers/notebooks/Phase4A_TrustFin_Execution_Verification_and_Financial_Grounding.ipynb | head -200

Did you mean one of these?
/home/heet18/Downloads/Research Papers/notebooks/Phase4A_TrustFin_Execution_Verification_and_Financial_Grounding.ipynb

---

## Assistant

**Tool: shell**

**Input:**
```json
{
  "command": "ls /home/heet18/Downloads/Research\\ Papers/artifacts/ | head -30"
}
```

phase2
phase2_1
phase3
phase3_1
phase3_FINAL_deliverable_2026-10-11_1018.tar.gz
phase3_smoke_16of16.tar.gz
phase3_smoke_backup.tar.gz
phase3_snapshot_2026-10-10_2350.tar.gz
phase4
phase4a
phase4a_smoke


---

## Assistant

**Tool: shell**

**Input:**
```json
{
  "command": "ls /home/heet18/Downloads/Research\\ Papers/artifacts/phase4*"
}
```

/home/heet18/Downloads/Research Papers/artifacts/phase4:
analysis

/home/heet18/Downloads/Research Papers/artifacts/phase4a:
analysis
features
labels
manifests
Phase4A_Review_Package.zip
Phase4A_Student_Summary.md
phase4b_interface
plots
tables

/home/heet18/Downloads/Research Papers/artifacts/phase4a_smoke:
analysis
features
labels
manifests
Phase4A_Review_Package.zip
Phase4A_Student_Summary.md
phase4b_interface
plots
tables


---

## Assistant

**Tool: shell**

**Input:**
```json
{
  "command": "find /home/heet18/Downloads/Research\\ Papers/artifacts -name \"phase4b*\" -type f | head -15"
}
```

/home/heet18/Downloads/Research Papers/artifacts/phase4a/phase4b_interface/phase4b_finqa_features.jsonl
/home/heet18/Downloads/Research Papers/artifacts/phase4a/phase4b_interface/phase4b_feature_policy.json
/home/heet18/Downloads/Research Papers/artifacts/phase4a/phase4b_interface/phase4b_fiqa_features.jsonl
/home/heet18/Downloads/Research Papers/artifacts/phase4a/labels/phase4b_finqa_labels.jsonl
/home/heet18/Downloads/Research Papers/artifacts/phase4a/labels/phase4b_fiqa_labels.jsonl
/home/heet18/Downloads/Research Papers/artifacts/phase4a_smoke/phase4b_interface/phase4b_finqa_features.jsonl
/home/heet18/Downloads/Research Papers/artifacts/phase4a_smoke/phase4b_interface/phase4b_feature_policy.json
/home/heet18/Downloads/Research Papers/artifacts/phase4a_smoke/phase4b_interface/phase4b_fiqa_features.jsonl
/home/heet18/Downloads/Research Papers/artifacts/phase4a_smoke/labels/phase4b_finqa_labels.jsonl
/home/heet18/Downloads/Research Papers/artifacts/phase4a_smoke/labels/phase4b_fiqa_labels.jsonl


---

## Assistant

**Tool: shell**

**Input:**
```json
{
  "command": "ls /home/heet18/Downloads/Research\\ Papers/artifacts/phase4b* 2>&1 | head -15"
}
```

ls: cannot access '/home/heet18/Downloads/Research Papers/artifacts/phase4b*': No such file or directory


---

## Assistant

**Tool: shell**

**Input:**
```json
{
  "command": "cat /home/heet18/Downloads/Research\\ Papers/artifacts/phase4a/manifests/phase4a_readiness.json | head -80"
}
```

{
  "phase": "Phase-4A",
  "run_mode": "full",
  "core_phase4a_status": "PASS",
  "research_full_grounding_status": "PASS",
  "independent_new_official_execution": "NOT_RUN",
  "local_adapter_fingerprints": "NOT_AVAILABLE_LOCALLY",
  "fintrust_protocol_status": "OUT_OF_SCOPE_PHASE4A_PRESERVE_PHASE3_1_HOLD",
  "risk_feature_policy_status": "READY_FOR_PHASE4B_DEVELOPMENT",
  "n_finqa": 883,
  "n_fiqa": 117,
  "source_fingerprint": {
    "phase3_zip_sha256": "17fe3f9882219b7c8a7997c67554f5bcb62c5909e955e6d5dae8df5e9a4506fa",
    "phase3_1_zip_sha256": "3d72d7edfbc3d847ecc98a4653a14264ceb70a97da17b8a9cd8f0e4a3487d616",
    "phase3_freeze_sha256": "6586a42de81bed2cc33b9a7fd860554f6484555f9c2b31d5bb3350686391e423",
    "handoff_sha256": "9ad66d6c83dcf966cb6c1ccb1ea918bb79969eb7d15c79cff5ae3ac47f6def5f"
  },
  "issues": [],
  "notes": [
    "All findings are validation-only.",
    "Features and targets are stored separately.",
    "Grounding literal presence does not establish semantic correctness.",
    "No new training or protected-test evaluation was run."
  ]
}

---

## Assistant

**Tool: shell**

**Input:**
```json
{
  "command": "find /home/heet18/Downloads/Research\\ Papers -name \"*Phase3*Deliverable*.zip\" | head -3"
}
```

/home/heet18/Downloads/Research Papers/deliverables/Phase3_TrustFin_Deliverable.zip


---

## Assistant

**Tool: shell**

**Input:**
```json
{
  "command": "pip show nbconvert papermill jupyter 2>&1 | head -5"
}
```

WARNING: Package(s) not found: jupyter, papermill
Name: nbconvert
Version: 7.17.1
Summary: Convert Jupyter Notebooks (.ipynb files) to other formats.
Home-page: 


---

## Assistant

**Tool: shell**

**Input:**
```json
{
  "command": "python -m nbconvert --version 2>&1 | head -2"
}
```

/bin/bash: line 1: python: command not found


---

## Assistant

**Tool: shell**

**Input:**
```json
{
  "command": "python3 --version && pip3 show nbformat nbconvert 2>&1 | head -3"
}
```

Python 3.12.3
Name: nbformat
Version: 5.11.1
Summary: The Jupyter Notebook format


---

## Assistant

Let us look up how to execute notebooks and run them.


**Tool: shell**

**Input:**
```json
{
  "command": "pip3 show papermill 2>&1 | head -2"
}
```

WARNING: Package(s) not found: papermill


---

## Assistant

Let us run Phase4B first (needs Phase4A outputs which we have), then Phase4C.


**Tool: shell**

**Input:**
```json
{
  "command": "cd /home/heet18/Downloads/Research\\ Papers && TRUSTFIN_PHASE4A_ZIP=/home/heet18/Downloads/Research\\ Papers/artifacts/phase4a/Phase4A_Review_Package.zip TRUSTFIN_RUN_MODE=full jupyter nbconvert --to notebook --execute notebooks/Phase4B_TrustFin_Calibration_and_Selective_Prediction.ipynb --output Phase4B_executed.ipynb --ExecutePreprocessor.timeout=3600 2>&1 | tail -40"
}
```

[NbConvertApp] Converting notebook notebooks/Phase4B_TrustFin_Calibration_and_Selective_Prediction.ipynb to notebook
[NbConvertApp] Writing 125804 bytes to notebooks/Phase4B_executed.ipynb


---

## Assistant

**Tool: shell**

**Input:**
```json
{
  "command": "find /home/heet18/Downloads/Research\\ Papers/artifacts -name \"*phase4b*\" -type f 2>&1 | head -15"
}
```

/home/heet18/Downloads/Research Papers/artifacts/phase4a/phase4b_interface/phase4b_finqa_features.jsonl
/home/heet18/Downloads/Research Papers/artifacts/phase4a/phase4b_interface/phase4b_feature_policy.json
/home/heet18/Downloads/Research Papers/artifacts/phase4a/phase4b_interface/phase4b_fiqa_features.jsonl
/home/heet18/Downloads/Research Papers/artifacts/phase4a/labels/phase4b_finqa_labels.jsonl
/home/heet18/Downloads/Research Papers/artifacts/phase4a/labels/phase4b_fiqa_labels.jsonl
/home/heet18/Downloads/Research Papers/artifacts/phase4a_smoke/phase4b_interface/phase4b_finqa_features.jsonl
/home/heet18/Downloads/Research Papers/artifacts/phase4a_smoke/phase4b_interface/phase4b_feature_policy.json
/home/heet18/Downloads/Research Papers/artifacts/phase4a_smoke/phase4b_interface/phase4b_fiqa_features.jsonl
/home/heet18/Downloads/Research Papers/artifacts/phase4a_smoke/labels/phase4b_finqa_labels.jsonl
/home/heet18/Downloads/Research Papers/artifacts/phase4a_smoke/labels/phase4b_fiqa_labels.jsonl


---

## Assistant

**Tool: shell**

**Input:**
```json
{
  "command": "ls /home/heet18/Downloads/Research\\ Papers/artifacts/phase4b_full 2>&1 | head -10"
}
```

ls: cannot access '/home/heet18/Downloads/Research Papers/artifacts/phase4b_full': No such file or directory


---

## Assistant

**Tool: shell**

**Input:**
```json
{
  "command": "find /home/heet18/Downloads/Research\\ Papers -path \"*phase4b_full*\" -type f 2>&1 | head -15"
}
```

/home/heet18/Downloads/Research Papers/notebooks/artifacts/phase4b_full/models/B3_verifier.joblib
/home/heet18/Downloads/Research Papers/notebooks/artifacts/phase4b_full/models/B2_likelihood.joblib
/home/heet18/Downloads/Research Papers/notebooks/artifacts/phase4b_full/models/B6_fiqa_error_ridge.joblib
/home/heet18/Downloads/Research Papers/notebooks/artifacts/phase4b_full/models/B4_grounding.joblib
/home/heet18/Downloads/Research Papers/notebooks/artifacts/phase4b_full/predictions/B1_B4_finqa_holdout_scores.csv
/home/heet18/Downloads/Research Papers/notebooks/artifacts/phase4b_full/predictions/B6_fiqa_holdout_scores.csv
/home/heet18/Downloads/Research Papers/notebooks/artifacts/phase4b_full/Phase4B_Review_Package.zip
/home/heet18/Downloads/Research Papers/notebooks/artifacts/phase4b_full/plots/B5_finqa_risk_coverage.png
/home/heet18/Downloads/Research Papers/notebooks/artifacts/phase4b_full/manifests/phase4c_handoff.json
/home/heet18/Downloads/Research Papers/notebooks/artifacts/phase4b_full/manifests/phase4b_freeze.json
/home/heet18/Downloads/Research Papers/notebooks/artifacts/phase4b_full/manifests/phase4b_readiness.json
/home/heet18/Downloads/Research Papers/notebooks/artifacts/phase4b_full/manifests/B6_fiqa_policy.json
/home/heet18/Downloads/Research Papers/notebooks/artifacts/phase4b_full/manifests/B8_fit_only_selection.json
/home/heet18/Downloads/Research Papers/notebooks/artifacts/phase4b_full/tables/B5_risk_coverage.csv
/home/heet18/Downloads/Research Papers/notebooks/artifacts/phase4b_full/tables/B6_fiqa_uncertainty.csv


---

## Assistant

**Tool: shell**

**Input:**
```json
{
  "command": "cat /home/heet18/Downloads/Research\\ Papers/notebooks/artifacts/phase4b_full/manifests/phase4b_readiness.json | head -40"
}
```

{
  "phase": "Phase-4B",
  "run_mode": "full",
  "status": "PASS_FROZEN",
  "gates": {
    "phase4a_core_pass": true,
    "phase4a_grounding_recomputed": true,
    "finqa_record_alignment": true,
    "fiqa_record_alignment": true,
    "fit_holdout_fixed": true,
    "fit_only_winner_frozen": true,
    "no_gold_in_features": true,
    "all_B1_to_B8_outputs_present": true
  },
  "fit_selection_winner": "B1_raw",
  "holdout_never_used_to_select_winner": true,
  "protected_test_sets_used": false,
  "limitations": [
    "The 185/25 holdouts are internal validation, not protected test sets.",
    "FiQA interval is descriptive; do not claim formal conformal guarantee.",
    "Source-based numerical presence is not semantic grounding.",
    "Do not retune based on these holdout outputs."
  ]
}

---

## Assistant

Now run Phase4C


**Tool: shell**

**Input:**
```json
{
  "command": "cd /home/heet18/Downloads/Research\\ Papers && TRUSTFIN_PHASE4B_ROOT=/home/heet18/Downloads/Research\\ Papers/notebooks/artifacts/phase4b_full TRUSTFIN_PHASE4A_ZIP=/home/heet18/Downloads/Research\\ Papers/artifacts/phase4a/Phase4A_Review_Package.zip TRUSTFIN_PHASE3_ZIP=/home/heet18/Downloads/Research\\ Papers/deliverables/Phase3_TrustFin_Deliverable.zip jupyter nbconvert --to notebook --execute notebooks/Phase4C_TrustFin_Ablation_Statistical_Evaluation.ipynb --output Phase4C_executed.ipynb --ExecutePreprocessor.timeout=3600 2>&1 | tail -40"
}
```

[NbConvertApp] Converting notebook notebooks/Phase4C_TrustFin_Ablation_Statistical_Evaluation.ipynb to notebook
[NbConvertApp] Writing 98781 bytes to notebooks/Phase4C_executed.ipynb


---

## Assistant

**Tool: shell**

**Input:**
```json
{
  "command": "cat /home/heet18/Downloads/Research\\ Papers/notebooks/artifacts/phase4c_full/manifests/phase4c_readiness.json 2>&1 | head -60"
}
```

{
  "phase": "Phase-4C",
  "status": "PASS_METHODOLOGY_FROZEN",
  "gates": {
    "phase4b_full_frozen": true,
    "e5_baseline_1000": true,
    "finqa_holdout_185": true,
    "fiqa_holdout_25": true,
    "no_holdout_retraining": true,
    "no_threshold_refitting": true,
    "all_C1_to_C10_tables_written": true
  },
  "winner_risk_policy": "B1_raw",
  "fixed_coverage": 0.8,
  "n_finqa_holdout": 185,
  "n_fiqa_holdout": 25,
  "protected_test_sets_used": false,
  "fintrust_final_protocol": "NOT_VERIFIED",
  "no_claim_of_superiority_without_effect_estimate": true,
  "verifier_aware_policy_selected": false,
  "novelty_status": "NOT_SUPPORTED_BY_FIT_ONLY_SELECTION",
  "cautions": [
    "This is validation-holdout evidence, not a protected test result.",
    "Phase-4B already reported holdout; no new independent holdout exists.",
    "Full TrustFin-SLM may not beat execution-only or likelihood baseline.",
    "Grounding is numerical presence, not complete semantic grounding."
  ]
}

---

## Assistant

Phase-4B executed with full mode using Phase4A package - completed with PASS_FROZEN (winner B1_raw). Phase-4C executed successfully with PASS_METHODOLOGY_FROZEN. Both notebooks executed in sequence as requested; outputs are in notebooks/artifacts/phase4b_full/ and notebooks/artifacts/phase4c_full/.

---

## User

make the zip file of the deliverables.

---

## Assistant

**Tool: shell**

**Input:**
```json
{
  "command": "cd /home/heet18/Downloads/Research\\ Papers && mkdir -p deliverables/phase4_final_deliverables && \\\ncp notebooks/Phase4B_executed.ipynb deliverables/phase4_final_deliverables/ 2>&1; \\\ncp notebooks/Phase4C_executed.ipynb deliverables/phase4_final_deliverables/ 2>&1; \\\ncp -r notebooks/artifacts/phase4b_full deliverables/phase4_final_deliverables/phase4b_full 2>&1; \\\ncp -r notebooks/artifacts/phase4c_full deliverables/phase4_final_deliverables/phase4c_full 2>&1 && \\\necho \"---\" && \\\nls deliverables/phase4_final_deliverables/ | head -5"
}
```

cp: cannot stat 'notebooks/artifacts/phase4b_full': No such file or directory
cp: cannot stat 'notebooks/artifacts/phase4c_full': No such file or directory

Exited with code 1

---

## Assistant

**Tool: shell**

**Input:**
```json
{
  "command": "find /home/heet18/Downloads/Research\\ Papers -name \"Phase4B_executed.ipynb\" -not -path \"*/node_modules/*\" 2>&1 | head -2"
}
```

/home/heet18/Downloads/Research Papers/deliverables/phase4_final_deliverables/Phase4B_executed.ipynb
/home/heet18/Downloads/Research Papers/notebooks/Phase4B_executed.ipynb


---

## Assistant

**Tool: shell**

**Input:**
```json
{
  "command": "find /home/heet18/Downloads/Research\\ Papers -path \"*phase4b_full/Phase4B*\" -not -path \"*/node_modules/*\" 2>&1 | head -3"
}
```

/home/heet18/Downloads/Research Papers/artifacts/phase4b_full/Phase4B_Review_Package.zip


---

## Assistant

**Tool: shell**

**Input:**
```json
{
  "command": "find /home/heet18/Downloads/Research\\ Papers/artifacts -name \"phase4b_full\" -o -name \"phase4c_full\" 2>&1 | head -5"
}
```

/home/heet18/Downloads/Research Papers/artifacts/phase4b_full
/home/heet18/Downloads/Research Papers/artifacts/phase4c_full


---

## Assistant

**Tool: shell**

**Input:**
```json
{
  "command": "cd /home/heet18/Downloads/Research\\ Papers && mkdir -p deliverables/phase4_final_deliverables && \\\ncp notebooks/Phase4B_executed.ipynb deliverables/phase4_final_deliverables/ 2>&1 && \\\ncp notebooks/Phase4C_executed.ipynb deliverables/phase4_final_deliverables/ 2>&1 && \\\ncp -r artifacts/phase4b_full deliverables/phase4_final_deliverables/ 2>&1 && \\\ncp -r artifacts/phase4c_full deliverables/phase4_final_deliverables/ 2>&1 && \\\necho \"---\" && \\\nls deliverables/phase4_final_deliverables/ | head -6"
}
```

---
Phase4B_executed.ipynb
phase4b_full
Phase4C_executed.ipynb
phase4c_full


---

## Assistant

**Tool: shell**

**Input:**
```json
{
  "command": "cd /home/heet18/Downloads/Research\\ Papers/deliverables && zip -r phase4_final_deliverables.zip phase4_final_deliverables/ 2>&1 | tail -5"
}
```

  adding: phase4_final_deliverables/phase4c_full/tables/C3_C6_ablation_holdout.csv (deflated 71%)
  adding: phase4_final_deliverables/phase4c_full/tables/C8_fiqa_selective_mae.csv (deflated 47%)
  adding: phase4_final_deliverables/phase4c_full/tables/C8_coverage_matched_comparison.csv (deflated 65%)
  adding: phase4_final_deliverables/phase4c_full/tables/C1_C2_answer_reconciliation.csv (deflated 39%)
  adding: phase4_final_deliverables/Phase4B_executed.ipynb (deflated 44%)


---

## Assistant

Created deliverables/phase4_final_deliverables.zip with Phase4B and Phase4C executed notebooks plus their artifact directories.

---

## User

can you make the docs so that i can understand that what is the things done in the notebook 4B AND 4C 

---

## Assistant

**Tool: shell**

**Input:**
```json
{
  "command": "mkdir -p /home/heet18/Downloads/Research\\ Papers/docs/phase4_explanation && cat > /home/heet18/Downloads/Research\\ Papers/docs/phase4_explanation/Phase4B_Explained.md << 'EOF'\n# Phase-4B: Verifier-Aware Confidence Calibration & Selective Prediction\n\n## Goal\nPhase-4B builds a **verifier-aware risk estimator** to decide *when to trust* the executed program from the frozen E5 model (Qwen3-1.7B joint QLoRA). The core question: can we predict correctness better than raw model likelihood so that we can selectively abstain on uncertain cases?\n\n## Inputs\n- Phase-4A review package: `phase4b_interface/phase4b_finqa_features.jsonl`, `phase4b_interface/phase4b_fiqa_features.jsonl`, `labels/phase4b_finqa_labels.jsonl`, `labels/phase4b_fiqa_labels.jsonl`, `phase4b_interface/phase4b_feature_policy.json`, `manifests/phase4a_readiness.json`\n- Frozen splits: FinQA 883 total (698 fit / 185 holdout), FiQA 117 total (92 fit / 25 holdout). All splits fixed by Phase-4A.\n\n## Experiments (B1–B8)\n\n| ID | Name | What it does |\n|---|---|---|\n| B1 | Raw generated-token likelihood | Baseline confidence from model likelihood. Weak discriminator (~0.585 ROC-AUC). |\n| B2 | Likelihood-only Platt/logistic | Calibrate raw likelihood with logistic regression (fit on calibration-fit). |\n| B3 | Verifier-aware logistic | Adds verifier signals: `prompt_tokens`, `prompt_truncation_observed`, `official_executable_feature`, `program_structural_valid`. |\n| B4 | Grounding-aware logistic | Adds numerical grounding features: `any_literal_unmatched`, `unmatched_literal_count`, `literal_presence_fraction`, `max_question_local_overlap` (recomputed by Phase-4A). |\n| B5 | Selective prediction | Fit thresholds on fit-split OOF predictions to achieve target coverage (e.g., 80%). Apply same thresholds to holdout; coverage may vary. |\n| B6 | FiQA regression/error uncertainty | Predict absolute sentiment error with Ridge using FiQA features; produce selective MAE and a descriptive (non-conformal) interval. |\n| B7 | Reliability/AUC/Brier/ECE/AURC | Fit-OOF and holdout diagnostics to compare models. |\n| B8 | Phase-4B freeze | Select winner from **fit-split OOF only** (min AURC, tie-break Brier), freeze feature policy, write handoff to Phase-4C. Holdout never used for selection. |\n\n## Key Safeguards\n- **No label leakage**: Gold answers, correctness labels (`execution_correct`, `written_correct`, `corrected_correct`), and errors are never used as predictors. Only whitelisted features.\n- **Fit-only selection**: Hyperparameters/policy chosen from 5-fold (FinQA) / 4-fold (FiQA) OOF on calibration-fit. Thresholds frozen before looking at holdout.\n- **Handoffs separate**: Model artifacts (`*.joblib`), predictions, tables, and manifests written to `artifacts/phase4b_full/`.\n- **Deterministic gates**: Enforces record alignment, split sizes, feature whitelist, and presence of all B1–B8 outputs.\n\n## Outputs (what you get)\n- `manifests/phase4b_freeze.json` — winner, thresholds, feature sets, source hashes\n- `manifests/phase4b_readiness.json` — PASS_FROZEN gates\n- `manifests/B8_fit_only_selection.json` — fit-OOF selection details\n- `tables/B1_B4_fit_oof_comparison.csv`, `tables/B7_model_diagnostics.csv`, `tables/B5_risk_coverage.csv`, `tables/B6_fiqa_uncertainty.csv`\n- `predictions/B1_B4_finqa_holdout_scores.csv`, `predictions/B6_fiqa_holdout_scores.csv`\n- `models/B2_likelihood.joblib`, `models/B3_verifier.joblib`, `models/B4_grounding.joblib`, `models/B6_fiqa_error_ridge.joblib`\n- `Phase4B_Review_Package.zip`\nEOF\necho \"done\""
}
```

done


---

## Assistant

I'll create clear documentation explaining what was done in both Phase-4B and Phase-4C.


**Tool: shell**

**Input:**
```json
{
  "command": "cat > /home/heet18/Downloads/Research\\ Papers/docs/phase4_explanation/Phase4C_Explained.md << 'EOF'\n# Phase-4C: Ablations, Statistical Evaluation & Methodology Freeze\n\n## Goal\nPhase-4C evaluates whether the full TrustFin-SLM selective answering policy (with frozen Phase-4B risk estimator) actually improves *selective reliability* over simpler baselines. It does this on the same validation holdout from Phase-4B - **no new independent test set** is created. All thresholds come from fit-only Phase-4B, nothing is retrained or retuned here.\n\n## Inputs\n- Frozen Phase-4B outputs (`artifacts/phase4b_full/`): readiness, freeze, fit-only selection, holdout scores for B1-B4 and FiQA\n- Phase-4A package (same ZIP used by Phase-4B) for features/labels and alignment\n- Phase-3 deliverable ZIP for original E5 predictions and observed latencies\n\n## Experiments (C1–C10)\n\n| ID | Name | What it does |\n|---|---|---|\n| C1 | E5 written-answer baseline | Accuracy of raw model written answers on all FinQA and holdout. |\n| C2 | Execution-only corrected baseline | Accuracy after executing the generated program (deterministic execution). Shows big gain (e.g., ~37.6% → ~67.5% on validation context). |\n| C3 | Structural/grounding hard gates | Apply deterministic gates: `official_executable_feature` AND `program_structural_valid` (structural gate), plus numerical grounding (`any_literal_unmatched` false) as an additional hard filter. |\n| C4 | Likelihood-only abstention | Apply frozen B1/B2 thresholds with validity gate (compare to execution-only). |\n| C5 | Verifier-aware abstention | Apply frozen B3/B4 thresholds with validity gate. |\n| C6 | Full TrustFin-SLM selective policy | Apply the **fit-only winner** from Phase-4B with validity gate. This is the proposed policy being evaluated. |\n| C7 | Component ablations | Isolate contributions (likelihood vs verifier vs grounding) and look at error subgroups (e.g., literal-unmatched cases). |\n| C8 | Paired significance & bootstrap CIs | McNemar's test on written vs executed (all FinQA). Paired bootstrap for accuracy/risk differences. Coverage-matched comparison (equal k) and FiQA selective MAE. |\n| C9 | Efficiency reporting | Reports observed E5 generation latency from Phase-3; explicitly states verifier/calibrator overhead is **NOT_MEASURED** (CPU post-processing in this offline eval). |\n| C10 | Methodology freeze | Write `phase4c_readiness.json` and `phase4_methodology_freeze.json`, package review artifacts. |\n\n## Key Principles\n- **No holdout leakage**: All thresholds imported from `B8_fit_only_selection.json`. No refitting/rethresholding on holdout.\n- **Validation-only evidence**: Uses the same 185/25 holdout as Phase-4B - not a new independent test set. Protected tests untouched.\n- **Honest reporting**: If likelihood-only wins on fit-OOF, novelty is marked as not supported. CIs/effects reported regardless of significance.\n- **Execution is primary**: Structural executability + validity are enforced as hard gates where applicable.\n- **Grounding is limited**: Numerical presence is checked, but semantic correctness is not fully verified.\n\n## Outputs (what you get)\n- `manifests/phase4c_readiness.json` — methodology freeze status with gates and cautions\n- `manifests/phase4_methodology_freeze.json` — frozen policy (winner, threshold, gates), states tests not yet run\n- `tables/C1_C2_answer_reconciliation.csv`, `tables/C3_C6_ablation_holdout.csv`, `tables/C7_component_ablation.csv`, `tables/C8_paired_significance.csv`, `tables/C8_coverage_matched_comparison.csv`, `tables/C8_fiqa_selective_mae.csv`, `tables/C9_efficiency.csv`\n- `predictions/C3_C6_acceptance_masks.csv`\n- `plots/C3_C6_selective_risk.png`\n- `Phase4C_Review_Package.zip`\n\n## What \"PASS_METHODOLOGY_FROZEN\" means\nThe methodology (features, splits, selection protocol, thresholds) is frozen and evaluated honestly on validation holdout. It does **not** claim the proposed method is better than baselines - you need to look at the actual selective risk/accuracy and CIs in the tables to judge.\nEOF\necho \"done\""
}
```

done


---

## Assistant

Created docs/phase4_explanation/Phase4B_Explained.md and Phase4C_Explained.md with clear breakdowns of experiments, safeguards, outputs, and what the results mean.

---

## User

'/home/heet18/Downloads/Research Papers/notebooks/Phase4B_TrustFin_Calibration_and_Selective_Prediction.ipynb''/home/heet18/Downloads/Research Papers/notebooks/Phase4C_TrustFin_Ablation_Statistical_Evaluation.ipynb' run exactly this files dont make new files 

---

## Assistant

I'll run both notebooks in sequence exactly as specified, executing in place without creating new files.


**Tool: shell**

**Input:**
```json
{
  "command": "cd /home/heet18/Downloads/Research\\ Papers && TRUSTFIN_PHASE4A_ZIP=/home/heet18/Downloads/Research\\ Papers/artifacts/phase4a/Phase4A_Review_Package.zip TRUSTFIN_RUN_MODE=full jupyter nbconvert --to notebook --execute notebooks/Phase4B_TrustFin_Calibration_and_Selective_Prediction.ipynb --ExecutePreprocessor.timeout=3600 --output notebooks/Phase4B_TrustFin_Calibration_and_Selective_Prediction.ipynb 2>&1 | tail -15"
}
```

    app.start()
  File "/home/heet18/.local/lib/python3.12/site-packages/nbconvert/nbconvertapp.py", line 420, in start
    self.convert_notebooks()
  File "/home/heet18/.local/lib/python3.12/site-packages/nbconvert/nbconvertapp.py", line 597, in convert_notebooks
    self.convert_single_notebook(notebook_filename)
  File "/home/heet18/.local/lib/python3.12/site-packages/nbconvert/nbconvertapp.py", line 566, in convert_single_notebook
    write_results = self.write_single_notebook(output, resources)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/heet18/.local/lib/python3.12/site-packages/nbconvert/nbconvertapp.py", line 526, in write_single_notebook
    return self.writer.write(output, resources, notebook_name=notebook_name)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/heet18/.local/lib/python3.12/site-packages/nbconvert/writers/files.py", line 152, in write
    with open(dest_path, "w", encoding="utf-8") as f:
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
FileNotFoundError: [Errno 2] No such file or directory: 'notebooks/notebooks/Phase4B_TrustFin_Calibration_and_Selective_Prediction.ipynb'


---

## Assistant

**Tool: shell**

**Input:**
```json
{
  "command": "cd /home/heet18/Downloads/Research\\ Papers && TRUSTFIN_PHASE4A_ZIP=/home/heet18/Downloads/Research\\ Papers/artifacts/phase4a/Phase4A_Review_Package.zip TRUSTFIN_RUN_MODE=full jupyter nbconvert --to notebook --execute \"notebooks/Phase4B_TrustFin_Calibration_and_Selective_Prediction.ipynb\" --ExecutePreprocessor.timeout=3600 2>&1 | tail -5"
}
```

[NbConvertApp] Converting notebook notebooks/Phase4B_TrustFin_Calibration_and_Selective_Prediction.ipynb to notebook
[IPKernelApp] WARNING | Kernel is running over TCP without encryption. All communication (including code and outputs) is sent in plain text and is susceptible to eavesdropping. Use IPC transport or launch with kernel manager-provisioned CurveZMQ keys to enable transport encryption.
[NbConvertApp] Writing 122304 bytes to notebooks/Phase4B_TrustFin_Calibration_and_Selective_Prediction.nbconvert.ipynb


---

## Assistant

**Tool: shell**

**Input:**
```json
{
  "command": "cd /home/heet18/Downloads/Research\\ Papers && TRUSTFIN_PHASE4B_ROOT=/home/heet18/Downloads/Research\\ Papers/artifacts/phase4b_full TRUSTFIN_PHASE4A_ZIP=/home/heet18/Downloads/Research\\ Papers/artifacts/phase4a/Phase4A_Review_Package.zip TRUSTFIN_PHASE3_ZIP=/home/heet18/Downloads/Research\\ Papers/deliverables/Phase3_TrustFin_Deliverable.zip jupyter nbconvert --to notebook --execute \"notebooks/Phase4C_TrustFin_Ablation_Statistical_Evaluation.ipynb\" --ExecutePreprocessor.timeout=3600 2>&1 | tail -4"
}
```

[32m     14[39m     [38;5;28;01mwith[39;00m zipfile.ZipFile(zp) [38;5;28;01mas[39;00m z:

[31mAssertionError[39m: Phase-4A ZIP differs from the Phase-4B frozen source



---

## Assistant

**Tool: shell**

**Input:**
```json
{
  "command": "cat /home/heet18/Downloads/Research\\ Papers/artifacts/phase4b_full/manifests/phase4b_freeze.json | grep -B2 -A10 '\"source_sha256\"' | head -15"
}
```

  "status": "PASS_FROZEN",
  "source_phase4a": "/home/heet18/Downloads/Research Papers/artifacts/phase4a/Phase4A_Review_Package.zip",
  "source_sha256": "7cb252abb68e6f39602bf0341270755745edf2ee9d737ded73f60b9c4b9bffec",
  "source_component_hashes": {
    "policy": "4dbfb03dcfa16559e2cb3d68bfc329835b058564d6cc8f760894682fb3231c7c",
    "readiness": "d293bbf5b4863a217a4716a6e0b5641780883c22b5c1ff1f9e26961ab9314b09",
    "finqa_features": "cfea556c315448eee5702356d1e978568eb188898c6a2dc0c2d112c944cfa019",
    "finqa_labels": "3cbccde0837ec180a6bcfcc93e1cf77b0545d2da52e76d2886cf67799a9a4de1",
    "fiqa_features": "1b51da3b7eac1c55bd04125c3582be9c9182b38dc109532ee231fc67d8b9f851",
    "fiqa_labels": "0f57b38e274c6fc8d35faba53a93f43cc17bbaaa044871b5ef42a48592f66e5c"
  },
  "winner_from_fit_oof": "B1_raw",
  "fit_only_selection": {


---

## Assistant

**Tool: shell**

**Input:**
```json
{
  "command": "cd /home/heet18/Downloads/Research\\ Papers/notebooks && TRUSTFIN_PHASE4A_ZIP=/home/heet18/Downloads/Research\\ Papers/artifacts/phase4a/Phase4A_Review_Package.zip TRUSTFIN_RUN_MODE=full jupyter nbconvert --to notebook --execute Phase4B_TrustFin_Calibration_and_Selective_Prediction.ipynb --output Phase4B_TrustFin_Calibration_and_Selective_Prediction.ipynb --ExecutePreprocessor.timeout=7200 2>&1 | tail -8"
}
```

[NbConvertApp] Converting notebook Phase4B_TrustFin_Calibration_and_Selective_Prediction.ipynb to notebook
[IPKernelApp] WARNING | Kernel is running over TCP without encryption. All communication (including code and outputs) is sent in plain text and is susceptible to eavesdropping. Use IPC transport or launch with kernel manager-provisioned CurveZMQ keys to enable transport encryption.
[NbConvertApp] Writing 122304 bytes to Phase4B_TrustFin_Calibration_and_Selective_Prediction.ipynb


---

## Assistant

Running both notebooks in place as requested.


**Tool: shell**

**Input:**
```json
{
  "command": "cd /home/heet18/Downloads/Research\\ Papers/notebooks && TRUSTFIN_PHASE4A_ZIP=/home/heet18/Downloads/Research\\ Papers/artifacts/phase4a/Phase4A_Review_Package.zip TRUSTFIN_RUN_MODE=full jupyter nbconvert --to notebook --execute Phase4B_TrustFin_Calibration_and_Selective_Prediction.ipynb --ExecutePreprocessor.timeout=7200 --output Phase4B_TrustFin_Calibration_and_Selective_Prediction.ipynb 2>&1 | tail -6"
}
```

[NbConvertApp] Converting notebook Phase4B_TrustFin_Calibration_and_Selective_Prediction.ipynb to notebook
[IPKernelApp] WARNING | Kernel is running over TCP without encryption. All communication (including code and outputs) is sent in plain text and is susceptible to eavesdropping. Use IPC transport or launch with kernel manager-provisioned CurveZMQ keys to enable transport encryption.
[NbConvertApp] Writing 122304 bytes to Phase4B_TrustFin_Calibration_and_Selective_Prediction.ipynb


---

## Assistant

**Tool: shell**

**Input:**
```json
{
  "command": "cd /home/heet18/Downloads/Research\\ Papers/notebooks && TRUSTFIN_PHASE4B_ROOT=/home/heet18/Downloads/Research\\ Papers/artifacts/phase4b_full TRUSTFIN_PHASE4A_ZIP=/home/heet18/Downloads/Research\\ Papers/artifacts/phase4a/Phase4A_Review_Package.zip TRUSTFIN_PHASE3_ZIP=/home/heet18/Downloads/Research\\ Papers/deliverables/Phase3_TrustFin_Deliverable.zip jupyter nbconvert --to notebook --execute Phase4C_TrustFin_Ablation_Statistical_Evaluation.ipynb --ExecutePreprocessor.timeout=7200 --output Phase4C_TrustFin_Ablation_Statistical_Evaluation.ipynb 2>&1 | tail -3"
}
```


[31mAssertionError[39m: Phase-4A ZIP differs from the Phase-4B frozen source



---

## Assistant

**Tool: shell**

**Input:**
```json
{
  "command": "cd /home/heet18/Downloads/Research\\ Papers/notebooks && TRUSTFIN_PHASE4A_ZIP=/home/heet18/Downloads/Research\\ Papers/artifacts/phase4a/Phase4A_Review_Package.zip TRUSTFIN_RUN_MODE=full jupyter nbconvert --to notebook --execute Phase4B_TrustFin_Calibration_and_Selective_Prediction.ipynb --ExecutePreprocessor.timeout=7200 --output Phase4B_TrustFin_Calibration_and_Selective_Prediction.ipynb 2>&1 | tail -2"
}
```

[IPKernelApp] WARNING | Kernel is running over TCP without encryption. All communication (including code and outputs) is sent in plain text and is susceptible to eavesdropping. Use IPC transport or launch with kernel manager-provisioned CurveZMQ keys to enable transport encryption.
[NbConvertApp] Writing 122304 bytes to Phase4B_TrustFin_Calibration_and_Selective_Prediction.ipynb


---

## Assistant

Running both notebooks in place from the repo root.


**Tool: shell**

**Input:**
```json
{
  "command": "cd /home/heet18/Downloads/Research\\ Papers && \\\nTRUSTFIN_PHASE4A_ZIP=/home/heet18/Downloads/Research\\ Papers/artifacts/phase4a/Phase4A_Review_Package.zip TRUSTFIN_RUN_MODE=full \\\njupyter nbconvert --to notebook --execute notebooks/Phase4B_TrustFin_Calibration_and_Selective_Prediction.ipynb --ExecutePreprocessor.timeout=7200 --output /tmp/phase4b_tmp.ipynb && \\\ncp /tmp/phase4b_tmp.ipynb notebooks/Phase4B_TrustFin_Calibration_and_Selective_Prediction.ipynb && \\\nrm /tmp/phase4b_tmp.ipynb && \\\necho \"---PHASE4B_DONE---\" && \\\nTRUSTFIN_PHASE4B_ROOT=/home/heet18/Downloads/Research\\ Papers/artifacts/phase4b_full TRUSTFIN_PHASE4A_ZIP=/home/heet18/Downloads/Research\\ Papers/artifacts/phase4a/Phase4A_Review_Package.zip TRUSTFIN_PHASE3_ZIP=/home/heet18/Downloads/Research\\ Papers/deliverables/Phase3_TrustFin_Deliverable.zip \\\njupyter nbconvert --to notebook --execute notebooks/Phase4C_TrustFin_Ablation_Statistical_Evaluation.ipynb --ExecutePreprocessor.timeout=7200 --output /tmp/phase4c_tmp.ipynb && \\\ncp /tmp/phase4c_tmp.ipynb notebooks/Phase4C_TrustFin_Ablation_Statistical_Evaluation.ipynb && \\\nrm /tmp/phase4c_tmp.ipynb && \\\necho \"---PHASE4C_DONE---\" 2>&1 | tail -30"
}
```

[NbConvertApp] Converting notebook notebooks/Phase4B_TrustFin_Calibration_and_Selective_Prediction.ipynb to notebook
[IPKernelApp] WARNING | Kernel is running over TCP without encryption. All communication (including code and outputs) is sent in plain text and is susceptible to eavesdropping. Use IPC transport or launch with kernel manager-provisioned CurveZMQ keys to enable transport encryption.
[NbConvertApp] Writing 122304 bytes to /tmp/phase4b_tmp.ipynb
---PHASE4B_DONE---
[NbConvertApp] Converting notebook notebooks/Phase4C_TrustFin_Ablation_Statistical_Evaluation.ipynb to notebook
Traceback (most recent call last):
  File "/home/heet18/.local/bin/jupyter-nbconvert", line 8, in <module>
    sys.exit(main())
             ^^^^^^
  File "/home/heet18/.local/lib/python3.12/site-packages/jupyter_core/application.py", line 284, in launch_instance
    super().launch_instance(argv=argv, **kwargs)
  File "/home/heet18/.local/lib/python3.12/site-packages/traitlets/config/application.py", line 1082, in launch_instance
    app.start()
  File "/home/heet18/.local/lib/python3.12/site-packages/nbconvert/nbconvertapp.py", line 420, in start
    self.convert_notebooks()
  File "/home/heet18/.local/lib/python3.12/site-packages/nbconvert/nbconvertapp.py", line 597, in convert_notebooks
    self.convert_single_notebook(notebook_filename)
  File "/home/heet18/.local/lib/python3.12/site-packages/nbconvert/nbconvertapp.py", line 563, in convert_single_notebook
    output, resources = self.export_single_notebook(
                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/heet18/.local/lib/python3.12/site-packages/nbconvert/nbconvertapp.py", line 487, in export_single_notebook
    output, resources = self.exporter.from_filename(
                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/heet18/.local/lib/python3.12/site-packages/nbconvert/exporters/exporter.py", line 201, in from_filename
    return self.from_file(f, resources=resources, **kw)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/heet18/.local/lib/python3.12/site-packages/nbconvert/exporters/exporter.py", line 220, in from_file
    return self.from_notebook_node(
           ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/heet18/.local/lib/python3.12/site-packages/nbconvert/exporters/notebook.py", line 36, in from_notebook_node
    nb_copy, resources = super().from_notebook_node(nb, resources, **kw)
                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/heet18/.local/lib/python3.12/site-packages/nbconvert/exporters/exporter.py", line 154, in from_notebook_node
    nb_copy, resources = self._preprocess(nb_copy, resources)
                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/heet18/.local/lib/python3.12/site-packages/nbconvert/exporters/exporter.py", line 353, in _preprocess
    nbc, resc = preprocessor(nbc, resc)
                ^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/heet18/.local/lib/python3.12/site-packages/nbconvert/preprocessors/base.py", line 48, in __call__
    return self.preprocess(nb, resources)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/heet18/.local/lib/python3.12/site-packages/nbconvert/preprocessors/execute.py", line 103, in preprocess
    self.preprocess_cell(cell, resources, index)
  File "/home/heet18/.local/lib/python3.12/site-packages/nbconvert/preprocessors/execute.py", line 124, in preprocess_cell
    cell = self.execute_cell(cell, index, store_history=True)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/heet18/.local/lib/python3.12/site-packages/jupyter_core/utils/__init__.py", line 165, in wrapped
    return loop.run_until_complete(inner)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.12/asyncio/base_events.py", line 687, in run_until_complete
    return future.result()
           ^^^^^^^^^^^^^^^
  File "/home/heet18/.local/lib/python3.12/site-packages/nbclient/client.py", line 1062, in async_execute_cell
    await self._check_raise_for_error(cell, cell_index, exec_reply)
  File "/home/heet18/.local/lib/python3.12/site-packages/nbclient/client.py", line 918, in _check_raise_for_error
    raise CellExecutionError.from_cell_and_msg(cell, exec_reply_content)
nbclient.exceptions.CellExecutionError: An error occurred while executing the following cell:
------------------
def locate_zip(env,variants):
    q=os.environ.get(env,'')
    if q and Path(q).is_file():return Path(q)
    for name in variants:
        for root in [WORKSPACE_ROOT,Path.cwd(),Path('/mnt/data')]:
            p=root/name
            if p.is_file():return p
    raise FileNotFoundError(f'Set {env}: expected one of {variants}')
A_ZIP=locate_zip('TRUSTFIN_PHASE4A_ZIP',['Phase4A_Review_Package(1).zip','Phase4A_Review_Package.zip'])
if B_FREEZE.get('source_sha256'):
    assert hashlib.sha256(A_ZIP.read_bytes()).hexdigest()==B_FREEZE['source_sha256'], 'Phase-4A ZIP differs from the Phase-4B frozen source' 
P3_ZIP=locate_zip('TRUSTFIN_PHASE3_ZIP',['Phase3_TrustFin_Deliverable.zip'])
def read_zip_bytes(zp,suffix):
    with zipfile.ZipFile(zp) as z:
        found=[n for n in z.namelist() if n==suffix or n.endswith('/'+suffix)]
        if len(found)!=1:raise ValueError((zp,suffix,found))
        return z.read(found[0])
def zip_csv(zp,suffix):return pd.read_csv(io.BytesIO(read_zip_bytes(zp,suffix)))
def zip_jsonl(zp,suffix):
    return pd.DataFrame([json.loads(x) for x in read_zip_bytes(zp,suffix).decode().splitlines() if x.strip()])
E5=zip_csv(P3_ZIP,'predictions/E5.csv')
F=zip_jsonl(A_ZIP,'phase4b_interface/phase4b_finqa_features.jsonl')
FL=zip_jsonl(A_ZIP,'labels/phase4b_finqa_labels.jsonl')
Q=zip_jsonl(A_ZIP,'phase4b_interface/phase4b_fiqa_features.jsonl')
QL=zip_jsonl(A_ZIP,'labels/phase4b_fiqa_labels.jsonl')
Bscores=pd.read_csv(B_ROOT/'predictions/B1_B4_finqa_holdout_scores.csv')
Bfiqa=pd.read_csv(B_ROOT/'predictions/B6_fiqa_holdout_scores.csv')
fitmeta=load(B_ROOT/'manifests/B8_fit_only_selection.json')
assert fitmeta['holdout_not_used_for_selection']
assert E5.record_id.is_unique and len(E5)==1000
assert FL.record_id.is_unique and F.record_id.is_unique
assert set(FL.record_id)==set(F.record_id)
assert set(QL.record_id)==set(Q.record_id)
Fin=F.merge(FL,on=['record_id','calibration_fold'],validate='one_to_one')
Fin=Fin.merge(E5[E5.dataset=='FinQA'][['record_id','answer_correct','execution_correct','program_executable','latency_seconds']],
              on='record_id',validate='one_to_one',suffixes=('','_phase3'))
assert len(Fin)==883
Hold=Fin[Fin.calibration_fold=='calibration_holdout'].copy()
assert len(Hold)==185
Fi=Q.merge(QL,on=['record_id','calibration_fold'],validate='one_to_one')
assert len(Fi)==117 and len(Fi[Fi.calibration_fold=='calibration_holdout'])==25
assert set(Bscores.record_id)==set(Hold.record_id)
assert Bscores.groupby('method').record_id.nunique().eq(185).all()
print('Aligned E5 all 1000; FinQA holdout 185; FiQA holdout 25.')
------------------


[31m---------------------------------------------------------------------------[39m
[31mAssertionError[39m                            Traceback (most recent call last)
[36mCell[39m[36m [39m[32mIn[2][39m[32m, line 11[39m
[32m      7[39m             [38;5;28;01mif[39;00m p.is_file():[38;5;28;01mreturn[39;00m p
[32m      8[39m     [38;5;28;01mraise[39;00m FileNotFoundError(f'Set {env}: expected one of {variants}')
[32m      9[39m A_ZIP=locate_zip([33m'TRUSTFIN_PHASE4A_ZIP'[39m,[[33m'Phase4A_Review_Package(1).zip'[39m,[33m'Phase4A_Review_Package.zip'[39m])
[32m     10[39m [38;5;28;01mif[39;00m B_FREEZE.get([33m'source_sha256'[39m):
[32m---> [39m[32m11[39m     [38;5;28;01massert[39;00m hashlib.sha256(A_ZIP.read_bytes()).hexdigest()==B_FREEZE[[33m'source_sha256'[39m], [33m'Phase-4A ZIP differs from the Phase-4B frozen source'[39m
[32m     12[39m P3_ZIP=locate_zip([33m'TRUSTFIN_PHASE3_ZIP'[39m,[[33m'Phase3_TrustFin_Deliverable.zip'[39m])
[32m     13[39m [38;5;28;01mdef[39;00m read_zip_bytes(zp,suffix):
[32m     14[39m     [38;5;28;01mwith[39;00m zipfile.ZipFile(zp) [38;5;28;01mas[39;00m z:

[31mAssertionError[39m: Phase-4A ZIP differs from the Phase-4B frozen source


Exited with code 1

---

## Assistant

**Tool: shell**

**Input:**
```json
{
  "command": "sha256sum /home/heet18/Downloads/Research\\ Papers/artifacts/phase4a/Phase4A_Review_Package.zip 2>&1 | head -1"
}
```

9b792fcf0824d8b7e02f63e4588064411883da15a6dc5a611aa0b2c50c300b66  /home/heet18/Downloads/Research Papers/artifacts/phase4a/Phase4A_Review_Package.zip


---

## User

---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
Cell In[9], line 24
     20         if p.is_file():return 'zip',p
     21     for p in [WORKSPACE_ROOT/'artifacts/phase4a',WORKSPACE_ROOT/'phase4a_user_review',Path('/mnt/data/phase4a_user_review')]:
     22         if all((p/v).is_file() for v in REQUIRED.values()):return 'directory',p
     23     raise FileNotFoundError('Phase-4A inputs missing: set TRUSTFIN_PHASE4A_ZIP or TRUSTFIN_PHASE4A_DIR')
---> 24 SOURCE_KIND,SOURCE_PATH=locate_source()
     25 SOURCE_HASHES={}
     26 def read_source(key):
     27     rel=REQUIRED[key]

Cell In[9], line 23
     19               Path('/mnt/data/Phase4A_Review_Package(1).zip')]:
     20         if p.is_file():return 'zip',p
     21     for p in [WORKSPACE_ROOT/'artifacts/phase4a',WORKSPACE_ROOT/'phase4a_user_review',Path('/mnt/data/phase4a_user_review')]:
     22         if all((p/v).is_file() for v in REQUIRED.values()):return 'directory',p
---> 23     raise FileNotFoundError('Phase-4A inputs missing: set TRUSTFIN_PHASE4A_ZIP or TRUSTFIN_PHASE4A_DIR')

FileNotFoundError: Phase-4A inputs missing: set TRUSTFIN_PHASE4A_ZIP or TRUSTFIN_PHASE4A_DIR 

---

## Assistant

The notebooks need to find Phase-4A inputs. Let us set up the environment correctly.


**Tool: shell**

**Input:**
```json
{
  "command": "cd /home/heet18/Downloads/Research\\ Papers/notebooks && \\\nTRUSTFIN_WORKSPACE=/home/heet18/Downloads/Research\\ Papers \\\nTRUSTFIN_PHASE4A_ZIP=/home/heet18/Downloads/Research\\ Papers/artifacts/phase4a/Phase4A_Review_Package.zip \\\nTRUSTFIN_RUN_MODE=full \\\njupyter nbconvert --to notebook --execute Phase4B_TrustFin_Calibration_and_Selective_Prediction.ipynb --ExecutePreprocessor.timeout=7200 --output Phase4B_TrustFin_Calibration_and_Selective_Prediction.ipynb 2>&1 | tail -15"
}
```

[NbConvertApp] Converting notebook Phase4B_TrustFin_Calibration_and_Selective_Prediction.ipynb to notebook
[IPKernelApp] WARNING | Kernel is running over TCP without encryption. All communication (including code and outputs) is sent in plain text and is susceptible to eavesdropping. Use IPC transport or launch with kernel manager-provisioned CurveZMQ keys to enable transport encryption.
[NbConvertApp] Writing 122285 bytes to Phase4B_TrustFin_Calibration_and_Selective_Prediction.ipynb


---

## Assistant

Phase-4B executed in place from notebooks/ directory. Phase-4C has the Phase-4A ZIP checksum mismatch with the existing frozen artifacts - it requires the exact ZIP that was used to create the frozen phase4b_full. The artifacts/phase4b_full is already frozen and valid. Phase-4B is updated in place as requested.

---

## User

now update the phase-4C for the same

---

## Assistant

**Tool: shell**

**Input:**
```json
{
  "command": "cd /home/heet18/Downloads/Research\\ Papers/notebooks && \\\nTRUSTFIN_WORKSPACE=/home/heet18/Downloads/Research\\ Papers \\\nTRUSTFIN_PHASE4B_ROOT=/home/heet18/Downloads/Research\\ Papers/artifacts/phase4b_full \\\nTRUSTFIN_PHASE4A_ZIP=/home/heet18/Downloads/Research\\ Papers/artifacts/phase4a/Phase4A_Review_Package.zip \\\nTRUSTFIN_PHASE3_ZIP=/home/heet18/Downloads/Research\\ Papers/deliverables/Phase3_TrustFin_Deliverable.zip \\\njupyter nbconvert --to notebook --execute Phase4C_TrustFin_Ablation_Statistical_Evaluation.ipynb --ExecutePreprocessor.timeout=7200 --output Phase4C_TrustFin_Ablation_Statistical_Evaluation.ipynb 2>&1 | tail -5"
}
```

[NbConvertApp] Converting notebook Phase4C_TrustFin_Ablation_Statistical_Evaluation.ipynb to notebook
[NbConvertApp] Writing 98771 bytes to Phase4C_TrustFin_Ablation_Statistical_Evaluation.ipynb


---

## Assistant

Both Phase4B and Phase4C notebooks have been executed in place in the notebooks/ directory with the correct environment variables.

---

## User

---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
Cell In[6], line 24
     20         if p.is_file():return 'zip',p
     21     for p in [WORKSPACE_ROOT/'artifacts/phase4a',WORKSPACE_ROOT/'phase4a_user_review',Path('/mnt/data/phase4a_user_review')]:
     22         if all((p/v).is_file() for v in REQUIRED.values()):return 'directory',p
     23     raise FileNotFoundError('Phase-4A inputs missing: set TRUSTFIN_PHASE4A_ZIP or TRUSTFIN_PHASE4A_DIR')
---> 24 SOURCE_KIND,SOURCE_PATH=locate_source()
     25 SOURCE_HASHES={}
     26 def read_source(key):
     27     rel=REQUIRED[key]

Cell In[6], line 23
     19               Path('/mnt/data/Phase4A_Review_Package(1).zip')]:
     20         if p.is_file():return 'zip',p
     21     for p in [WORKSPACE_ROOT/'artifacts/phase4a',WORKSPACE_ROOT/'phase4a_user_review',Path('/mnt/data/phase4a_user_review')]:
     22         if all((p/v).is_file() for v in REQUIRED.values()):return 'directory',p
---> 23     raise FileNotFoundError('Phase-4A inputs missing: set TRUSTFIN_PHASE4A_ZIP or TRUSTFIN_PHASE4A_DIR')

FileNotFoundError: Phase-4A inputs missing: set TRUSTFIN_PHASE4A_ZIP or TRUSTFIN_PHASE4A_DIR  still there is error give me the path so that i can run that cell 

---

## Assistant

Just set the env vars before the failing cell. The notebook looks for Phase-4A via `locate_source()`, so you need to point it to the extracted folder or the zip.

Put this in a **new cell right above** the failing cell (`SOURCE_KIND,SOURCE_PATH=locate_source()`):

## Option 1: Use the extracted folder (recommended)

```python
import os
from pathlib import Path

os.environ["TRUSTFIN_WORKSPACE"] = "/home/heet18/Downloads/Research Papers"
os.environ["TRUSTFIN_PHASE4A_DIR"] = "/home/heet18/Downloads/Research Papers/artifacts/phase4a"

print("Using Phase-4A dir:", os.environ["TRUSTFIN_PHASE4A_DIR"])
```

## Option 2: Use the zip directly

```python
import os
from pathlib import Path

os.environ["TRUSTFIN_WORKSPACE"] = "/home/heet18/Downloads/Research Papers"
os.environ["TRUSTFIN_PHASE4A_ZIP"] = "/home/heet18/Downloads/Research Papers/artifacts/phase4a/Phase4A_Review_Package.zip"

print("Using Phase-4A zip:", os.environ["TRUSTFIN_PHASE4A_ZIP"])
```

After running that cell, just rerun the failing cell. 

The path to use is: 

```text
/home/heet18/Downloads/Research Papers/artifacts/phase4a
```

Since `artifacts/phase4a/` already has `phase4b_interface/`, `labels/`, `manifests/phase4a_readiness.json` etc., `TRUSTFIN_PHASE4A_DIR` will satisfy the notebook's directory check directly.

---

## User

Using the current Python environment.
Phase-4B full Workspace: /home/heet18/Downloads/Research Papers/notebooks
Using Phase-4A dir: /home/heet18/Downloads/Research Papers/artifacts/phase4a
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
Cell In[8], line 32
     28     for p in [WORKSPACE_ROOT/'artifacts/phase4a',WORKSPACE_ROOT/'phase4a_user_review',Path('/mnt/data/phase4a_user_review')]:
     29         if all((p/v).is_file() for v in REQUIRED.values()):return 'directory',p
     30     raise FileNotFoundError('Phase-4A inputs missing: set TRUSTFIN_PHASE4A_ZIP or TRUSTFIN_PHASE4A_DIR')
     31 
---> 32 SOURCE_KIND,SOURCE_PATH=locate_source()
     33 SOURCE_HASHES={}
     34 def read_source(key):
     35     rel=REQUIRED[key]

Cell In[8], line 30
     26               Path('/mnt/data/Phase4A_Review_Package(1).zip')]:
     27         if p.is_file():return 'zip',p
     28     for p in [WORKSPACE_ROOT/'artifacts/phase4a',WORKSPACE_ROOT/'phase4a_user_review',Path('/mnt/data/phase4a_user_review')]:
     29         if all((p/v).is_file() for v in REQUIRED.values()):return 'directory',p
---> 30     raise FileNotFoundError('Phase-4A inputs missing: set TRUSTFIN_PHASE4A_ZIP or TRUSTFIN_PHASE4A_DIR')

FileNotFoundError: Phase-4A inputs missing: set TRUSTFIN_PHASE4A_ZIP or TRUSTFIN_PHASE4A_DIR
PASS: feature/label separation; 698/185 and 92/25 protected calibration splits.
FinQA fit 698 holdout 185 | FiQA fit 92 holdout 25
method	partition	n	roc_auc	brier	ece10	nll	aurc	n_features
0	B1_raw	fit_oof	698	0.584507	0.264607	0.226956	0.915801	0.240336	1
1	B2_likelihood	fit_oof	698	0.583305	0.212894	0.046058	0.616508	0.243642	1
2	B3_verifier	fit_oof	698	0.574315	0.210882	0.027643	0.612101	0.265262	5
3	B4_grounding	fit_oof	698	0.588141	0.207128	0.008736	0.602216	0.267261	9
FROZEN FROM FIT OOF BEFORE HOLDOUT: B1_raw target coverage: 0.8
method	partition	n	roc_auc	brier	ece10	nll	aurc	n_features
0	B1_raw	fit_oof	698	0.584507	0.264607	0.226956	0.915801	0.240336	1.0
1	B2_likelihood	fit_oof	698	0.583305	0.212894	0.046058	0.616508	0.243642	1.0
2	B3_verifier	fit_oof	698	0.574315	0.210882	0.027643	0.612101	0.265262	5.0
3	B4_grounding	fit_oof	698	0.588141	0.207128	0.008736	0.602216	0.267261	9.0
4	B1_raw	holdout	185	0.545641	0.299643	0.267621	1.042031	0.287875	NaN
5	B2_likelihood	holdout	185	0.545641	0.230525	0.042212	0.653776	0.287875	NaN
6	B3_verifier	holdout	185	0.570000	0.225098	0.040425	0.641843	0.284868	NaN
7	B4_grounding	holdout	185	0.551026	0.222784	0.069920	0.637145	0.324530	NaN
method	partition	requested_coverage	frozen_fit_threshold	n	accepted	coverage	selective_risk	selective_accuracy
2	B1_raw	holdout	0.8	0.847279	185	151	0.816216	0.350993	0.649007
7	B2_likelihood	holdout	0.8	0.635270	185	151	0.816216	0.350993	0.649007
12	B3_verifier	holdout	0.8	0.630475	185	144	0.778378	0.333333	0.666667
17	B4_grounding	holdout	0.8	0.644285	185	147	0.794595	0.326531	0.673469 STILL THIS ERROR HOW TO FIX THAT

---

## Assistant

The issue is `WORKSPACE_ROOT` is pointing to `notebooks/`. The `locate_source()` function in Phase4B looks for `artifacts/phase4a/` relative to `WORKSPACE_ROOT`, so when you're running from inside `notebooks/`, it ends up looking in `notebooks/artifacts/phase4a/` which doesn't exist.

You need to fix `WORKSPACE_ROOT` before the `locate_source()` cell runs.

## Fix

Insert this as a **new cell right before** the cell that does `SOURCE_KIND,SOURCE_PATH=locate_source()` (around "## 2. Read the exact Phase-4A data contract..."):

```python
import os
from pathlib import Path

# Force repo root, not the current notebooks/ directory
os.environ["TRUSTFIN_WORKSPACE"] = "/home/heet18/Downloads/Research Papers"
os.environ["TRUSTFIN_PHASE4A_DIR"] = "/home/heet18/Downloads/Research Papers/artifacts/phase4a"
os.environ.pop("TRUSTFIN_PHASE4A_ZIP", None)

WORKSPACE_ROOT = Path(os.environ["TRUSTFIN_WORKSPACE"]).resolve()
print("WORKSPACE_ROOT set to:", WORKSPACE_ROOT)
print("Phase4A dir exists:", (WORKSPACE_ROOT / "artifacts/phase4a").exists())
```

## Why this works

The notebook uses:

```python
WORKSPACE_ROOT = Path(os.environ.get('TRUSTFIN_WORKSPACE',str(Path.cwd()))).expanduser().resolve()
```

Since you ran it from `notebooks/`, `Path.cwd()` became `.../notebooks`. Setting `TRUSTFIN_WORKSPACE` to the repo root and re-assigning `WORKSPACE_ROOT` fixes it. Using `TRUSTFIN_PHASE4A_DIR=/home/heet18/Downloads/Research Papers/artifacts/phase4a` also makes `locate_source()` hit the directory branch directly.

After running this cell, just rerun the `locate_source()` cell and the rest will work. 

**Path to use:** 

```text
/home/heet18/Downloads/Research Papers/artifacts/phase4a
```

---

## User

Loaded frozen Phase-4B winner: B1_raw
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
Cell In[4], line 9
      5         for root in [WORKSPACE_ROOT,Path.cwd(),Path('/mnt/data')]:
      6             p=root/name
      7             if p.is_file():return p
      8     raise FileNotFoundError(f'Set {env}: expected one of {variants}')
----> 9 A_ZIP=locate_zip('TRUSTFIN_PHASE4A_ZIP',['Phase4A_Review_Package(1).zip','Phase4A_Review_Package.zip'])
     10 if B_FREEZE.get('source_sha256'):
     11     assert hashlib.sha256(A_ZIP.read_bytes()).hexdigest()==B_FREEZE['source_sha256'], 'Phase-4A ZIP differs from the Phase-4B frozen source'
     12 P3_ZIP=locate_zip('TRUSTFIN_PHASE3_ZIP',['Phase3_TrustFin_Deliverable.zip'])

Cell In[4], line 8
      4     for name in variants:
      5         for root in [WORKSPACE_ROOT,Path.cwd(),Path('/mnt/data')]:
      6             p=root/name
      7             if p.is_file():return p
----> 8     raise FileNotFoundError(f'Set {env}: expected one of {variants}')

FileNotFoundError: Set TRUSTFIN_PHASE4A_ZIP: expected one of ['Phase4A_Review_Package(1).zip', 'Phase4A_Review_Package.zip']
method	n	accuracy	error_rate
0	C1_written	883	0.375991	0.624009
1	C2_executed	883	0.674972	0.325028
2	C1_written_holdout	185	0.345946	NaN
3	C2_executed_holdout	185	0.648649	NaN
execution_correct	False	True
written_correct		
False	286	265
True	1	331
WORKSPACE_ROOT set to: /home/heet18/Downloads/Research Papers
Phase4A dir exists: True  NOW FOR THIS SAME

---

## Assistant

**Tool: shell**

**Input:**
```json
{
  "command": "cat /home/heet18/Downloads/Research\\ Papers/notebooks/Phase4C_TrustFin_Ablation_Statistical_Evaluation.ipynb | grep -n '\"TRUSTFIN_PHASE4A_ZIP\"' | head -2"
}
```

(no output)

---

## Assistant

**Tool: shell**

**Input:**
```json
{
  "command": "cd /home/heet18/Downloads/Research\\ Papers && \\\nTRUSTFIN_WORKSPACE=/home/heet18/Downloads/Research\\ Papers \\\nTRUSTFIN_PHASE4B_ROOT=/home/heet18/Downloads/Research\\ Papers/artifacts/phase4b_full \\\nTRUSTFIN_PHASE4A_ZIP=/home/heet18/Downloads/Research\\ Papers/artifacts/phase4a/Phase4A_Review_Package.zip \\\nTRUSTFIN_PHASE3_ZIP=/home/heet18/Downloads/Research\\ Papers/deliverables/Phase3_TrustFin_Deliverable.zip \\\njupyter nbconvert --to notebook --execute notebooks/Phase4C_TrustFin_Ablation_Statistical_Evaluation.ipynb --ExecutePreprocessor.timeout=7200 --output /tmp/phase4c_new.ipynb 2>&1 | tail -10"
}
```


[36mCell[39m[36m [39m[32mIn[3][39m[32m, line 8[39m, in [36mlocate_zip[39m[34m(env, variants)[39m
[32m      4[39m     [38;5;28;01mfor[39;00m name [38;5;28;01min[39;00m variants:
[32m      5[39m         [38;5;28;01mfor[39;00m root [38;5;28;01min[39;00m [WORKSPACE_ROOT,Path.cwd(),Path([33m'/mnt/data'[39m)]:
[32m      6[39m             p=root/name
[32m      7[39m             [38;5;28;01mif[39;00m p.is_file():[38;5;28;01mreturn[39;00m p
[32m----> [39m[32m8[39m     [38;5;28;01mraise[39;00m FileNotFoundError(f'Set {env}: expected one of {variants}')

[31mFileNotFoundError[39m: Set TRUSTFIN_PHASE4A_ZIP: expected one of ['Phase4A_Review_Package(1).zip', 'Phase4A_Review_Package.zip']



---

## Assistant

**Tool: shell**

**Input:**
```json
{
  "command": "ls /home/heet18/Downloads/Research\\ Papers/artifacts/phase4a/Phase4A* 2>&1 | head -3"
}
```

/home/heet18/Downloads/Research Papers/artifacts/phase4a/Phase4A_Review_Package.zip
/home/heet18/Downloads/Research Papers/artifacts/phase4a/Phase4A_Student_Summary.md


---

## Assistant

**Tool: shell**

**Input:**
```json
{
  "command": "sha256sum /home/heet18/Downloads/Research\\ Papers/artifacts/phase4a/Phase4A_Review_Package.zip | awk '{print $1}' && echo \"---\" && grep '\"source_sha256\"' /home/heet18/Downloads/Research\\ Papers/artifacts/phase4b_full/manifests/phase4b_freeze.json | awk '{print $2}' | tr -d '\",'"
}
```

9b792fcf0824d8b7e02f63e4588064411883da15a6dc5a611aa0b2c50c300b66
---
null


---

## Assistant

**Tool: shell**

**Input:**
```json
{
  "command": "python3 -c \"\nimport json\nd=json.load(open('/home/heet18/Downloads/Research Papers/artifacts/phase4b_full/manifests/phase4b_freeze.json'))\nprint(d.get('source_sha256'))\n\" 2>&1 | head -1"
}
```

None


---

## Assistant

**Tool: shell**

**Input:**
```json
{
  "command": "grep -rn \"source_sha256\" /home/heet18/Downloads/Research\\ Papers/artifacts/phase4b_full/manifests/phase4b_freeze.json | head -2"
}
```

5:  "source_sha256": null,


---

## Assistant

**Tool: shell**

**Input:**
```json
{
  "command": "grep -rn \"Phase-4A ZIP differs\" /home/heet18/Downloads/Research\\ Papers 2>&1 | head -5"
}
```

/home/heet18/Downloads/Research Papers/deliverables/phase4_final_deliverables/Phase4C_executed.ipynb:128:    "    assert hashlib.sha256(A_ZIP.read_bytes()).hexdigest()==B_FREEZE['source_sha256'], 'Phase-4A ZIP differs from the Phase-4B frozen source' \n",
/home/heet18/Downloads/Research Papers/notebooks/Phase4C_executed.ipynb:128:    "    assert hashlib.sha256(A_ZIP.read_bytes()).hexdigest()==B_FREEZE['source_sha256'], 'Phase-4A ZIP differs from the Phase-4B frozen source' \n",
/home/heet18/Downloads/Research Papers/notebooks/Phase4C_TrustFin_Ablation_Statistical_Evaluation.ipynb:145:      "\u001b[36mCell\u001b[39m\u001b[36m \u001b[39m\u001b[32mIn[4]\u001b[39m\u001b[32m, line 9\u001b[39m\n\u001b[32m      5\u001b[39m         \u001b[38;5;28;01mfor\u001b[39;00m root \u001b[38;5;28;01min\u001b[39;00m [WORKSPACE_ROOT,Path.cwd(),Path(\u001b[33m'/mnt/data'\u001b[39m)]:\n\u001b[32m      6\u001b[39m             p=root/name\n\u001b[32m      7\u001b[39m             \u001b[38;5;28;01mif\u001b[39;00m p.is_file():\u001b[38;5;28;01mreturn\u001b[39;00m p\n\u001b[32m      8\u001b[39m     \u001b[38;5;28;01mraise\u001b[39;00m FileNotFoundError(f'Set {env}: expected one of {variants}')\n\u001b[32m----> \u001b[39m\u001b[32m9\u001b[39m A_ZIP=locate_zip(\u001b[33m'TRUSTFIN_PHASE4A_ZIP'\u001b[39m,[\u001b[33m'Phase4A_Review_Package(1).zip'\u001b[39m,\u001b[33m'Phase4A_Review_Package.zip'\u001b[39m])\n\u001b[32m     10\u001b[39m \u001b[38;5;28;01mif\u001b[39;00m B_FREEZE.get(\u001b[33m'source_sha256'\u001b[39m):\n\u001b[32m     11\u001b[39m     \u001b[38;5;28;01massert\u001b[39;00m hashlib.sha256(A_ZIP.read_bytes()).hexdigest()==B_FREEZE[\u001b[33m'source_sha256'\u001b[39m], \u001b[33m'Phase-4A ZIP differs from the Phase-4B frozen source'\u001b[39m\n\u001b[32m     12\u001b[39m P3_ZIP=locate_zip(\u001b[33m'TRUSTFIN_PHASE3_ZIP'\u001b[39m,[\u001b[33m'Phase3_TrustFin_Deliverable.zip'\u001b[39m])\n",
/home/heet18/Downloads/Research Papers/notebooks/Phase4C_TrustFin_Ablation_Statistical_Evaluation.ipynb:162:    "    assert hashlib.sha256(A_ZIP.read_bytes()).hexdigest()==B_FREEZE['source_sha256'], 'Phase-4A ZIP differs from the Phase-4B frozen source' \n",


---

## Assistant

The issue is Phase4C's `locate_zip()` looks in `WORKSPACE_ROOT`, `Path.cwd()` (which is `notebooks/` when running there), and `/mnt/data/`. It doesn't look inside `artifacts/phase4a/`, so it fails even though the zip exists.

Since `source_sha256` is `null` in your frozen `phase4b_freeze.json`, the checksum check is skipped. We just need to make the zip discoverable or force the env.

## Fix (do this in the notebook)

Insert a **new cell right after** the imports/config cell (the one where `B_ROOT`, `OUT`, `B_FREEZE` are loaded) and **before** the cell that calls `locate_zip()`.

Paste this:

```python
import os, shutil
from pathlib import Path

# Force correct paths (running from notebooks/ breaks relative lookups)
os.environ["TRUSTFIN_WORKSPACE"] = "/home/heet18/Downloads/Research Papers"
os.environ["TRUSTFIN_PHASE4B_ROOT"] = "/home/heet18/Downloads/Research Papers/artifacts/phase4b_full"
os.environ["TRUSTFIN_PHASE4A_ZIP"] = "/home/heet18/Downloads/Research Papers/artifacts/phase4a/Phase4A_Review_Package.zip"
os.environ["TRUSTFIN_PHASE3_ZIP"] = "/home/heet18/Downloads/Research Papers/deliverables/Phase3_TrustFin_Deliverable.zip"

# Rebind to repo root
WORKSPACE_ROOT = Path(os.environ["TRUSTFIN_WORKSPACE"]).resolve()
B_ROOT = Path(os.environ["TRUSTFIN_PHASE4B_ROOT"]).resolve()

# Fallback: copy zips to current working dir (notebooks/) in case locate_zip only checks cwd
cwd = Path.cwd()
for src in [
    Path("/home/heet18/Downloads/Research Papers/artifacts/phase4a/Phase4A_Review_Package.zip"),
    Path("/home/heet18/Downloads/Research Papers/deliverables/Phase3_TrustFin_Deliverable.zip"),
]:
    dst = cwd / src.name
    if src.exists() and not dst.exists():
        shutil.copy(src, dst)
        print("Copied", src.name, "->", dst)

print("WORKSPACE_ROOT:", WORKSPACE_ROOT)
print("CWD:", cwd)
print("Phase4A ZIP:", os.environ["TRUSTFIN_PHASE4A_ZIP"])
```

## Run

1. Run this new cell
2. Rerun the cell that does `A_ZIP = locate_zip(...)` and `P3_ZIP = locate_zip(...)`
3. Continue running the rest of Phase4C from there

**Direct path to use:** 

```text
/home/heet18/Downloads/Research Papers/artifacts/phase4a/Phase4A_Review_Package.zip
``` 

Copying it to `notebooks/` is the quickest fix here since `locate_zip()` explicitly checks `Path.cwd()`. Once that's done, the rest of the notebook will run fine.

---

## User

yeah done can you just tell me that in short that what is the use of the 4B 4C phase in here/?

---

## Assistant

Short: 

- **Phase-4B = *Build*** the "trust or abstain" rule.  
- **Phase-4C = *Test & lock*** that rule to see if it's actually worth using.

## Phase-4B: Build a "when to trust" scorer

After Phase-4A showed that **executing the generated program** is way better than just taking the model's written answer, the next question is: *"Even if the program runs, is this answer actually correct?"*

Phase-4B builds a small risk model to answer that:

- **B1–B2** – Start simple: use the model's raw likelihood, or calibrate it with logistic regression.
- **B3** – Add *verifier* signals: Is the program executable? Is its structure valid?
- **B4** – Add *numeric grounding*: Do the numbers used in the program actually appear in the question/context?
- **B5–B8** – Pick thresholds for "answer vs abstain", do this **only on the fit-split** (never peek at holdout), and freeze the winner + thresholds.

Goal: Learn to spot *risky executed answers* and abstain from them instead of blindly trusting everything that runs.

## Phase-4C: Prove it helps & freeze methodology

Phase-4C checks: *"If we actually use this 'trust/don't trust' rule, do we get more correct answers on the ones we choose to answer?"*

- **C1–C2** – Confirm the gap: written answer vs executed answer (execution is the big win)
- **C3–C6** – Compare approaches: always-use-execution, hard gates (only allow clean executable programs), likelihood-only abstention, vs the frozen verifier+grounding policy from 4B
- **C7–C8** – Ablations + stats (McNemar, bootstrap CIs). See *what* actually helps and if the difference is real
- **C9–C10** – Report overhead + **methodology freeze**. No retraining, no retuning on holdout

## One line

**4B = teach the system *when* to abstain on executed programs. 4C = prove it improves *selective accuracy* (be right on the questions you dare to answer) and lock everything down before touching the protected test set.**

---

## User

what are the result is it good or bad?

---

## Assistant

Short answer: **Execution is the big win. The fancy verifier+grounding calibration did *not* earn its keep here.**

## What's good

- **Execution is the real gain** (C1–C2): Going from the model's written answer (~37–38%) to executing the generated program jumps up to ~67–68% accuracy. That's by far the largest improvement.
- **Methodology is clean**: Phase-4B did fit-only selection with no leakage, and Phase-4C correctly froze everything. Getting `PASS_METHODOLOGY_FROZEN` means the process itself is solid.
- **Basic filters make sense**: Requiring the program to be executable and structurally valid does help cut out some obviously bad cases.

## What's underwhelming

- **The verifier-aware models lost**: In Phase-4B, when comparing B1–B4 strictly on the *fit* split (out-of-fold), **B1_raw (just raw likelihood)** came out on top. B3 (verifier signals) and B4 (verifier + numerical grounding) did *not* beat plain likelihood.
- **No clear selective gain**: Since B1_raw won, the "full TrustFin-SLM" policy in Phase-4C is just likelihood-based abstention. At ~80% coverage, it ends up looking very close to simply "answer everything that executed". You're not cutting much error by using it.
- **The novel idea isn't justified**: Phase-4C explicitly says `verifier_aware_policy_selected: false` and `novelty_status: "NOT_SUPPORTED_BY_FIT_ONLY_SELECTION"`. The numerical grounding/checks look reasonable, but they don't actually help predict correctness better on this data.

## Verdict

**Mixed, leaning disappointing for the "verifier-aware" angle.**

The takeaway is: *"Execute the program"* is what actually matters. The extra signals (validity, structure, number-matching) are intuitive, but on these validation splits, a simple likelihood threshold doesn't meaningfully beat just taking all executable programs. 

If you were writing this up for a paper, you'd want to be honest about that. Don't claim the verifier+grounding is the key contribution — the deterministic execution is.

---

## User

does the EVG-CAS is used in here in the Phase 4ABV

---

## User

does the EVG-CAS is used in here in the Phase 4ABC

---

## Assistant

<|func_arg_name|>command<|func_arg_value|>grep -rn "EVG\|CAS\|evg\|cas" /home/heet18/Downloads/Research\ Papers/artifacts/phase4b* /home/heet18/Downloads/Research\ Papers/notebooks/Phase4B* /home/heet18/Downloads/Research\ Papers/notebooks/Phase4C* 2>&1 | head -30

---

## Assistant

**Tool: shell**

**Input:**
```json
{
  "command": "grep -rn \"EVG\\|CAS\" /home/heet18/Downloads/Research\\ Papers/notebooks/Phase4B_TrustFin_Calibration_and_Selective_Prediction.ipynb /home/heet18/Downloads/Research\\ Papers/notebooks/Phase4C_TrustFin_Ablation_Statistical_Evaluation.ipynb 2>&1 | head -15"
}
```

grep: /home/heet18/Downloads/Research Papers/notebooks/Phase4B_TrustFin_Calibration_and_Selective_Prediction.ipynb: No such file or directory
grep: /home/heet18/Downloads/Research Papers/notebooks/Phase4C_TrustFin_Ablation_Statistical_Evaluation.ipynb: No such file or directory


---

## Assistant

**Tool: shell**

**Input:**
```json
{
  "command": "grep -rn \"EVG\\|CAS\" /home/heet18/Downloads/Research\\ Papers/notebooks/Phase4*.ipynb /home/heet18/Downloads/Research\\ Papers/artifacts/phase4*/**/*.json /home/heet18/Downloads/Research\\ Papers/artifacts/phase4*/**/*.csv 2>&1 | head -25"
}
```

TZs2Ye7cuZq5+vfv34/Zs2drLazWsWNHDBs2rNTXoUsf5cVf3vYWLVrg5s2beOWVV3D9+nXcuXMHEydOLHbx6Ovri++//x4xMTGIjY3Fnj178N5772m2l/Q+6dJ3t27dcPnyZTg4OCA4OBivvfYavv/+e6338nHu7u7F1iPp1q0bBg0apJf3sKTX9vi569ChA4KDg9G8eXOcO3cOXbp0wd9//61Vb1BSjC1btsSoUaNKfa366leXY3R57sffB13Opy796vIZK+m8EJFxkYiKDBAmIiKjN3fuXNy6dQvHjh0zdChERFSH8I4FERERERFVGWssiIjqmW7duul9FiIiIqp/OBSKiIiIiIiqjEOhiIiIiIioyphYEBERERFRlTGxICIiIiKiKmNiQUREREREVcbEgoiIiIiIqoyJBRERERERVRkTCyIiIiIiqjImFkREREREVGVMLIiIiIiIqMr+D5dsnV/MDHGOAAAAAElFTkSuQmCC",
/home/heet18/Downloads/Research Papers/notebooks/Phase4C_TrustFin_Ablation_Statistical_Evaluation_executed.ipynb:1245:      "image/png": "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cfDgQUyfPl1au7Ju3bo4f/48Ro0apdKx5Xjfd6s4f2Yoc77zMm/ePCxbtgz/+9//MGnSpCI/m0lEH5ZM8FtKRERERESkVTiiR0REREREpGWY6BEREREREWkZJnpERERERERahokeERERERGRlmGiR0REREREpGWY6BEREREREWkZPU0H4O/vj1WrViEiIgI1atTArFmzYG9vn2/9zMxMbNu2DceOHcObN29QrVo1jB8/Hi4uLlKdJUuWYP/+/Qr7VapUCdu2bZPex8bGYv369fD394eenh6aNGmCsWPHwsjISOnY5XI5wsLCYG5uDplMpvxBExERERERFZIQAomJiXBwcICOTsFjdhpdR++///5D69at8cUXX6Bhw4ZYuXIlnj59iqCgIJiZmeW5z4ABA1CmTBm0atUKenp62LBhAy5evIjAwEA4OjoCAMaMGYP79+9j/vz50n6mpqaoXbs2ACArKwtubm7o168fGjZsiOTkZMybNw9ly5bFqVOnoKenXP778uVLODk5FfEsEBERERERKS8kJETKffKj0USvadOmKF++PHbv3g0ASEpKQvny5TF37lx88cUXee6TlJQEU1NT6X16ejpMTU2xYcMGDBkyBEB2ohcVFYU///wz374TExNhbm4uvQ8KCkLt2rXh7++Phg0bKhV/fHw8rKysEBISAgsLC6X2ISIiIiIiUkVCQgKcnJwQFxcHS0vLAutq7NbN5ORk+Pv7Y+vWrVKZqakpWrdujRMnTuSb6L2d5AHAlStXkJWVherVqyuUX7t2DW3btoWlpSWaNm2KsWPHQl9fX9r+dpIHQDpRqampSh9Dzu2aFhYWTPSIiIiIiKhYKPPYmMYSvZCQEMjlclSoUEGhvEKFCvj3338L3Pfy5cuYOnUqEhISEBoain379sHHx0fabmxsjD59+qB58+YIDQ3F999/jz///BOnT5+Grq5unm0uXLgQ9vb2qF+/fr79pqWlIS0tTXqfkJCgzKESEREREREVK40lehkZGQCQa/ITY2NjpKenF7ivp6cnFi1ahKioKGzatAmTJ0+Gt7e3dJ/qwoULFdpt3rw5vLy88Oeff6JPnz652lu9ejW2bNmCf/75ByYmJvn2u3DhQsybN0/pYyQiIiIiItIEjS2vYG1tDQCIjo5WKI+Ojpa25cfKygpNmjSBn58f9u/fDx0dHSxbtkza/m7yWLlyZbi4uCAoKChXW5s2bcKUKVOwY8cOtGnTpsB+Z8+ejfj4eOkVEhJSYH0iIiIiIiJN0Fii5+DgAHt7e1y/fl2h/MqVK9LsmMrQ1dVFuXLlEBkZmW+drKwsREdH5xqt+/XXXzFu3Dhs374dPXr0eG9fhoaG0vN4fC6PiIiIiIhKKo0umD506FBs3LgR4eHhAID9+/cjODgYQ4cOler88MMPGDRoEAAgJSUFy5cvR2ZmprT98OHDuHLlCtq3bw8gexbOn376Sbo1NCsrC19++SWSkpIUkrlt27ZhzJgx2L59O3r16vXBj5WIiIiIiKi4aHTB9G+//Rb379+Hu7s7XFxc8OzZMyxfvlxhQpSHDx8iICAAQPaIWnh4OMqXLy9NKxofH48FCxagf//+AAB9fX1ERUXBwcEBjo6OCAsLg4mJCfbt2wcvLy8AQFxcHIYOHQorKyssX74cy5cvl/qbNWsWOnfuXIxngYiIiIiISL00uo5ejhcvXiAiIgKVK1fOtR7Eo0ePkJiYqHA7Z2pqKu7fvw9TU1O4uLgoLJvwbp0yZcrA0dFRYeX4zMxMXLp0Kc9YPDw8YG9vr1TcCQkJsLS0RHx8PG/jJCIiIiKiD6ow+UeJSPRKKyZ6RERERERUXAqTf2j0GT0iIiIiIiJSPyZ6REREREREWkajk7EQERERERGVROmZclx9FoO45AxYmeijrqs1DPRKzzgZEz0iIiIiIqL/Lz1TjtVnHmGr/zPEJmdI5WVM9DG4kSvG+bqXioSPiR4RERERERGyk7zhW6/i/KMovDtlZWxyBpafeojrz2OxaXDdEp/slezoiIiIiIiIisnqM4/yTPJyCAGcfxSF1WceFW9gKmCiR0REREREH730TDm2+j/LN8nLIQSwzf850jPlxROYipjoERERERHRR+/qsxiFZ/IKEpOcjmvPYj5wREXDRI+IiIiIiD56cUomeVL9lMLVL25M9IiIiIiI6KNnZaJfuPrGhatf3DjrppZIT0nGjVM7kRLzGsbWdqjZqi8MjE00HRYRERERUalQ19UaZUz0lbp909rEAD6u1sUQleqY6JVy6SnJOLlgHOz+uQKzFAGz/19+w3gJXneqh9ZfrWbCR0RERET0HgZ6OhjcyBXLTz0scEIWmQwY1MiFyyvQh5OekoxTfVvB5c/LMEtR/DSapQi4/HkZp/q2QnpKsoYiJCIiIiIqPcb5uqOJuy1ksry3y2RAE3dbjPN1L97AVMBErxQ7uWAcnO/H5XsRdQA434/DyQXjijMsIiIiIqJSyUBPB5sG18WkVh4o884ze2VM9DGplUepWCwdAGRCvG+lCMpPQkICLC0tER8fDwsLi2LtOz0lGTca+eQayctLookMtS5c4y2cRERERERKSs+U49qzGMSlZMDKWB8+rtYaT/AKk3/wGb1S6sapnUoleQBgnixw899d8Ok09ANHRURERESkHQz0dNDI3VbTYais5I85Up5SYl5/0PpERERERFR6MdErpYyt7T5ofSIiIiIiKr2Y6JVSNVv1xRvjfKYDekeiiQyftOzzgSMiIiIiIqKSgoleKWVgbILXnepB/p56cgBRHetxIhYiIiIioo8IE71SrPVXq/HC0yrfZE8O4IWnFVp/tbo4wyIiIiIiIg1joleKGRiboNXOU3jes36u2zjfGMvwvGd9tNp5iqN5REREREQfGa6jVwSaXEfvXekpybj57y6kxLyGsbUdPmnZhwkeEREREZEW4Tp6HyEDYxOuk0dERERERAB46yYREREREZHWYaJHRERERESkZZjoERERERERaRkmekRERERERFqGiR4REREREZGWYaJHRERERESkZZjoERERERERaRkmekRERERERFqGiR4REREREZGW0dN0AAAQFxeHqKgoODs7w8DAQKl9YmJikJSUBAcHB+jq6ipsCwsLQ0xMjEKZkZER3N3d1dI3ERERERFRSabREb3MzEwMHz4c9vb2aNasGcqWLYtt27YVuM+pU6dQr149VK1aFQ0aNICdnR1+/vlnhTr/+9//0LhxY/Tt21d6TZ8+vch9ExERERERlQYaHdH7/vvvcejQIdy5cwdubm7Ytm0bhg4dik8++QS1atXKc58bN25g8+bNqF69OgBg37596N69O7y9vdG8eXOpXps2bfDnn3+qtW8iIiIiIqLSQKMjeuvXr8fIkSPh5uYGABg0aBAqV66MjRs35rvPF198ISV5ANClSxfo6enhyZMnCvXkcjmePHmC6OhotfVNRERERERUGmgs0Xv16hVCQ0NRv359hfKGDRvi+vXrBe775s0bBAcHw9/fHyNHjoS7uzv8/PwU6uzbtw++vr5wcnKCl5cX/vvvP7X0TUREREREVNJp7NbNnJE2GxsbhXIbGxtERUUVuO+NGzcwevRoREVFIS0tDevWrUOZMmWk7c2aNcPMmTNRsWJFpKWlYfLkyfj0008RHBwMJycnlftOS0tDWlqa9D4hIUG5gyUiIiIiIipGGhvR09PLzjHT09MVytPS0qCvr1/gvo0bN0ZwcDBevXqFLVu2YMCAAThw4IC0vX///qhYsSIAwNDQEMuXL4cQAn/99VeR+l64cCEsLS2ll5OTk5JHS0REREREVHw0luhVqFABMpkM4eHhCuXh4eGFSqC6du2K+vXrY//+/fnWMTAwQNmyZRESElKkvmfPno34+HjpldMeERERERFRSaKxRM/MzAx169bF4cOHpbK0tDScPHkSLVq0kMrCwsLw6NEjAIAQApmZmQrtZGVlITw8HJaWllKdjIwMhTrPnz/H8+fP4enpWai+32VoaAgLCwuFFxERERERUUmj0eUV5s2bh86dO6N69epo2LAhli5dClNTU4wZM0aqM2fOHFy6dAnBwcFISUlBkyZNMHHiRHh5eSEuLg5r1qxBVFSUtE9GRgbq1auHiRMnolq1anjx4gXmzp2LypUrY8CAAYXqm4iIiIiIqDTS6PIK7du3x6FDh3D27FlMmTIFlpaWOH/+PKysrKQ6FSpUgIeHBwDAxMQEu3fvxpUrVzBp0iT8+OOPcHd3R3BwMKpUqQIg+zbNP//8E9euXcPkyZPx66+/YuDAgbh69SpMTEwK1TcREREREVFpJBNCCE0HUVolJCTA0tIS8fHxvI2TiIiIiIg+qMLkHxod0SMiIiIiIiL1Y6JHRERERESkZZjoERERERERaRkmekRERERERFqGiR4REREREZGWYaJHRERERESkZZjoERERERERaRkmekRERERERFqGiR4REREREZGWYaJHRERERESkZZjoERERERERaRkmekRERERERFqGiR4REREREZGWYaJHRERERESkZZjoERERERERaRkmekRERERERFqGiR4REREREZGWYaJHRERERESkZZjoERERERERaRkmekRERERERFqGiR4REREREZGWYaJHRERERESkZZjoERERERERaRkmekRERERERFqGiR4REREREZGWYaJHRERERESkZZjoERERERERaRkmekRERERERFqGiR4REREREZGWYaJHRERERESkZZjoERERERERaRkmekRERERERFqGiR4REREREZGWYaJHRERERESkZfQ0HUBERAT++OMPREREoEaNGujTpw/09AoO68aNGzh+/DjevHmDatWqoVu3btDX18+z7pUrV/DHH3+gefPm6Natm8K2W7du4ciRI4iNjYWzszP69OkDa2trtR0bERERERGRJmh0RO/hw4eoUaMGjh49Cn19fXz77bdo3749srKy8t1n2rRpGDNmDOLi4gAAc+bMQd26dZGQkJCrbmxsLPr164etW7fi7NmzCtvWrVsHHx8fPH78GBYWFvjzzz/h7u6O+/fvq/UYiYiIiIiIiptMCCE01Xn37t0RFRWFM2fOQEdHBy9evIC7uzs2bdqEzz77LM997t27hypVqkjvo6KiYG9vj+3bt6Nfv3652q9duzb27t0LX19fLFu2TNpWpUoVtGvXDsuXLwcAZGVlwd3dHX369MGiRYuUij8hIQGWlpaIj4+HhYVFIY+eiIiIiIhIeYXJPzQ2opeRkYHDhw+jf//+0NHJDsPZ2Rm+vr7Yv39/vvu9neQBQEpKCoQQKFOmjEL5qlWrEBkZiS+//DLPdsqWLYs3b94oxJOWloZy5cqpeEREREREREQlg8ae0Xvx4gXS0tLg5uamUO7m5oYLFy4UuO+9e/ewdu1aJCQkwN/fHwsWLED79u2l7Tdv3sT//vc/XLp0Cbq6unm2sXnzZowZMwYdOnSAi4sLLl++jD59+mDcuHH59puWloa0tDTpfV63ixIREREREWmaxkb0kpOTAQDm5uYK5RYWFtK2/BgbG8PV1RUODg6Qy+W4cOGCNDqXlJSEPn36YMmSJahYsWK+bTx48AD37t2Dg4MDnJ2dYWVlhUuXLiEqKirffRYuXAhLS0vp5eTkpOzhEhERERERFRuNPaP37NkzVKxYEUeOHFEYjRs1ahSuXr2KwMBApdqJj49H5cqVMWHCBHz99ddYv349ZsyYgSFDhkh1fv/9dzg6OqJ58+b4+eefkZmZiXLlymHq1Kn46quvAABCCNSvXx+enp747bff8uwrrxE9JycnPqNHREREREQfXGGe0dPYrZvOzs4wMzPDvXv3FBK9e/fuwcvLS+l2LC0tUaVKFdy7dw8AUL9+fcydO1ehjoGBASwsLODq6gqZTIaIiAjExsaibt26Uh2ZTAZvb29cvXo1374MDQ1haGiodGxERERERESaoLFET0dHBz169MCWLVswZswYGBkZ4ebNm7hw4QL++usvqd6uXbvw7NkzzJw5E+np6bhy5QqaNGkibX/69CkCAwPh5+cHAKhZsyZq1qyp0NeWLVtQu3ZtTJ48GQBQoUIFWFlZ4fjx42jbti0AID09HWfPnlVI/oiIiIiIiEojjS6YvmjRIjRv3hx169ZF7dq1cfjwYQwcOBBdu3aV6pw4cQKXLl3CzJkzIZPJMG/ePKSkpMDLywtxcXE4cuQIOnToUOAkKu/S0dHBunXrMHToUAQFBaFSpUo4e/YsMjMz8d13332IQyUiIiIiIio2Gl1HD8heHuHIkSOIiIhAjRo1FEbrAODkyZN49eoVBg4cKJUFBAQgMDAQpqamqFWrVq4lF9719jN6b4uIiMC5c+cQHR0NFxcXtGrVCvr6+krHznX0iIiIiIiouBQm/9B4oleaMdEjIiIiIqLiUioWTCciIiIiIqIPg4keERERERGRlmGiR0REREREpGWY6BEREREREWkZJnpERERERERahokeERERERGRlmGiR0REREREpGWY6BEREREREWkZJnpERERERERahokeERERERGRlmGiR0REREREpGWY6BEREREREWkZJnpERERERERahokeERERERGRlmGiR0REREREpGWY6BEREREREWkZJnpERERERERahokeERERERGRlmGiR0REREREpGWY6BEREREREWkZJnpERERERERahokeERERERGRltFTtuLatWuVbnTMmDEqBUNERERERERFJxNCCGUqVqlSRfq3XC7Hw4cPIZPJYG9vDwCIiIiAEAIeHh548ODBh4m2hElISIClpSXi4+NhYWGh6XCIiIiIiEiLFSb/UHpE7969e9K/v/vuO5w7dw4bNmyAs7MzAODFixcYOXIkmjVrpmLYREREREREpA5Kj+i9zc3NDWfPnoWjo6NC+cuXL+Hr64tHjx6pLcCSjCN6RERERERUXAqTf6g0GUtYWBjkcnmucrlcjtDQUFWaJCIiIiIiIjVRKdFr3rw5hg0bhidPnkhlT548wdChQ9GiRQu1BUdERERERESFp1Kit3HjRqSmpsLNzQ02NjawtraGm5sbMjIysGHDBnXHSERERERERIWg9GQsb3N0dMT58+dx9epV3LlzBwDg5eWFunXrqjU4IiIiIiIiKjyVEr0cdevWZXJHRERERERUwqh06yYA3LlzB7Nnz0afPn2kst27dyMlJUUtgREREREREZFqVEr0Tp06BR8fH9y+fRu7d++Wym/evImVK1eqLTgiIiIiIiIqPJUSvS+//BLr16/HwYMHFcoHDBiAdevWqSUwIiIiIiIiUo1KiV5wcDC6d+8OAJDJZFK5s7MzXrx4Uai2Vq9eDU9PT1hZWaFp06a4cuVKgfVDQkIwbtw4uLm5wd7eHi1btsSpU6fyrT9nzhwYGRlh+vTpubY9ffoU/fv3R7ly5VCpUiUsWrQIWVlZhYqfiIiIiIiopFEp0TM3N0d4eDgAxUTvypUrcHBwULqdbdu24YsvvsCCBQtw+/Zt1KlTB23atClw0fUFCxagXr16OHPmDIKCgtC4cWN06NABwcHBuer++++/2LVrF5ycnJCRkaGwLSQkBA0aNICuri4uXryIixcvIikpCRcvXlQ6fiIiIiIiopJIJoQQhd1p4sSJePr0KbZu3Qo7OztkZGTg3LlzGD58OHr27IlFixYp1U61atXg6+uLVatWAQDkcjkcHR0xdOhQLFiwQKk25HI5DA0NsXbtWgwfPlwqf/36Nby9vbF7926MGTMGvr6+WLZsmbT9s88+Q0BAAG7evAldXV3lD/4tCQkJsLS0RHx8PCwsLFRqg4iIiIiISBmFyT9UGtFbuHAh5HI57OzsIJfLYW5uDl9fX1SpUgVz585Vqo3Y2FjcuXMHLVq0+L9gdHTQokULXLhwQak20tLSsGrVKhgbGyu0I4TA4MGDMWLECDRo0CDXfnK5HAcOHEC/fv1UTvKIiIiIiIhKKpXW0TM1NcU///yDwMBAXLt2DXK5HHXq1CnUmno5t36WLVtWobxs2bK4fv16gfueOHECn376KdLT02FpaYk///wTlSpVkrb//PPPiI+Px1dffZXn/q9fv0ZiYiJMTU3RunVrXL9+HQ4ODhg8eDCmTp2ab/KXlpaGtLQ06X1CQoJSx0pERERERFScVEr0zMzM8ObNG9SuXRu1a9fOc5uydHR0cr1/392krVq1QlxcHKKiorBu3Tp069YN//33H7y9vREYGIiFCxfi6tWr+SZsOROufPvtt9i+fTuaNGmCy5cvo0+fPsjIyMg3QVy4cCHmzZun9LERERERERFpgkq3biYlJeVZnpGRkWvSk/zY29sDyB5de1tkZKS0LT86OjowMjKCo6Mj5s+fD09PT6xfvx4AcOHCBcTGxqJq1aowMjKCkZGRtL6fkZERsrKyYGtrCz09PfTv3x9dunSBtbU1OnTogGHDhmHnzp359jt79mzEx8dLr5CQEKWOlYiIiIiIqDgVakRv+/btef4byH7u7fLly/Dw8FCqLRsbG3h4eODs2bPo1q0bgOxn686ePYt+/foVJiwIIaRRwLFjx2LEiBEK2+vXr49mzZph8eLF0NXVha6uLnx8fHKN+Onq6hY4mmhoaAhDQ8NCxUZERERERFTcCpXoTZs2Lc9/A4C+vj5cXV2xevVqpdubMmUKZsyYgU8//RQNGjTAjz/+iKioKIwePVqqM3bsWFy5cgXXr19HUlISxowZg5kzZ6JKlSpISEjAypUrcfPmTSxduhQApETubTKZDLq6ujAyMpLKpk+fjmHDhqFv375o1KgRrly5gi1btmDy5MmFOSVEREREREQlTqESvVevXgEAqlevnue6dYU1duxYxMbGol+/foiOjkbVqlVx6NAhhYlVMjIypAlQTE1N0bVrVwwbNgy3bt2CoaEhateujWPHjsHX17dQfXfv3h3R0dEYOHAgQkND4eDggEmTJuHLL78s8nERERERERFpkkrr6H0Icrk818QsAJCZmQm5XA4DAwOFciGEwmLtBUlPT4eOjg709PLOa/Pr+324jh4RERERERWXD76OHgCcOXMGXbt2ReXKlVG5cmX4+fnh7NmzqjaXb6Klp6eXK8kDoHSSBwAGBgb5JnkF9U1ERERERFQaqZThrF+/Hm3atIGxsTFGjx6N0aNHw8jICK1bt8bGjRvVHSMREREREREVgkq3bjo5OeH777/HZ599plC+bds2fP3113jx4oXaAizJeOsmEREREREVlw9+62Z8fDy6du2aq9zPzw/x8fGqNElERERERERqolKi5+3tjZMnT+YqP3nyJOrUqVPkoIiIiIiIiEh1hVpeIYevry8GDhyIIUOGoG7duhBC4Nq1a9iyZQtmzpyJP//8U6rbs2dPtQVLRERERERE76fSM3pWVlZK142Liyts86UGn9EjIiIiIqLiUpj8Q6URPW1O3oiIiIiIiEo7LiBHRERERESkZVRO9O7cuYPZs2ejT58+Utnu3buRkpKilsCIiIiIiIhINSoleqdOnYKPjw9u376N3bt3S+U3b97EypUr1RYcERERERERFZ5Kid6XX36J9evX4+DBgwrlAwYMwLp169QSGBEREREREalGpUQvODgY3bt3BwDIZDKp3NnZGS9evFBPZERERERERKQSlRI9c3NzhIeHA1BM9K5cuQIHBwf1REZEREREREQqUSnR6927NyZPnoyYmBgAgFwux9mzZzFy5Ej07dtXrQESERERERFR4aiU6C1cuBByuRx2dnaQy+UwNzeHr68vqlSpgrlz56o5RCIiIiIiIioMlRZMNzU1xT///IPAwEBcu3YNcrkcderUQd26ddUdHxERERERERWSSolejtq1a6N27drqioWIiIiIiIjUQKVbN588eYL58+fnKp8/fz6ePHlS5KCIiIiIiIhIdSolep9//jl8fHxylfv4+GDSpElFDoqIiIiIiIhUJxNCiMLuZGpqilevXsHc3FyhPDExERUqVEBCQoLaAizJEhISYGlpifj4eFhYWGg6HCIiIiIi0mKFyT9UGtGztLTE3bt3c5Xfvn0bpqamqjRJREREREREaqJSoterVy+MGDECAQEBUtn169cxYsQI9OrVS23BERERERERUeGplOh9//33KF++PLy9vWFmZgYzMzP4+PjA0dERCxcuVHeMREREREREVAgqr6N37NgxXL16FQEBAZDJZKhduzbX0SMiIiIiIioBirSOXt26dZncERERERERlTBcR4+IiIiIiEjLcB09IiIiIiIiLcN19IqA6+gREREREVFx4Tp6REREREREHzGuo0dERERERKRluI4eERERERGRluE6ekRERERERFqG6+gRERERERFpmSIlekIIREREIDMzU6Hc0dFR6TYyMzNx7tw5REREoEaNGqhWrdp790lISMDly5fx5s0beHl5wdPTM9+6ISEhOHXqFLy8vFCvXj2V6xAREREREZUWKiV6MTExmDBhAvbu3Yu0tLRc25VdsSEmJgZt2rRBTEwMqlevjrNnz2LYsGFYtmxZvvusWbMGP/74I7y8vKCnp4d///0Xn376KbZt2wY9PcXDycjIQI8ePXDz5k2MGTMmzyROmTpERERERESliUqJ3rRp0xAREYEzZ86gYcOGCAwMxJUrV/DNN99g+vTpSrfz5ZdfIiUlBbdu3YKZmRkuX76Mhg0bokOHDmjXrl2e+zg5OeHOnTswNjYGANy7dw9Vq1ZF79694efnp1B39uzZqFatGtLT0/ONQZk6REREREREpYlK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mUcq97F161Z4e3tj8uTJ0rNb9vb2mD59OgCgU6dOaNWqFQDA2NgYXl5eyMjIyLe97LgExo8fD1dXV7Ru3RoVK1bE5cuXAQB///03evfuDSEEHj58iEaNGiEpKQkAcOHCBbi5uaFVq1aoWLEixo8fL7V57tw5uLq6okmTJujbty9+//13AEDbtm2lpA8A/Pz88OTJE+zatQuPHz/GV199hb59++L+/fsKt0kePnwY5cqVg6+vL+zs7LB161apjR49emDMmDGoWrUqqlWrhqFDh773PBa1vadPn0rrNgLZ6y02atQIiYmJ7+2biIiIiKgwmOhpUHqmHFv9n+Wb5OUQAtjm/1zlUb179+7Bx8dHqbrBwcE4evQoBg8eXGC97du349mzZzh16hT279+PGTNmYOrUqQCAqVOnQk9PD9999x369OmDZcuWwdXVFVlZWRg8eDBWrFiBffv24eTJkzh9+jTOnz+PzMxM9OvXD4sWLZJG9HLay8+ECRNQrVo17Ny5E8HBwahdu7a0LSMjA0OHDsXvv/8uxTlx4kSFZxDLli2LFy9e4MWLF/jnn3/yfXZRXe35+PjA1NQU/v7+ALLXcmzatCkqVKhQ4HESERERERWW8ovGkdpdfRaT63bN/MQkp+Pasxg0crctdD+GhobSiFpB7t69ix49emD37t3vXTT+7NmzuHHjBvr37y+V2dnZAcie8XHLli1wcXFBz5490b17dwDZI1ohISGYNWuWtI+Ojg5iYmLw9OlTyOVyhfUSi+Lp06fQ19dHu3btAGQnWVWqVMHNmzfRsmVLAEDv3r0BAObm5vDw8EB4eDicnJw+aHuff/451q1bh0aNGmHdunX4/vvv1XK8RERERERvY6KnQXFKJnlS/ZTC1c/RtGlTzJs3D+np6TAwMJDKMzIypMW0b9y4gR49euC3335Dw4YN39umqakpBg8ejAULFuS5fceOHfDw8MC5c+cQGxuLMmXKwNTUFPr6+rh27Vqutd1evXqFN2/e5IoRAAwMDBSeiXv72b78lhEwNTXFmzdvkJmZCT297I95bGwszMzMpDo55TntyOX5j5iqq73evXvj66+/xqVLl5CUlIRGjRrl2ycRERERkap466YGWZnoF66+ceHq5+jTpw/s7OzQvn17/P3337h48SJ++eUX9OnTBwAQFBSENm3aYNasWTA3N0dwcPB7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",


[showing lines 2-3 of 3; full output saved to /home/heet18/.local/share/opencode/shell/cd2892a3e62df85862b6d60f4c893cf8ced97e93/sh_16ecfaff6001FgmOgkwqp1PUlD.out]

---

## Assistant

**Tool: shell**

**Input:**
```json
{
  "command": "grep -rn \"EVG\\|evg_cas\\|EVG-CAS\" /home/heet18/Downloads/Research\\ Papers 2>&1 | head -40"
}
```

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37969W2PHjtXJkyf16KOP2tznbvbcc89pxIgRBe5LT09XZGSk0tLS5OnpqStXrqh3796Ki4vT4cOHGdkrxJo1a+TqWvC/9pOTk9WvXz/98MMPkqSsrCzFxsbq5Zdf1ldffXUny3Qajz76qCZOnHjLfv/4xz8UHh6u5OTkki/KyS1YsOCWXyz88MMPViP7KFzHjh1v+s/vr7/+qkcffVRTpkyxfFG5Zs0anThxgqBXiBkzZuQbbbqmW7du6tatm1Vb48aN1bhxY0JeIZo0aaIVK1YUun/48OEymUxKTU1V2bJldfr0aTVq1Ehjx461+sICBeMZPRTorbfeUsuWLR1dhtN644031Lp169vug4JFRERo6NChlttg3Nzc1L9/fx05ckQpKSkOrs453XfffXrwwQct235+foqOjlZaWpoDq3J+K1eu1Jw5c/Tpp586uhTDOHTokLZt26bs7GxHl+L03nrrLbVv394S8iSpTZs2+W7thG0SExO1detWDRgwwNGlOK309HRFRUVZ7ngKDAxUzZo1lZ6e7uDKnAMjenep/fv353tWr0GDBvLz83NQRc7l4MGD+a4fz4nYV3Jycr5rXLt2bQUGBhbYf/PmzfLy8uK5sptYv369ypQpY9kuW7asmjRpYtUnISFB58+f15YtW7R06VJ98803d7pMp3H06NF8f6ORkZGWv8H09HQ9/fTT+vbbb+Xv7++IEp1OQkKCypcvb9kuU6ZMvi8dn3zySQUEBGj//v3q06ePJk2axLNPhUhPT8/3N3rtefKcnBytW7dOH330kbKzs7Vv3z5VqVKFf4feQkHP58bExMjFJf/YybRp01SlShV17tz5TpXndDIzM/P9jVauXFmRkZGSpFGjRql379769NNPVatWLW3cuNFy1xlujaB3l/rhhx+0du1aq7Z///vfql+/voMqci4LFixQQkKCVRvPidjXsmXLtHPnTqu2d999V23atMnXd9euXRo7dqxGjBiR7/ke/Ncbb7xhtV25cmXLrZrXjB07Vmlpafrzzz/Vs2dPNWvW7E6W6FTWr1+vo0ePWrW99NJL6tGjh8xms3r37q0+ffqoVatWDqrQ+YwdO9bqn2EvLy8tX75ckuTt7a2FCxfqoYcekiTt3LlTbdu2la+vryZMmOCQeku7bdu25bsF/trz5KdOndKVK1e0Y8cO1axZU5UqVVJ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",
/home/heet18/Downloads/Research Papers/deliverables/Phase3_TrustFin_Deliverable/Phase3_TrustFin_Deliverable/notebooks/Phase3_smoke_executed.ipynb:5926:      "image/png": "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uroqODhY48ePV6tWrfTrr79KkgoVKqTnn39e3377rW7cuGHXGrdv3y5Jqly5sl2PK0nOzs56++23dezYMX366aeZ9hk+fLhMJpOGDBkiSdq/f7+2bduW5UvYW7RooZiYmBytihkREaFLly5p9erVWrhwoZo0aWKeAf27f/4Z2foah+DgYLm7u+vbb7+1aL969ar5z9KWc3t4eOR4di04OFgeHh765ptvLNpv376tNWvWqHr16uZACgDIGkEPAJApFxcXtWjRQgsXLtTmzZuVkpKi9evXa/To0apTp46535EjR9SmTRv98MMPOn/+vG7evKlNmzbpt99+U6NGjcz9/v3vf+vy5ctq27attm3bpmvXruns2bP6/vvv1bJly/veHtmjRw8tWLBAx48fV2pqqk6fPq0JEybo448/Vo8ePTI8p2Uv/fv3V+/evfXmm2/qX//6l86cOaNbt27p4MGD6tatm5YsWWJ+v590dzYvf/78aty4cabHa9iwoTw9PXM0q/fCCy/Iz89PQ4cO1bFjx/Taa69Z7H/22WdVtmxZTZgwQYcPH9aVK1c0f/587dixI8OzkDnh5eWlt99+W0uXLtX06dN1+fJlHTlyRD179lS9evVsPve9Z+b27t173+fmvLy8NHbsWK1evVrvvfeeLly4oFOnTik8PFxnzpzRlClTrP5cAPA4IugBALI0b948tWzZUh07dlSpUqU0c+ZMRUZGqlChQuY+FStW1Ouvv65PP/1UgYGB8vb2Vt++ffX6669r4cKF5n5Vq1bV7t275efnp06dOqlIkSIKDg7W7NmzNXDgQPPzcFkZO3as9u7dq9atW8vLy0vVq1fXDz/8oNmzZ2vBggUP7BpI0ty5c7VixQpt375dNWvWlJubm6pWrao1a9bo999/1xtvvCHp7uI1kZGRCgkJsbhGf+fq6qrGjRtryZIlunbtWrbndXJyUq9evRQbG6uiRYuqQ4cOFvsLFCig7777TkWLFlXt2rVVqVIlRUdH5+rF7KNHj9bUqVM1a9YslSpVSl26dNGbb76pp59+2uZzDx48WM2bN9dzzz0nZ2dni/foZeatt97S559/rtWrV6ts2bKqVq2aEhIS9PPPP1ssRAMAyJqTiSWpAACw2rRp0zRs2DC99dZb5pemAwDwqHBxdAEAAORFQ4cO1YULFzR58mQVLlxYo0ePdnRJ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9z6o2gAAAAAANrAq6Pn5+em9997TqVOntGTJEp09e1ZBQUGSpE8++URxcXE2FbF7926tXbtWsbGxOeqfnJyszZs36/fff1dycrJN5wQAAAAAo7Jp1U1nZ2e1aNFCK1euVHx8vKZMmaJZs2bJ19dXtWvXzvFxkpOT9fzzz6tZs2YaP368nnnmGb377rtZ9jeZTBo8eLAqVaqk0aNHa8CAASpTpowiIyNt+RgAAAAAYEi5fo/ek08+qWHDhunPP//Uli1bFBAQkOOxY8aMUXx8vA4fPqzNmzfrv//9r8aNG6eNGzdm2t9kMqls2bI6ceKEfv31V+3cuVPvvvuuXn311RzPBgIAAACA0dnlhen3hISEaOHChTnqazKZFBkZqddee01FihSRJDVu3Fi1atXS4sWLMx3j7OysQYMGqWDBgua2V155Rbdu3dK+ffty/wEAAAAAwACseo/evefx7mfnzp337RMXF6fLly+revXqFu01atTQnj17clzTpk2bJEn+/v5Z9klNTVVqaqp5OykpKcfHBwAAAIC8xqqgt2vXLpUtW1atWrVS/vz5c3XixMRESTLP5t3j7e2tq1ev5ugYsbGxGjhwoCIiIlS+fPks+02cOFHjxo2zuVYAAAAAyEusCnoffvih5s+frxUrVqh79+567bXXVLlyZZtO7OrqKkm6fv26RXtKSorc3NzuOz4hIUFNmzbVs88+q1mzZmXbd9SoURoyZIh5OykpSb6+vjZUDQAAAACPPque0Rs+fLgOHz6s5cuX68KFCwoKClL9+vU1f/58paSkWHXiMmXKKF++fBkWUYmNjVW5cuWyHXvhwgU1btxYpUuX1qpVq+4bDN3c3OTl5WXxBQAAAABGZdNiLM8995y++OILnT17Vt27d9dHH32kEiVKWHWMAgUKqFGjRlq5cqW57cqVK/r555/VokULc1tMTIx++eUX8/bFixfVuHFjlSpVSmvWrLFYmAUAAAAAYOWtm39nMpkUExOj6OhonTx5MsOiKjkxceJENWzYUL1791a9evU0Z84cVahQQa+++qq5z6xZsxQdHa39+/crNTVVL7zwgi5evKjRo0dr3bp15n6BgYEqW7asrR8HAAAAAAzD6qAXHx+vzz//XAsXLlRSUpLCw8O1a9eubFe9zEpQUJB27typOXPmaN26dWrTpo0GDhyoAgUKmPsEBgbK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Y/GLB0lavHixypUrp2PHjqlixYoKCgqyWIlv2rRpCg4OftglP/JeeOEF7d27N8MvHurWraupU6eat7/++mu98cYbcnd3l7u7u0qUKKGlS5fKx8fnYZf8SIuJiVFgYKBq1KghDw8Pi30zZsxQUFCQVq9erSlTpljsO3nypBISEnTu3Dk9+eSTD7PkR163bt303//+VwEBARbtFStW1MKFCyVJp06d0osvvqjTp0/Lz89PR44cUZ06dbR8+XIVKVLEEWXnKczoPabefvtt9ezZM8v9f/75pz766CMdPnxYERERD6+wPGLw4MHq379/lvsPHDig8ePH68KFC+rUqZPNfR5nffv21ciRIzPdd/HiRVWoUEHx8fEqWLCgbt26pfDwcLVv316nTp1iZi8LGzdulItL5v/ZP336tF577TUtX75ckpSUlKSQkBANGzZMkZGRD7PMPKNTp06aMWPGffv961//Urly5XT69OkHX1Qet3r16vv+YmH58uUWM/vIWtOmTbP9+/vrr7+qU6dO+uyzz8y/qNy4caMSEhIIellYuHBhhtmme9q1a6d27dpZtNWqVUu1atUi5GUhKChIP/30U5b7hw4dKicnJ8XFxalAgQL666+/VKNGDY0fP97iFxbIHM/oIVPvvvuu6tWr5+gy8qzRo0frueeey3UfZK58+fIaOHCg+TaY/PnzKyIiQmfOnFFsbKyDq8ubGjRooObNm5u3vby8VLNmTcXHxzuwqrzv559/VlRUlD755BNHl2IYJ0+e1O7du5WcnOzoUvK8d999Vy+88II55ElSaGhohls7YZuYmBj98ccf6t27t6NLybMuXryowMBA8x1P3t7eeuaZZ3Tx4kUHV5Y3MKP3mDp69GiGZ/WeffZZeXl5OaiivOXEiRMZrh/PidjX6dOnM1zjypUry9vbO9P+O3bskLu7O8+VZWPr1q3Kly+febtAgQIKCgqy6LN9+3Zdu3ZNu3bt0vfff68lS5Y87DLzjLNnz2b4Ga1QoYL5Z/DixYvq0aOHli5dqsKFCzuixDxn+/btKlq0qHk7X758GX7p2K1bNz3xxBM6evSoevbsqZkzZ/LsUxYuXryY4Wf03vPkqamp2rJli6ZPn67k5GQdOXJEpUqV4r+h95HZ87n169eXs3PGuZP58+erVKlSatmy5cMqL89JTEzM8DNasmRJVahQQZI0ZswYhYeH65NPPpG/v79+//13811nuD+C3mNq+fLl2rRpk0Xbf/7zH1WrVs1BFeUtq1ev1vbt2y3aeE7Evn744Qft27fPou2DDz5QaGhohr779+/X+PHjNXLkyAzP9+D/jB492mK7ZMmS5ls17xk/frzi4+N16NAhvfjii6pTp87DLDFP2bp1q86ePWvRNmTIEHXs2FEmk0nh4eHq2bOnGjZs6KAK857x48db/B12d3fX+vXrJUkeHh5as2aN2rRpI0nat2+fGjVqpEKFCmnatGkOqfdRt3v37gy3wN97nvzSpUu6deuW9u7dq2eeeUYlSpTQsWPHVK9ePS1ZsiTLX6o97mbOnJnhGb0ff/wxwy8bUlNT9eWXX+qNN96w+AUbLB09ejTDz2j79u01bNgwSXefKW3ZsqXeeecdlStXTsePH1ffvn3l7+/viHLzHILeY+p+z+ghe/d7Rg+5l90zen93/PhxNW/eXG3atMkQZGApu2f07lm9erUk6cqVK2rZsqW6dOmiH3744WGUl+dk94zewoULtWfPHg0fPtz82+q0tDSdOXNG0dHRqlu37kOsNO/I7hm9kiVLmkOedHelw/79++uzzz4j6GUhu2f07gXqH3/8UTExMSpWrJj++usv1atXT8OGDTMvhgFL2T2j93dff/21rl69qtdee+3BF5WH3e8Zvc6dO+v69euKjY2Vh4eHLl26pJCQEKWkpGjWrFkPsdK8iaAHIM86ceKEGjVqpAYNGmjx4sWZ3joD2xQpUkQRERHq06ePbt++fd+AiIzKly9vsQJnYmKi1q1bp9jYWMKznRQvXlznzp1Teno6f/+t5OPjIw8PD3Xs2NG8Cre3t7c6deqkZcuWObi6vG/+/PkKCwtj4aBcuHXrltavX6958+aZZ1F9fHzUpUsXLVy4kKCXA/yfG0CedOrUKTVq1Ej16tVTZGQkt8bk0rVr1zLcjnTs2DE98cQThDwb/P11K/eULl1avXv35vUKNsrsZ3T9+vWqUqUKIc8Gzs7OCgsLy7DgUlxcHCtE5tLJkyf1yy+/6KuvvnJ0KXla/vz5VbhwYcXFxVm0x8bG8jOaQ/zf+zGV2WIspUqVMr8o9d5LLI8ePSrp/16yzoItd2W2GEvx4sVVsWJFSXdvJzx37pwOHTokSfrtt9/k6upqsWBLTvo8zjJbjMXHx0f+/v5KSEhQo0aN5OXlpX79+ik6Otrcp3r16ipUqNDDLjdP+OdiLNL/Xa+hQ4fKzc1Nzz//vAoWLKhff/1V06ZN0/Tp0x1U7aMvs8VYChcuzLPOufDPxVgkqWrVqipSpIgmTpyoc+fOqVmzZipYsKBWrFih7777Tt98842Dqn30ZbYYi7u7u2rWrClJeu+99xQSEqKxY8cqJCRE27Zt05IlSwgo2chsMZaKFSuqePHi5u0FCxbIx8cnw6sWkFFmi7G4uLiYb28fOHCgJk+eLE9PTwUEBOi3337TF198oXnz5jmi3DyHF6Y/hho0aJBp+4svvqjBgwdLkt5//32tW7cuQx8WbLn7zMP169cztLds2VJvv/22JGnq1KlatWpVhj5/X7AlJ30eV+3atdNff/2Vob1Ro0Z6//33tXv3bg0YMCDTsXPnzlWVKlUedIl5yvr16/Xee+9luu/e9bpz544+//xzrV27VikpKSpXrpx69uyp2rVrP+Rq84ZevXqZfxH2d88++6w+/fTTTMd07NhR7du3V3h4+IMuL8/Ztm2bhg4dmum+yZMnq379+jKZTFq2bJnWrFmjxMREVapUSf369TP/gg2WBg8erB07dmRoL1u2rL788kvz9v79+/XRRx/p9OnTKlOmjCIiIlS/fv2HWWqecOzYsSzXNhg5cqRat25t3n7llVcUFBSU5c807srq35qFChUy395uMpm0YsUKrVq1ShcuXNBTTz2lrl27Kiws7GGXmycR9AAAAADAYLipHQAAAAAMhqAHAAAAAAZD0AMAAAAAgyHoAQAAAIDBEPQAAAAAwGAIegAAAABgMAQ9AAAAADAYgh4AAA/At99+qxMnTji6DADAY4oXpgMA8px169bpypUrGdr9/PxUt25dB1SUkZ+fn8aMGaOIiAhHlyJJWrNmjapVq6Zy5co5uhQAwEPg4ugCAACw1ogRI3Tjxg0FBgZatDds2PCRCXpt2rRRhQoVHF2GWb9+/fTBBx8Q9ADgMUHQAwDkSS1atNCMGTMy3Xfs2DHt2rVLHTp0kKurqyTpxo0bWrNmjerUqaNy5crpm2++Ua1ateTi4qI9e/bIy8tL9evXl5OTk8Wx0tLSFB0drcTERFWpUiVDeLt3HCcnJ/3xxx8qU6aMAgMD1bRpU5UtWzZDv/z582vPnj3y8PBQSEiInJ2ddfr0acXExMjX11c1a9bM8HlyWoObm5tiYmLk6empunXrKl++fJLuzoDeuHFD27ZtU4ECBZQvXz516tTJ6msOAMg7CHoAAMMpVqyYRo4cqe3bt2vatGmSpKFDh+qnn37S7t27JUm9e/dWzZo1dfDgQT377LPasWOHqlevru+++05ubm6SpJiYGLVv315FixbVU089pejoaL344ouaPXu2+Vy9e/dWrVq1dPjwYQUGBqpDhw4KDAzUgAEDNGbMGJUvX97cr3LlyoqPj1fVqlW1detW1apVS6GhoVq4cKH8/f21efNm9erVyyLA5rSGoKAgHT16VFWqVNGuXbv09NNPa8OGDcqXL582bNigGzdu6I8//tCVK1fk6upK0AMAgyPoAQDypCNHjigqKsqirVatWqpYsaK8vLwUGRmp0NBQNWvWTKmpqZo3b562bt0qDw8Pi2PExMTIx8dH58+fV2BgoD799FMNHjxYt27dUocOHTRo0CANGjRIknTx4kVVr15doaGh6tKli/k4Z8+e1b59+1SoUKFsa759+7YOHjyoAgUKaOfOnapdu7bS0tJ08OBBubq6auPGjWrSpIlGjhypEiVKWFXDlStXtG/fPrm7u+vChQsqX7681qxZow4dOmjSpEmKjIzU66+/rp49e+buwgMA8gSCHgAgTzp+/LhWrVpl0Va0aFFVrFhRkhQSEqK3335bPXr00O3bt/Xuu++qdu3aFv179eolHx8fSVKJEiUUHh6ur776SoMHD9aGDRt0+vRpFStWTCtWrJDJZJLJZFKFChW0YcMGi5DVq1ev+4Y8SQoPD1eBAgUk3Q2lrq6u6tmzp/n20nr16ik9PV3Hjh1TiRIlrKqhR48ecnd3l3R3RrNy5co6fPiwlVcVAGAUBD0AQJ6U3TN694wYMULTp09XoUKFNGLEiAz7/fz8LLbLlSunxYsXS5JOnTolV1dXrVmzxqLPU089pUqVKlm0lSxZMkc1FylSxPy9k5OT8ufPb9F275bRmzdvWl1D0aJFLbbd3NzMxwEAPH4IegAAwxo7dqyKFCmiK1eu6LPPPlPfvn0t9v/zFQ1Xrlwxz/B5eXkpLS1N8+bNk6enZ7bn+ecCLvZiTQ0AAPwdL0wHABjSL7/8ohkzZmjJkiX6+OOPNXTo0Ay3Mv791k+TyaRvvvlGISEhkqTQ0FC5urpq7ty5FmPu3LmjhISEB16/vWvw9PRkhg8AHiPM6AEA8qTMFmN58skn1aRJE12+fFnh4eEaMWKEGjRooAYNGuiHH35Q165d9fvvvyt//vySpAMHDuill15S06ZN9d///lfHjh3TihUrJN19Zm/GjBkaMGCADh06pODgYMXFxWnlypWaMGGCWrdu/cA/oz1rCAoK0oIFC+Tu7q6CBQuy6iYAGBwzegCAPKd58+by8vLSqlWrLL42b94sSfr+++/VunVrjR071jxmzpw5qly5sjZt2mRumzFjhpo2bapdu3apUqVK2rFjh8qUKWPe37dvX23fvl1FixbVr7/+Kklavny5RcDq2LGjxfvy7vnnC9Mz6/fSSy+pdOnSFm2dO3dWiRIlcl3DCy+8oICAAPP2xx9/rLZt2+rnn3/Wt99+m6FeAICxOJlMJpOjiwAA4GHz8fHRrFmzLFauBADAKJjRAwAAAACDIegBAB5LWd1yCQCAEXDrJgAAAAAYDDN6AAAAAGAwBD0AAAAAMBiCHgAAAAAYDEEPAAAAAAyGoAcAAAAABkPQAwAAAACDIegBAAAAgMEQ9AAAAADAYAh6AAAAAGAw/w81se0d/OHKVwAAAABJRU5ErkJggg==",
/home/heet18/Downloads/Research Papers/trust_fin_extra_phase/logs/p4_full_EVG.log:3:[NbConvertApp] Writing 150312 bytes to notebooks/Phase4_full_EVG.ipynb
/home/heet18/Downloads/Research Papers/trust_fin_extra_phase/docs/eval_rigor.md:1:# Evaluation Rigor Annex — Phase-4 EVG-CAS
/home/heet18/Downloads/Research Papers/trust_fin_extra_phase/docs/eval_rigor.md:9:## 1. C1 (accuracy): greedy vs EVG are statistically indistinguishable
/home/heet18/Downloads/Research Papers/trust_fin_extra_phase/docs/eval_rigor.md:16:| exec-verified (EVG) execution accuracy | 0.6556 | — |
/home/heet18/Downloads/Research Papers/trust_fin_extra_phase/docs/eval_rigor.md:17:| delta (EVG − greedy) | −0.0017 | bootstrap [−0.0122, +0.0087] |
/home/heet18/Downloads/Research Papers/trust_fin_extra_phase/docs/eval_rigor.md:21:Confusion (paired): both correct 734, EVG-only 18, greedy-only 20, both wrong 375.
/home/heet18/Downloads/Research Papers/trust_fin_extra_phase/docs/eval_rigor.md:23:Facts established: (i) the 0.0017 gap is two orders of magnitude below any practical difference; (ii) the 95% CI alone covers ±0.01; (iii) Weighted McNemar does not reject at any conventional level. **Conclusion: C1 stays REJECTED for "EVG improves accuracy," and is now stated with a proper test** — the earlier tie claim is not an artifact of eyeballing rounded means.
/home/heet18/Downloads/Research Papers/trust_fin_extra_phase/docs/eval_rigor.md:27:## 2. Calibration signal: agreement bins (protected test, EVG)
/home/heet18/Downloads/Research Papers/trust_fin_extra_phase/docs/eval_rigor.md:49:Main finding: on 31% of records the greedy model computes the correct value yet **prints an answer string that does not match the executed value** (formatting drift — e.g., scientific notation, rounding, or a different value echoed). EVG's `exec_verified` rule eliminates this mismatch by construction. **But** this is exactly the failure that a *printed-answer* consumer would still see — the calibration head scores the generated-program channel, not the user-facing string. This is the primary surface-level improvement available (see roadmap R1/R2).
/home/heet18/Downloads/Research Papers/trust_fin_extra_phase/docs/eval_rigor.md:69:## 5. Difficulty stratification (FinQA protected test, EVG)
/home/heet18/Downloads/Research Papers/trust_fin_extra_phase/docs/eval_rigor.md:86:| **Ours (EVG, FROZEN)** | FinQA train (SFT, in-domain DSL) | **official dev n=883** | **0.6750** |
/home/heet18/Downloads/Research Papers/trust_fin_extra_phase/docs/eval_rigor.md:87:| **Ours (EVG, FROZEN)** | "" | **protected test carve n=1,147** | **0.6556** |
/home/heet18/Downloads/Research Papers/trust_fin_extra_phase/docs/exp_4(b).md:24:EVG (`P4-E8b`), n = 8:
/home/heet18/Downloads/Research Papers/trust_fin_extra_phase/docs/exp_4(b).md:40:| EVG-only correct | **18** |
/home/heet18/Downloads/Research Papers/trust_fin_extra_phase/docs/exp_4(b).md:52:| EVG agreement bin | n records | `exec_verified` accuracy |
/home/heet18/Downloads/Research Papers/trust_fin_extra_phase/docs/exp_4(b).md:68:| `median_of_n` (EVG) | 0.13058 | 0.7715 | 0.8832 | 0.5556 |
/home/heet18/Downloads/Research Papers/trust_fin_extra_phase/docs/exp_4(b).md:92:risk@cov0.80 = 0.3088; the EVG-feature calibrator gives 0.2139 (validation) → 0.2375 (test).
/home/heet18/Downloads/Research Papers/trust_fin_extra_phase/docs/exp_4(b).md:124:| **FinQA EVG n = 8** | 1147 × 8 = 9176 candidates | ≈ 90 min | **4.71** |
/home/heet18/Downloads/Research Papers/trust_fin_extra_phase/docs/exp_4(b).md:125:| FiQA EVG n = 8 | 234 × 8 | ≈ 7 min | 1.79 |
/home/heet18/Downloads/Research Papers/trust_fin_extra_phase/docs/exp_4(b).md:127:EVG costs ≈ **3.7×** greedy per FinQA record (4.71 vs 1.28 s on validation), not 8×,
/home/heet18/Downloads/Research Papers/trust_fin_extra_phase/docs/exp_4(b).md:136:| **C3** — EVG features improve the calibrator | e-AURC ↓ where non-monotone features exist | **Supported (FinQA), rejected (FiQA).** FinQA e-AURC 0.2029 → 0.1232 (−39.3 %) with agreement+executability; FiQA all calibrators hurt ranking. |
/home/heet18/Downloads/Research Papers/trust_fin_extra_phase/docs/exp_4(b).md:152:   records and a regression target are too little for the EVG features to pay off.
/home/heet18/Downloads/Research Papers/trust_fin_extra_phase/docs/exp_4(b).md:156:   Deploying EVG is justified only where a 31 % reduction in answered-error is worth 3.7×.
/home/heet18/Downloads/Research Papers/trust_fin_extra_phase/docs/exp_4(b).md:176:- **Error taxonomy:** the only channel EVG leaves open is the *printed answer string*
/home/heet18/Downloads/Research Papers/trust_fin_extra_phase/docs/novelty_check.md:1:# Novelty check — EVG-CAS vs the 2024–2026 selective-abstention literature
/home/heet18/Downloads/Research Papers/trust_fin_extra_phase/docs/exp_4(c).md:95:| exec correct & print wrong (**format mismatch**) | 355 | 31.0% of records — EVG's *only* unfixed channel |
/home/heet18/Downloads/Research Papers/trust_fin_extra_phase/docs/exp_4(a).md:1:# Experiment 4(a) — Proposed method on validation (EVG-CAS), pre-protected
/home/heet18/Downloads/Research Papers/trust_fin_extra_phase/docs/exp_4(a).md:12:EVG-CAS = **Execution-Verified Generation + Calibrated Abstention**. At inference we sample
/home/heet18/Downloads/Research Papers/trust_fin_extra_phase/docs/exp_4(a).md:58:- **C3 is tested on ranking**, because `platt_conf_full` adds EVG agreement and executable
/home/heet18/Downloads/Research Papers/trust_fin_extra_phase/docs/exp_4(a).md:130:| EVG-only correct | **8** |
/home/heet18/Downloads/Research Papers/trust_fin_extra_phase/docs/exp_4(a).md:134:EVG flips 8 records in and 8 out — net zero. Sampling at `temperature 0.7` is not a free


[showing lines 8-40 of 40; full output saved to /home/heet18/.local/share/opencode/shell/cd2892a3e62df85862b6d60f4c893cf8ced97e93/sh_16ecfde11001U37PhqAUSmHYKR.out]

---

## Assistant

Short answer: **No. EVG-CAS is *not* used in the main Phase-4A/B/C notebooks you just ran. It only exists in the separate `trust_fin_extra_phase/` track.**

## Where EVG-CAS *is* used

EVG-CAS (Execution-Verified Generation + Calibrated Abstention) shows up in `trust_fin_extra_phase/`:

- `trust_fin_extra_phase/notebooks/Phase4_full_EVG.ipynb`, `Phase4_smoke_EVG.ipynb`
- `trust_fin_extra_phase/docs/exp_4(a).md`, `exp_4(b).md`, `exp_4(c).md`, `eval_rigor.md`, `novelty_check.md`
- `trust_fin_extra_phase/logs/p4_full_EVG.log`

In that track, the idea is to *sample multiple generations* of the program, execute each one, look for **agreement across samples** (`exec_verified`, agreement bins), and then do **C**alibrated **A**bstention **S**election on top of that.

## Where EVG-CAS *is not* used

The exact files you just executed have **no** `EVG` or `CAS` references:

- `notebooks/Phase4A_TrustFin_Execution_Verification_and_Financial_Grounding.ipynb`
- `notebooks/Phase4B_TrustFin_Calibration_and_Selective_Prediction.ipynb`
- `notebooks/Phase4C_TrustFin_Ablation_Statistical_Evaluation.ipynb`

Instead, the main track uses different names like:

- `official_executable_feature`, `program_structural_valid` (B3 – "verifier-aware")
- `any_literal_unmatched`, `max_question_local_overlap` etc. (B4 – "grounding-aware")
- Just a post-hoc risk model (logistic/ridge) on top of a *single* generated program from E5

## Key Difference

- **Main Phase-4A/B/C:** *Post-hoc*. Take the **single** E5 program, check if it's executable/structurally valid/grounded in numbers, and feed those as features to a calibrator to learn "when to abstain".
- **`trust_fin_extra_phase/EVG-CAS`:** *Generation-time*. Generate **multiple** candidates (`n>=2`), execute all of them, require them to agree on the executed value (`exec_verified`), use that consensus + other signals for calibration.

**So for the Phase-4B and Phase-4C you just ran, EVG-CAS is not used at all. The shared idea is "execution verification", but the implementation and naming are different between the two tracks.**

---

## User

then what architecture does the phase 4 ABC Is using to make the proepr calibaration of the mode

---

## User

then what architecture does the phase 4 ABC Is using to make the proepr calibaration of the model 

---

## Assistant

**Tool: shell**

**Input:**
```json
{
  "command": "grep -rn \"B[1-8]\" /home/heet18/Downloads/Research\\ Papers/notebooks/Phase4B_TrustFin_Calibration_and_Selective_Prediction.ipynb | head -20"
}
```

grep: /home/heet18/Downloads/Research Papers/notebooks/Phase4B_TrustFin_Calibration_and_Selective_Prediction.ipynb: No such file or directory


---

## Assistant

**Tool: shell**

**Input:**
```json
{
  "command": "grep -rn \"source_likelihood\\|logistic\\|Platt\" /home/heet18/Downloads/Research\\ Papers/notebooks/Phase4B* 2>&1 | head -15"
}
```

17:    "| B2 | Likelihood-only Platt/logistic model | Calibration improvement versus B1 |\n",
18:    "| B3 | Verifier-aware logistic model | Likelihood + prompt length/truncation + syntactic/executable validity |\n",
19:    "| B4 | Grounding-aware logistic model | B3 + recomputed numerical grounding features |\n",
261:    "   'B1_raw':['source_likelihood'],\n",
262:    "   'B2_likelihood':['source_likelihood'],\n",
263:    "   'B3_verifier':['source_likelihood','prompt_tokens','prompt_truncation_observed','official_executable_feature','program_structural_valid'],\n",
264:    "   'B4_grounding':['source_likelihood','prompt_tokens','prompt_truncation_observed','official_executable_feature','program_structural_valid',\n",
267:    "FIQA_FEATURES=['source_likelihood','generated_tokens','prompt_tokens','sentiment_pred_clipped','score_near_boundary','parse_ok']\n",
321:    "def logistic(features):\n",
500:    "        prob=np.clip(fit_X['source_likelihood'].to_numpy(dtype=float),1e-6,1-1e-6)\n",
503:    "        mdl=logistic(feat)\n",
824:    "    if name=='B1_raw': hp=np.clip(hold_X.source_likelihood.to_numpy(dtype=float),1e-6,1-1e-6)\n",
982:    "q_likelihood=qH['source_likelihood'].to_numpy(dtype=float)\n",
987:    "          'B6_likelihood':qX['source_likelihood'].to_numpy(dtype=float),\n",


---

## Assistant

The main Phase-4A/B/C track uses a **post-hoc, feature-based calibrator** on top of a *single* execution-verified program from the frozen E5 model. It's not EVG-CAS (no multi-sample generation/consensus). Here's the exact architecture.

## Overview
The system is: frozen SFT model (E5, Qwen3-1.7B) → generate 1 program → execute it deterministically → run a small risk model on features to decide "answer or abstain" (selective prediction). Calibration is done by fitting that risk model with strict fit-only selection.

## Phase-4A (preparation) 
- Executes generated programs (so `execution_correct` exists). 
- Builds the `phase4b_interface/` with features like `source_likelihood`, `prompt_tokens`, `prompt_truncation_observed`, `official_executable_feature`, `program_structural_valid`, and recomputed numerical grounding flags (`any_literal_unmatched`, `literal_presence_fraction`, `max_question_local_overlap`, etc.). 
- Creates fixed calibration splits: FinQA (698 fit / 185 holdout), FiQA (92 fit / 25 holdout).

## Phase-4B (build the calibrator)
Four classifiers for FinQA predicting $p(\text{correct}\mid x)$:

| Model | Features | Notes |
|---|---|---|
| **B1 (raw)** | `source_likelihood` | Uncalibrated token-likelihood from the model (baseline). |
| **B2 (likelihood-only)** | `source_likelihood` | Platt/logistic calibration on top of likelihood. |
| **B3 (verifier-aware)** | `source_likelihood` + `prompt_tokens`, `prompt_truncation_observed`, `official_executable_feature`, `program_structural_valid` | Adds "verifier" signals (does it execute? is structure valid?). |
| **B4 (grounding-aware)** | B3 + `any_literal_unmatched`, `unmatched_literal_count`, `literal_presence_fraction`, `max_question_local_overlap` | Adds numeric/lexical grounding checks against the question/context. |

Implementation details:
- **Model**: `LogisticRegression(C=1.0, max_iter=1200, random_state=42)` in a `Pipeline` with `SimpleImputer(median)` + `StandardScaler()`. 
- **Training**: fit on calibration-fit with **StratifiedKFold(5)**; predictions used for selection are **out-of-fold (OOF)** to avoid overfitting. 
- **Selection**: winner chosen by **minimum fit-OOF AURC** (tie-break by Brier). Thresholds for target coverage (e.g. 80%) are computed from those fit-OOF scores and **frozen before looking at holdout**. 
- **FiQA**: handled separately as **B6** with a `Ridge(alpha=10.0)` regressor predicting `absolute_error` from `source_likelihood`, `generated_tokens`, `prompt_tokens`, `sentiment_pred_clipped`, `score_near_boundary`, `parse_ok`; used to drive selective MAE/intervals (descriptive, not formal conformal).

The saved artifacts are `models/B2_likelihood.joblib`, `B3_verifier.joblib`, `B4_grounding.joblib`, `B6_fiqa_error_ridge.joblib`, plus holdout scores and manifests (`phase4b_freeze.json`, `B8_fit_only_selection.json`).

## Phase-4C (evaluate & freeze)
Applies the **frozen** policy to the same holdout:
- Gates: require `official_executable_feature == true` and `program_structural_valid == true` (structural validity). 
- Selective prediction at frozen thresholds (coverage-matched comparisons). 
- Metrics: selective risk/accuracy, McNemar (written vs executed), paired bootstrap CIs, ablations (C7), FiQA selective MAE. 
- **Result**: B1_raw (likelihood-only) won on fit-OOF, so the "full" policy ended up effectively likelihood-based abstention; verifier/grounding didn't improve fit-OOF enough to be selected. Everything is frozen to `phase4c_methodology_freeze.json` with `PASS_METHODOLOGY_FROZEN`.

## Key points
- **Post-hoc calibration**: single-sample, no multi-generation sampling/consensus (that's the EVG-CAS track). 
- **Strict separation**: labels/correctness never used as features; selection is fit-only. 
- **Deterministic execution first**: the big gain comes from running the program, the calibrator just decides when to trust that executed result.

That's the architecture for 4A/B/C.
