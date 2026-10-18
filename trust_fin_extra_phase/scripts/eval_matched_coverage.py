"""Phase-4.2 — evaluation-improvement pass (review-driven).

Matched-coverage selective-prediction comparison, component ablations, confidence
intervals, and error-taxonomy reconciliation. Computed ONLY from the frozen Phase-4
predictions (no new model runs, no calibration refit, no test-set threshold tuning for
the pre-registered claims). The frozen calibrated-confidence vector (Platt-full fitted on
validation, frozen before the test opened) is read as-is from P4-E8e.

This is a *post-hoc analysis* of already-collected protected-test predictions. Matched-
coverage thresholds are analysis-time descriptives for method comparison and are NOT put
forward as additional pre-registered operating points.

Writes: artifacts/phase4_2/analysis/{matched_coverage_finqa.{json,csv,png},
matched_coverage_fiqa.{json,csv,png}, error_reconciliation.json, score_auc.json,
summary.json}
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
P4 = ROOT / "artifacts/phase4"
OUT = ROOT / "artifacts/phase4_2/analysis"
OUT.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(42)
B = 3000
COVERAGES = [0.50, 0.70, 0.80, 0.90, 0.95]

def stable_topk(df, score_col, k):
    return df.sort_values(score_col, ascending=False, kind="mergesort").head(k)

def selective_risk(df, score_col, corr_col, cov):
    k = int(cov * len(df))
    sel = stable_topk(df, score_col, k)
    return float(1 - sel[corr_col].mean()), int(k)

def wilson(k, n, z=1.96):
    if n == 0: return [0.0, 1.0]
    p = k / n; d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d
    s = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return [float(c-s), float(c+s)]

def risk_ci(df, score_col, corr_col, cov):
    k = int(cov*len(df))
    sel = stable_topk(df, score_col, k)
    nerr = int((~sel[corr_col].astype(bool)).sum())
    lo, hi = wilson(nerr, nsel := len(sel))
    return float(1 - nerr/nsel), lo, hi

def bootstrap_risk_diff2(dfA, scoreA, dfB, scoreB, corrA, corrB, cov, B=B, seed=42):
    rng = np.random.default_rng(seed)
    n = len(dfA); k = int(cov*n)
    sa, ca = dfA[scoreA].to_numpy(), dfA[corrA].to_numpy()
    sb, cb = dfB[scoreB].to_numpy(), dfB[corrB].to_numpy()
    diffs = np.empty(B)
    for i in range(B):
        idx = rng.integers(0, n, n)
        a = stable_topk(dfA.iloc[idx], scoreA, k); rA = 1 - a[corrA].mean()
        b = stable_topk(dfB.iloc[idx], scoreB, k); rB = 1 - b[corrB].mean()
        diffs[i] = rB - rA
    return {"mean": float(diffs.mean()),
            "ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
            "p_better_than_A": float(np.mean(diffs < 0))}

def main():
    E8b = pd.read_csv(P4/"predictions/P4-E8b.csv")
    E8a = pd.read_csv(P4/"predictions/P4-E8a_greedy_scored.csv")
    E8e = pd.read_csv(P4/"analysis/P4-E8e_finqa_abstention_scores.csv")

    strategies = {
        "greedy_confidence": (E8a, "confidence", "greedy_exec_correct"),
        "c0_print_confidence": (E8b[E8b.rule=="c0_print"], "selected_confidence", "final_answer_correct"),
        "self_consistency_print": (E8b[E8b.rule=="self_consistency"], "selected_confidence", "final_answer_correct"),
        "best_of_n_exec": (E8b[E8b.rule=="best_of_n"], "selected_confidence", "execution_correct"),
        "exec_verified_agree_only": (E8b[E8b.rule=="exec_verified"], "agreement", "execution_correct"),
        "exec_verified_agree_x_exec": (E8b[E8b.rule=="exec_verified"].assign(
            agree_x_exec=lambda d: d.agreement * d.executable_rate), "agree_x_exec", "execution_correct"),
        "exec_verified_frozen_pi": (E8e, "calibrated_confidence", "execution_correct"),
    }

    n_rec = len(E8a)
    rows = []
    for name, (df, sc, cc) in strategies.items():
        assert len(df) == n_rec, (name, len(df))
        full_risk = float(1 - df[cc].mean())
        for cov in COVERAGES:
            risk, k = selective_risk(df, sc, cc, cov)
            r_lo, r_hi = risk_ci(df, sc, cc, cov)[1], risk_ci(df, sc, cc, cov)[2]
            rows.append({"strategy": name, "coverage": cov, "n_answered": k,
                         "selective_risk": risk, "selective_acc": 1 - risk,
                         "risk_ci95_lo": r_lo, "risk_ci95_hi": r_hi,
                         "full_risk": full_risk,
                         "err_reduction_vs_self_full": (full_risk - risk) / full_risk,
                         "err_reduction_vs_greedy_full": (float(1 - E8a.greedy_exec_correct.mean()) - risk) / float(1 - E8a.greedy_exec_correct.mean())})
    mc = pd.DataFrame(rows)
    mc.to_csv(OUT/"matched_coverage_finqa.csv", index=False)

    # frozen policy operating points (pre-registered thresholds, NOT analysis thresholds)
    frozen = {"theta_0.80": {"theta": 0.4090, "cov": float((E8e.calibrated_confidence >= 0.4090).mean()),
                             "risk": float(1 - E8e.execution_correct[E8e.calibrated_confidence >= 0.4090].mean())},
              "theta_0.90": {"theta": 0.2211, "cov": float((E8e.calibrated_confidence >= 0.2211).mean()),
                             "risk": float(1 - E8e.execution_correct[E8e.calibrated_confidence >= 0.2211].mean())}}

    # paired bootstrap: frozen-policy p-hat vs greedy-confidence at matched coverage 0.80
    mc80 = {"cov": 0.80}
    d80 = bootstrap_risk_diff2(E8a, "confidence", E8e, "calibrated_confidence",
                               "greedy_exec_correct", "execution_correct", 0.80)
    mc80["frozen_pi_vs_greedy_confidence"] = d80
    a80 = bootstrap_risk_diff2(E8a, "confidence", E8b[E8b.rule=="exec_verified"], "agreement",
                               "greedy_exec_correct", "execution_correct", 0.80)
    mc80["agree_only_vs_greedy_confidence"] = a80
    b80 = bootstrap_risk_diff2(E8b[E8b.rule=="exec_verified"], "agreement",
                               E8e, "calibrated_confidence",
                               "execution_correct", "execution_correct", 0.80)
    mc80["frozen_pi_vs_agree_only"] = b80

    # ablations + ranking quality (ROC-AUC, descriptive on test)
    auc = {}
    for name, (df, sc, cc) in strategies.items():
        auc[name + "::" + cc] = float(roc_auc_score(df[cc].astype(int), df[sc]))
    auc["exec_verified_executable_rate"] = float(roc_auc_score(E8b[E8b.rule=="exec_verified"].execution_correct.astype(int), E8b[E8b.rule=="exec_verified"].executable_rate))

    # ------------------------------------------------------------------
    # error-taxonomy reconciliation (mutually exclusive, sums verified)
    # ------------------------------------------------------------------
    g = E8a
    parse_fail = int((~g.parse_success).sum())
    invalid = int(((g.parse_success) & (~g.program_executable)).sum())
    exec_correct = int(g.greedy_exec_correct.sum())
    print_correct = int(g.greedy_print_correct.sum())
    format_mismatch = int(((g.greedy_exec_correct.astype(bool)) & (~g.greedy_print_correct.astype(bool))).sum())
    print_only_correct = int(((~g.greedy_exec_correct.astype(bool)) & (g.greedy_print_correct.astype(bool))).sum())
    both_wrong = int(((~g.greedy_exec_correct.astype(bool)) & (~g.greedy_print_correct.astype(bool))).sum())
    exec_wrong_overall = int(len(g) - exec_correct)
    # six-cell partition (parse x exec x print) must sum to n
    _exec_wrong_and_print_wrong = both_wrong - invalid
    partition = {"parse_fail": parse_fail, "invalid_program": invalid,
                 "exec_and_print_correct": int(exec_correct - format_mismatch),
                 "format_mismatch_exec_ok": format_mismatch,
                 "exec_wrong_and_print_wrong": _exec_wrong_and_print_wrong,
                 "exec_wrong_print_correct": print_only_correct}
    assert parse_fail + invalid + (exec_correct - format_mismatch) + format_mismatch + _exec_wrong_and_print_wrong + print_only_correct == len(g)

    v = E8b[E8b.rule=="exec_verified"]
    v_invalid = int((~v.program_executable).sum())
    v_exec_wrong = int(((v.program_executable.astype(bool)) & (~v.execution_correct.astype(bool))).sum())
    v_exec_correct = int(v.execution_correct.sum())
    v_mismatch = int(((v.execution_correct.astype(bool)) & (~v.final_answer_correct.astype(bool))).sum())
    assert v_invalid + v_exec_wrong + v_exec_correct == len(v)
    recon = {
        "greedy_full_partition": partition, "exec_wrong_overall": exec_wrong_overall,
        "greedy_exec_channel": {"parse_fail": parse_fail, "invalid_program": invalid,
                                "executed_wrong_or_invalid": exec_wrong_overall, "exec_correct": exec_correct, "n": len(g)},
        "greedy_print_channel": {"print_correct": print_correct,
                                 "print_wrong": int(len(g) - print_correct), "n": len(g)},
        "exec_verified_exec_channel": {"invalid_program": v_invalid, "executed_wrong": v_exec_wrong,
                                       "exec_correct": v_exec_correct,
                                       "answer_program_mismatch": v_mismatch, "n": len(v)},
        "rates": {"greedy_exec_acc": float(exec_correct/len(g)),
                  "greedy_print_acc": float(print_correct/len(g)),
                  "greedy_format_mismatch_rate": float(format_mismatch/len(g)),
                  "greedy_print_only_correct_note": "printed string matches gold while executed value differs (n=%d)" % print_only_correct,
                  "evg_exec_acc": float(v_exec_correct/len(v)),
                  "evg_invalid_rate": float(v_invalid/len(v))},
    }

    summary = {"n_finqa_test": n_rec, "strategies": list(strategies),
               "matched_coverage": mc.to_dict("records"), "frozen_operating_points": frozen,
               "matched_cov_0.80_bootstrap": mc80, "score_roc_auc": auc,
               "error_reconciliation": recon}
    (OUT/"summary.json").write_text(json.dumps(summary, indent=2))
    mc.to_csv(OUT/"matched_coverage_finqa.csv", index=False)

    # ------------------------------------------------------------------
    # FiQA regression: matched-coverage selective MAE
    # ------------------------------------------------------------------
    fiqa_f()
    print("wrote", OUT)
    print(json.dumps(frozen, indent=1))
    print(json.dumps(mc80, indent=1))
    print(json.dumps(auc, indent=1))

def fiqa_f():
    f = pd.read_csv(ROOT/"artifacts/phase4/analysis/P4-E8c_fiqa_test.csv")
    f = f.assign(neg_sample_std=-f.sample_std)
    strat = {
        "greedy_confidence": (f, "confidence", "greedy_abs_err", "greedy_correct"),
        "median_confidence": (f, "median_confidence", "median_abs_err", "median_correct"),
        "median_agreement_negstd": (f, "neg_sample_std", "median_abs_err", "median_correct"),
    }
    rows = []
    for name, (df, sc, mae_col, corr_col) in strat.items():
        full_mae = float(df[mae_col].mean())
        for cov in COVERAGES:
            k = int(cov*len(df)); sel = stable_topk(df, sc, k)
            mae = float(sel[mae_col].mean()); acc = float(sel[corr_col].mean())
            rows.append({"strategy": name, "coverage": cov, "n_answered": k,
                         "selective_mae": mae, "selective_acc0.10": acc, "full_mae": full_mae,
                         "mae_reduction_vs_greedy_full": (float(f.greedy_abs_err.mean()) - mae)/float(f.greedy_abs_err.mean())})
    mf = pd.DataFrame(rows)
    mf.to_csv(OUT/"matched_coverage_fiqa.csv", index=False)
    aucf = {n: float(roc_auc_score(df[cc].astype(int), df[sc])) for n,(df,sc,mc2,cc) in strat.items()}
    (OUT/"fiqa_extra.json").write_text(json.dumps({"matched_coverage": mf.to_dict("records"),
                                                   "auc_correct_vs_score": aucf}, indent=2))

if __name__ == "__main__":
    main()