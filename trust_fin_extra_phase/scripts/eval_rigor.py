"""Evaluation-rigor analysis for Phase-4 EVG-CAS.

Computes, from the *frozen* prediction artifacts only (no new model runs):
  1. C1 significance on the protected FinQA test  -- paired McNemar (exact) + bootstrap CI
  2. Agreement-bin accuracies with Wilson 95% CIs
  3. Error taxonomy (invalid-program / execution-wrong / format-mismatch) for greedy vs EVG
  4. Transfer confidence intervals for the frozen abstention policy (bootstrap @ fixed theta)
  5. Difficulty bucketing (selected-program length) vs abstention benefit
  6. FiQA: bootstrap MAE diff (greedy vs median-of-n) and abstention CI

Outputs: artifacts/phase4/analysis/eval_rigor/*.{json,csv}
"""
from __future__ import annotations
import json, os
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/phase4/analysis/eval_rigor"
OUT.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(42)
B = 10000

def bootstrap_ci(x, B=B, alpha=0.05) -> dict:
    boot = RNG.choice(x, size=(B, len(x)), replace=True).mean(axis=1)
    lo, hi = np.percentile(boot, [100*alpha/2, 100*(1-alpha/2)])
    return {"mean": float(x.mean()), "ci95_lo": float(lo), "ci95_hi": float(hi)}

def wilson_ci(k, n, z=1.96):
    if n == 0: return (0.0, 1.0)
    p = k / n
    d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d
    s = z * np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return (float(c - s), float(c + s))

def exact_mcnemar(b, c):
    b, c = int(b), int(c)
    n = b + c
    if n == 0: return 1.0, b, c, n
    from math import comb
    # two-sided exact p = 2 * sum_{k <= min(b,c)} C(n,k)/2^n
    kmin = min(b, c)
    p_tail = sum(comb(n, k) for k in range(0, kmin+1)) / (2**n)
    return min(1.0, 2.0*p_tail), b, c, n

def load_e8b(rule="exec_verified"):
    df = pd.read_csv(ROOT/"artifacts/phase4/predictions/P4-E8b.csv")
    return df[df.rule == rule]

def main():
    sumr = {}
    # ------------------------------------------------------------------
    # 1. C1: FinQA protected test, greedy vs exec_verified (paired)
    # ------------------------------------------------------------------
    g = pd.read_csv(ROOT/"artifacts/phase4/predictions/P4-E8a_greedy_scored.csv")
    v = load_e8b("exec_verified")
    m = g[["record_id","greedy_exec_correct"]].merge(
        v[["record_id","execution_correct","program_executable","final_answer_correct"]], on="record_id")
    assert len(m) == 1147, len(m)
    a = m.greedy_exec_correct.astype(bool).to_numpy()
    b = m.execution_correct.astype(bool).to_numpy()          # b: both, c: EVG-only...
    both = int((a & b).sum()); evg_only = int((~a & b).sum()); g_only = int((a & ~b).sum()); none = int((~a & ~b).sum())
    p_exact, _, _, _ = exact_mcnemar(evg_only, g_only)
    d = b.astype(int) - a.astype(int)
    delta_boot = [float((RNG.choice(d, len(d), replace=True)).mean()) for _ in range(B)]
    lo, hi = np.percentile(delta_boot, [2.5, 97.5])
    p_gt0 = np.mean(np.array(delta_boot) > 0)
    acc_g = a.mean(); acc_e = b.mean()
    c1 = {
        "n": len(m),
        "greedy_exec_acc": float(acc_g), "exec_verified_exec_acc": float(acc_e),
        "delta": float(acc_e - acc_g),
        "confusion": {"both_correct": both, "evg_only": evg_only,
                      "greedy_only": g_only, "both_wrong": none},
        "mcnemar_exact_p": p_exact,
        "bootstrap_delta_ci95": [float(lo), float(hi)],
        "p_delta_gt_0": float(p_gt0),
    }
    sumr["c1_finqa_test"] = c1

    # candidate-0 (prints) vs EVG on test as an internal sanity check
    c0 = load_e8b("c0_exec")
    mc = g[["record_id","greedy_exec_correct"]].merge(
        c0[["record_id","execution_correct"]], on="record_id")
    sumr["c1_sanity_c0_vs_evg"] = {
        "c0_exec_acc": float(c0.execution_correct.mean()), "delta_vs_greedy": float(c0.execution_correct.mean() - acc_g),
        "n": int(len(c0))}

    # ------------------------------------------------------------------
    # 2. Agreement bins with Wilson CIs (protected test, EVG)
    # ------------------------------------------------------------------
    bins = pd.cut(v.agreement, [0, 0.25, 0.5, 0.75, 1.001], include_lowest=False)
    t = v.assign(bin=bins).groupby("bin", observed=True).agg(
        n=("execution_correct", "size"), acc=("execution_correct", "mean"))
    ag = []
    for b_, row in t.iterrows():
        k = int(round(row.n * row.acc))
        lo, hi = wilson_ci(k, int(row.n))
        ag.append({"bin": str(b_), "n": int(row.n), "acc": float(row.acc),
                   "acc_ci95": [lo, hi]})
    sumr["agreement_bins_test"] = ag
    pd.DataFrame(ag).to_csv(OUT/"agreement_bins.csv", index=False)

    # ------------------------------------------------------------------
    # 3. Error taxonomy (greedy vs EVG) on protected test
    # ------------------------------------------------------------------
    tax = {"greedy": {
        "n": int(len(g)),
        "parse_fail": int((~g.parse_success).sum()),
        "execution_wrong": int(((g.parse_success) & (~g.greedy_exec_correct.astype(bool))).sum()),
        "exec_correct_but_print_wrong_format": int(((g.greedy_exec_correct.astype(bool)) & (~g.greedy_print_correct.astype(bool))).sum()),
        "program_invalid_rate": float((~g.program_executable).mean()),
    }, "exec_verified": {
        "n": int(len(v)),
        "program_invalid": int((~v.program_executable).sum()),
        "execution_wrong": int(((v.program_executable.astype(bool)) & (~v.execution_correct.astype(bool))).sum()),
        "answer_program_mismatch": int(((v.execution_correct.astype(bool)) & (~v.final_answer_correct.astype(bool))).sum()),
        "program_invalid_rate": float((~v.program_executable).mean()),
    }}
    sumr["error_taxonomy_finqa_test"] = tax

    # ------------------------------------------------------------------
    # 4. Transfer CIs for the frozen abstention policy (FinQA)
    # ------------------------------------------------------------------
    e = pd.read_csv(ROOT/"artifacts/phase4/analysis/P4-E8e_finqa_abstention_scores.csv")
    ef = e[["record_id","calibrated_confidence","execution_correct"]].copy()
    ef["err"] = 1 - ef.execution_correct.astype(int)
    full_risk = ef.err.mean()
    # sanity: achieved coverage at frozen thresholds
    cov80 = (ef.calibrated_confidence >= 0.4090).mean()
    cov90 = (ef.calibrated_confidence >= 0.2211).mean()
    # bootstrap risk reduction @ 0.80
    stat = []
    for _ in range(B):
        s = ef.sample(n=len(ef), replace=True, random_state=RNG.integers(0, 2**31))
        def risk_at(theta):
            sel = s.calibrated_confidence >= theta
            if sel.sum() == 0: return 1.0, 0.0
            return float(s.err[sel].mean()), float(sel.mean())
        r80, cv80 = risk_at(0.4090); r90, cv90 = risk_at(0.2211)
        stat.append((r80, cv80, r90, cv90, (full_risk - r80) / full_risk))
    st = np.array(stat)
    q = lambda xcol: [float(np.percentile(xcol, 2.5)), float(np.percentile(xcol, 97.5))]
    sumr["transfer_finqa"] = {
        "frozen_thetas": {"theta_0.80": 0.4090, "theta_0.90": 0.2211},
        "achieved_coverage": {"cov_measured_0.80": float(cov80), "cov_measured_0.90": float(cov90)},
        "full_risk": float(full_risk),
        "selective_risk": {
            "at_0.80": {"mean": float(st[:,0].mean()), "ci95": q(st[:,0])},
            "at_0.90": {"mean": float(st[:,2].mean()), "ci95": q(st[:,2])}},
        "coverage": {
            "at_0.80": {"mean": float(st[:,1].mean()), "ci95": q(st[:,1])},
            "at_0.90": {"mean": float(st[:,3].mean()), "ci95": q(st[:,3])}},
        "error_reduction_at_0.80": {"mean": float(st[:,4].mean()), "ci95": q(st[:,4])},
    }

    # ------------------------------------------------------------------
    # 5. Difficulty buckets (arithmetic-op count, FinQA's own reasoning-depth proxy)
    #    vs abstention benefit
    # ------------------------------------------------------------------
    v = v.assign(
        n_ops=v.selected_program.astype(str).str.count(r"(add|subtract|multiply|divide)\s*\("),
    ).copy()
    qq = pd.cut(v.n_ops, bins=[-0.5, 0.5, 1.5, 2.5, 100],
            labels=["0 ops", "1 op", "2 ops", "3+ ops"])
    e8 = e[["record_id","calibrated_confidence","execution_correct"]].rename(columns={"execution_correct":"exec_correct_e"})
    vv = v.merge(e8, on="record_id").assign(bin=qq.to_numpy())
    vv["err"] = 1 - vv["exec_correct_e"].astype(int)
    rows = []
    for lab, sub in vv.groupby("bin", observed=True, sort=True):
        sel = sub.calibrated_confidence >= 0.4090
        ab = (~sel).mean()
        diff = sub.err.mean() - (sub.err[sel].mean() if sel.sum() else float("nan"))
        rows.append({"difficulty": str(lab), "n_ops_range": f"{sub.n_ops.min()}-{sub.n_ops.max()}", "n": int(len(sub)),
                     "acc_full": float(1 - sub.err.mean()),
                     "acc_selected_80": float(1 - sub.err[sel].mean()) if sel.sum() else None,
                     "abstention_rate": float(ab),
                     "risk_reduction": float(diff / sub.err.mean()) if sub.err.mean() > 0 else None})
    sums_check = sum(r["n"] for r in rows)
    # no fallback needed: cut() covers every record; assert instead
    sumr["difficulty_buckets_by_ops"] = rows
    pd.DataFrame(rows).to_csv(OUT/"difficulty_buckets.csv", index=False)

    # ------------------------------------------------------------------
    # 6. FiQA: greedy vs median-of-n (MAE) + abstention CI
    # ------------------------------------------------------------------
    f = pd.read_csv(ROOT/"artifacts/phase4/analysis/P4-E8c_fiqa_test.csv")
    d_f = f.greedy_abs_err.to_numpy()
    m_f = f.median_abs_err.to_numpy()
    mae = {"n": int(len(f)),
           "greedy_mae": float(d_f.mean()), "median_mae": float(m_f.mean()),
           "delta_mae_ci95": [float(np.percentile(RNG.choice(m_f-d_f, (B,len(f))).mean(1),2.5)),
                              float(np.percentile(RNG.choice(m_f-d_f, (B,len(f))).mean(1),97.5))]}
    sumr["c1_fiqa_test"] = mae

    fe = pd.read_csv(ROOT/"artifacts/phase4/analysis/P4-E8e_fiqa_abstention_scores.csv")
    thr80 = float(np.quantile(fe.calibrated_confidence, 0.20))   # threshold for coverage≈0.80
    st_f = []
    for _ in range(B):
        s = fe.sample(n=len(fe), replace=True, random_state=RNG.integers(0, 2**31))
        sel = s.calibrated_confidence >= thr80
        st_f.append((float(1 - s.greedy_correct[sel].mean()) if sel.sum() else 1.0, float(sel.mean())))
    stf = np.array(st_f)
    sumr["transfer_fiqa_ref80"] = {
        "coverage_mean": float(stf[:,1].mean()), "coverage_ci95": q(stf[:,1]),
        "selective_risk_mean": float(stf[:,0].mean()), "selective_risk_ci95": q(stf[:,0]),
        "full_risk": float(1 - fe.greedy_correct.mean()),
    }

    (OUT/"eval_rigor_summary.json").write_text(json.dumps(sumr, indent=2))
    print("wrrited eval_rigor_summary.json")
    print(json.dumps(sumr["c1_finqa_test"], indent=1))
    print(json.dumps(sumr["transfer_finqa"], indent=1)[:700])

if __name__ == "__main__":
    main()