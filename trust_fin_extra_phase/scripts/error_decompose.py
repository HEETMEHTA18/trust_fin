#!/usr/bin/env python3
"""error_decompose.py — where do FinQA protected-test points actually go?

Read-only analysis (roadmap Phase-6 planning): decomposes the frozen greedy
P4-E8a results beyond the parse/execute/print taxonomy of docs/exp_4(c).md §3.

  exec-wrong records   -> wrong numbers selected (retrieval/value pick)
                          vs same numbers but wrong operation/structure (reasoning)
                          vs mixed / unexecutable
  exec-right/print-wrong (the 355-record format channel)
                          -> rounding vs scaling/percent vs unrelated printed value

Inputs : artifacts/phase2_1/eval/finqa_test.jsonl   (gold programs)
         artifacts/phase4/predictions/P4-E8a_greedy_scored.csv   (frozen)
Output : artifacts/phase5/analysis/error_decompose.json + stdout summary

Nothing here tunes on the protected test — it only *describes* frozen errors.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from paths import find_repo_root, resolve_recorded  # noqa: E402

REPO = find_repo_root()
WS = REPO / "trust_fin_extra_phase"

OP_RE = re.compile(r"([a-z_]+)\(")
NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def parse_program(prog: str) -> tuple[list[str], list[str]]:
    """Return (ordered operations, numeric literals) of a FinQA program."""
    ops = OP_RE.findall(str(prog).lower())
    nums = [float(x) for x in NUM_RE.findall(str(prog))]
    return ops, nums


def same_value(a: float, b: float, tol: float = 1e-6) -> bool:
    if a == b:
        return True
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def literal_overlap(pred: list[float], gold: list[float]) -> tuple[int, int]:
    """(matched gold literals, matched pred literals) with tolerance pairing."""
    g_left, p_left = list(gold), list(pred)
    matched = 0
    for g in list(g_left):
        for i, p in enumerate(p_left):
            if same_value(g, p, tol=1e-9):
                matched += 1
                g_left.remove(g)
                p_left.pop(i)
                break
    return matched, len(gold) - matched, len(pred) - matched


def main() -> None:
    gold_path = REPO / "artifacts/phase2_1/eval/finqa_test.jsonl"
    scored_path = WS / "artifacts/phase4/predictions/P4-E8a_greedy_scored.csv"

    gold = {}
    with gold_path.open() as f:
        for line in f:
            r = json.loads(line)
            gold[r["record_id"]] = r
    df = pd.read_csv(scored_path)

    exec_wrong, print_only_wrong = [], []
    for _, row in df.iterrows():
        g = gold[row["record_id"]]
        g_ops, g_nums = parse_program(g["gold_program"])
        p_ops, p_nums = parse_program(row["program"])
        rec = {
            "record_id": row["record_id"],
            "gold_program": g["gold_program"],
            "pred_program": str(row["program"]),
            "gold_ops": g_ops, "pred_ops": p_ops,
            "gold_len": len(g_ops), "pred_len": len(p_ops),
            "confidence": float(row["confidence"]),
        }
        if not bool(row["greedy_exec_correct"]):
            if not bool(row["program_executable"]):
                rec["cause"] = "unexecutable"
            else:
                matched, gold_miss, pred_extra = literal_overlap(p_nums, g_nums)
                same_numbers = gold_miss == 0 and pred_extra == 0
                same_ops = p_ops == g_ops
                if same_numbers and same_ops:
                    rec["cause"] = "exec_discrepancy_same_program"  # tolerance/executor edge
                elif same_numbers:
                    rec["cause"] = "wrong_operation_same_numbers"
                elif matched == 0:
                    rec["cause"] = "wrong_numbers_no_overlap"
                else:
                    rec["cause"] = "mixed_partial_numbers"
                rec["gold_missed_literals"] = gold_miss
                rec["pred_extra_literals"] = pred_extra
            exec_wrong.append(rec)
        elif not bool(row["greedy_print_correct"]):
            # format channel: executed value is right, printed string is wrong
            try:
                printed = float(str(row["parsed_answer"]).replace(",", ""))
                executed = float(row["executed_answer"])
            except (TypeError, ValueError):
                print_only_wrong.append({**rec, "cause": "unparseable_print"})
                continue
            if same_value(printed, executed, tol=1e-9):
                cause = "identical_but_gold_tol"      # exact-match gold comparison edge
            elif executed != 0 and same_value(printed / executed, 100.0, 1e-6):
                cause = "scaling_x100_percent"
            elif executed != 0 and same_value(printed / executed, 0.01, 1e-6):
                cause = "scaling_div100"
            elif same_value(printed, executed, tol=1e-3):
                cause = "rounding"
            else:
                cause = "unrelated_printed_value"
            print_only_wrong.append({**rec, "cause": cause,
                                     "printed": printed, "executed": executed})

    # ---- summaries ---------------------------------------------------------
    def tally(items, key="cause"):
        return dict(Counter(i[key] for i in items).most_common())

    n = len(df)
    exec_wrong_by_len = Counter(r["gold_len"] for r in exec_wrong
                                if r["cause"] != "unexecutable")
    gold_by_len = Counter()
    correct_by_len = Counter()
    for rid, g in gold.items():
        row = df[df.record_id == rid]
        if row.empty:
            continue
        l = len(parse_program(g["gold_program"])[0])
        gold_by_len[l] += 1
        if bool(row.iloc[0]["greedy_exec_correct"]):
            correct_by_len[l] += 1

    depth_acc = {str(l): round(correct_by_len[l] / gold_by_len[l], 4)
                 for l in sorted(gold_by_len)}
    conf_wrong = [r["confidence"] for r in exec_wrong]
    conf_right = df[df.greedy_exec_correct]["confidence"].tolist()

    summary = {
        "schema": "trustfin-error-decompose-v1",
        "source": "frozen P4-E8a greedy (descriptive only; no tuning on protected test)",
        "n": n,
        "headline": {
            "exec_accuracy": round(float(df.greedy_exec_correct.mean()), 4),
            "print_accuracy": round(float(df.greedy_print_correct.mean()), 4),
        },
        "exec_wrong_total": len(exec_wrong),
        "exec_wrong_causes": tally(exec_wrong),
        "print_wrong_but_exec_right_total": len(print_only_wrong),
        "print_channel_causes": tally(print_only_wrong),
        "exec_accuracy_by_gold_program_length": depth_acc,
        "mean_confidence": {
            "exec_right": round(sum(conf_right) / max(len(conf_right), 1), 4),
            "exec_wrong": round(sum(conf_wrong) / max(len(conf_wrong), 1), 4),
        },
        "exec_wrong_records": exec_wrong,
        "print_channel_records": print_only_wrong,
    }

    out = WS / "artifacts/phase5/analysis/error_decompose.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=1) + "\n")

    print(f"n={n} exec_acc={summary['headline']['exec_accuracy']} "
          f"print_acc={summary['headline']['print_accuracy']}")
    print(f"\nexec-wrong ({len(exec_wrong)}):")
    for k, v in summary["exec_wrong_causes"].items():
        print(f"  {v:5d}  {k}")
    print(f"\nprint-wrong-but-exec-right ({len(print_only_wrong)}):")
    for k, v in summary["print_channel_causes"].items():
        print(f"  {v:5d}  {k}")
    print(f"\naccuracy by gold program length (steps): {depth_acc}")
    print(f"mean confidence: right={summary['mean_confidence']['exec_right']} "
          f"wrong={summary['mean_confidence']['exec_wrong']}")
    print(f"\n-> {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
