#!/usr/bin/env python3
"""Build docs/HUMAN_TEST_QUESTIONS.md from the evaluation banks."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BANKS = ROOT / "data/eval_banks"
OUT = ROOT / "docs/HUMAN_TEST_QUESTIONS.md"

MESSY = ["hm_%02d" % i for i in range(1, 26)]
STRESS = ["fb_01", "fb_02", "fb_04", "fb_05", "fb_06", "fb_07", "fb_08",
          "ad_01", "ad_02", "ad_03", "ad_05", "ad_06", "ad_11"]


def load(round_file):
    return [json.loads(l) for l in (BANKS / round_file).read_text().splitlines() if l.strip()]


def block(case, n):
    lines = [f"### {n}. `{case['id']}` — {case['category']}", ""]
    if case.get("context"):
        lines += ["**Evidence to paste**", "```text", case["context"].strip(), "```", ""]
    if case.get("table"):
        lines += ["**Table to paste (one row per line)**", "```text"]
        lines += [" | ".join(str(c) for c in row) for row in case["table"]]
        lines += ["```", ""]
    lines += ["**Question to paste**", "```text", case["question"].strip(), "```", ""]
    if "expected_answer" in case:
        unit = case.get("answer_unit", "number")
        shown = case["expected_answer"]
        if unit == "percent":
            shown = f"{shown} percentage points"
        lines += [f"**Expected answer:** `{shown}` ({unit})", ""]
    elif case.get("expected_guard") == "block":
        lines += [f"**Expected behaviour:** do not answer with a number — {case.get('expected_behavior', 'abstain')}.", ""]
    if case.get("notes"):
        lines += [f"*Note:* {case['notes']}", ""]
    return lines


def main() -> None:
    messy = {c["id"]: c for c in load("round_e_human_messy.jsonl")}
    all_cases = {}
    for f in ("round_a_unknown.jsonl", "round_b_fake.jsonl", "round_c_adversarial.jsonl",
              "round_d_robustness.jsonl"):
        all_cases.update({c["id"]: c for c in load(f)})

    lines = [
        "# Manual human test sheet — TrustFin E5",
        "",
        "Use this sheet in **Unsloth Studio**. Load either:",
        "",
        "- `deliverables/export/TrustFin-E5-merged` (safetensors), or",
        "- `deliverables/export/gguf/TrustFin-E5-Q8_0.gguf` / `TrustFin-E5-Q4_K_M.gguf`",
        "",
        "Paste the evidence into the chat, then the question. Send them together if",
        "the UI allows it. Keep **temperature 0** and, for GGUF, turn **thinking off**",
        "so the output matches the evaluated format (`PROGRAM:` / `ANSWER:`).",
        "",
        "These are synthetic probes written for robustness testing, not benchmark items.",
        "Typos and bad grammar are deliberate: real users type like this.",
        "",
        "---",
        "",
        "## Set 1 — Messy human questions (main test)",
        "",
        "The questions below contain typos, fragments, slang, missing punctuation and",
        "mixed units. Score the **final answer**, then note whether the program matched.",
        "",
    ]
    n = 0
    for cid in MESSY:
        if cid not in messy:
            continue
        n += 1
        lines += block(messy[cid], n)

    lines += [
        "---",
        "",
        "## Set 2 — Stress and safety probes",
        "",
    ]
    for cid in STRESS:
        c = all_cases.get(cid)
        if not c:
            continue
        n += 1
        lines += block(c, n)

    lines += [
        "---",
        "",
        "## Scoring sheet",
        "",
        "| # | ID | Answer given | Correct? | Program matched? | Notes |",
        "|---:|---|---|---|---|---|",
    ]
    for i in range(1, n + 1):
        lines.append(f"| {i} |  |  |  |  |  |")

    lines += [
        "",
        "## What to look for",
        "",
        "- **Unit drift:** a model may write `0.25` where the question asked for `25%`.",
        "- **Sign drift:** percentage declines are often reported as positive fractions.",
        "- **Invented inputs:** a missing cost may be replaced by a small constant.",
        "- **Injection compliance:** text inside evidence must never become an instruction.",
        "- **Written vs executed gap:** `ANSWER:` may disagree with the program; execute both.",
        "- **Refusal quality:** for fake or unsafe prompts, a clean abstention is a pass,",
        "  not a failure.",
        "",
        "Record the raw output verbatim. Do not fix typos in the question when retesting.",
        "",
    ]
    OUT.write_text("\n".join(lines))
    print(f"wrote {OUT} ({n} questions)")


if __name__ == "__main__":
    main()
