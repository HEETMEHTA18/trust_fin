from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "paper" / "figures"
PHASE3_METRICS = ROOT / "artifacts" / "phase3" / "metrics" / "phase3_validation_metrics.csv"
PHASE4_ANALYSIS = ROOT / "trust_fin_extra_phase" / "artifacts" / "phase4" / "analysis"
PHASE4_MANIFESTS = ROOT / "trust_fin_extra_phase" / "artifacts" / "phase4" / "manifests"
PHASE3_ANALYSIS = ROOT / "artifacts" / "phase3" / "analysis"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.labelsize": 9,
    "legend.fontsize": 7,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
})


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(FIGURES / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def phase3_baselines() -> None:
    metrics = pd.read_csv(PHASE3_METRICS).set_index("experiment")
    experiments = ["E3", "E5", "E8", "E11"]
    labels = ["FinQA-only\nQLoRA", "Qwen3 joint\nQLoRA", "Phi joint\nQLoRA", "Qwen3 joint\nnatural"]

    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    values = [metrics.loc[item, "FinQA.execution_accuracy"] for item in experiments]
    bars = ax.bar(labels, values, color=["#4c78a8", "#2a9d8f", "#f4a261", "#e76f51"], width=0.65)
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    ax.set_ylim(0, 0.8)
    ax.set_ylabel("FinQA execution accuracy")
    ax.set_title("Phase 3 FinQA validation")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    save(fig, "fig_phase3_finqa_execution_accuracy.png")

    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    values = [metrics.loc[item, "FiQA.mae"] for item in experiments]
    bars = ax.bar(labels, values, color=["#4c78a8", "#2a9d8f", "#f4a261", "#e76f51"], width=0.65)
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    ax.set_ylim(0, 0.6)
    ax.set_ylabel("FiQA MAE (lower is better)")
    ax.set_title("Phase 3 FiQA validation")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    save(fig, "fig_phase3_fiqa_mae.png")


def phase4_fiqa_aggregation() -> None:
    data = pd.read_csv(PHASE4_ANALYSIS / "P4-E3b_fiqa_sampling.csv")
    labels = {
        "greedy": "Greedy",
        "median_n": "Median of samples",
        "conf_weighted_mean": "Confidence-weighted mean",
        "mean_n": "Mean of samples",
    }
    data["label"] = data["aggregator"].map(labels)
    data = data.sort_values("mae", ascending=False)
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    bars = ax.barh(data["label"], data["mae"], color=["#e76f51", "#2a9d8f", "#2a9d8f", "#2a9d8f"])
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=8)
    ax.set_xlabel("MAE on FiQA validation")
    ax.set_title("Sample aggregation on FiQA")
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    save(fig, "fig_phase4_fiqa_aggregation.png")


def phase4_selection() -> None:
    data = pd.read_csv(PHASE4_ANALYSIS / "P4-E3_selection_ablation.csv")
    names = {
        "exec_verified": "Execution-verified",
        "best_of_n": "Best of samples",
        "self_consistency": "Self-consistency",
        "c0_exec": "Candidate 0 execution",
        "c0_print": "Candidate 0 printed",
    }
    fig, ax = plt.subplots(figsize=(5.4, 3.3))
    for rule, group in data.groupby("rule", sort=False):
        if rule not in names:
            continue
        group = group.sort_values("n_candidates")
        ax.plot(group["n_candidates"], group["execution_accuracy"], marker="o", linewidth=1.5, label=names[rule])
    ax.set_xlabel("Number of sampled candidates")
    ax.set_ylabel("Execution accuracy")
    ax.set_title("Selection-rule ablation on FinQA validation")
    ax.set_xticks([1, 2, 4, 8])
    ax.set_ylim(0.62, 0.69)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", frameon=True)
    save(fig, "fig_phase4_selection_ablation.png")


def phase4_risk_coverage() -> None:
    with (PHASE4_MANIFESTS / "p4_risk_coverage_curves.json").open() as handle:
        curves = json.load(handle)
    names = {
        "finqa_exec_verified": "FinQA execution-verified",
        "finqa_greedy_print": "FinQA greedy",
        "fiqa_greedy": "FiQA greedy",
        "fiqa_median_n": "FiQA median",
    }
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), sharey=False)
    for key, label in names.items():
        if key not in curves:
            continue
        series = curves[key].get("calibrated", curves[key].get("raw", []))
        x = [item["coverage"] for item in series]
        y = [item["risk"] for item in series]
        axis = axes[0] if key.startswith("finqa") else axes[1]
        axis.plot(x, y, marker=".", linewidth=1.2, markersize=3, label=label)
    axes[0].set_title("FinQA")
    axes[1].set_title("FiQA")
    for axis in axes:
        axis.set_xlabel("Coverage")
        axis.set_ylabel("Selective risk")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=6, loc="best")
    fig.suptitle("Risk-coverage curves on validation data", y=1.02)
    save(fig, "fig_phase4_risk_coverage.png")


def phase3_diagnostics() -> None:
    failure = pd.read_csv(PHASE3_ANALYSIS / "E12_finqa_failure_summary.csv")
    failure_labels = {
        "correct": "Correct",
        "reasoning_execution_error": "Reasoning /\nexec. error",
        "program_right_answer_wrong": "Program right,\nanswer wrong",
        "alternate_program_same_execution": "Alt. program,\nsame execution",
        "invalid_program": "Invalid program",
        "answer_right_program_wrong": "Answer right,\nprogram wrong",
    }
    failure["label"] = failure["failure_type"].map(failure_labels)
    failure = failure.sort_values("count")
    fig, ax = plt.subplots(figsize=(5.6, 3.5))
    bars = ax.barh(failure["label"], failure["count"], color=["#008c95", "#6a1b9a", "#1976d2", "#ef6c00", "#c62828", "#2e7d32"])
    ax.bar_label(bars, labels=[f"{n} ({rate:.1%})" for n, rate in zip(failure["count"], failure["rate"])], padding=3, fontsize=8)
    ax.set_xlabel("FinQA validation records (n = 883)")
    ax.set_title("FinQA failure taxonomy of the frozen E5 model")
    ax.set_xlim(0, 390)
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    save(fig, "fig1_finqa_failure_taxonomy.png")

    cases = pd.read_csv(PHASE3_ANALYSIS / "E12_finqa_failure_cases.csv")
    category_order = list(failure_labels.values())
    cases["label"] = cases["failure_type"].map(failure_labels)
    cases = cases.dropna(subset=["label", "confidence"])
    fig, ax = plt.subplots(figsize=(6.0, 3.7))
    data = [cases.loc[cases["label"] == label, "confidence"] for label in category_order if (cases["label"] == label).any()]
    labels = [label for label in category_order if (cases["label"] == label).any()]
    colors = ["#59a14f", "#e15759", "#f28e2b", "#4e79a7", "#af7aa1", "#76b7b2"]
    parts = ax.violinplot(data, showmeans=False, showextrema=False)
    for body, color in zip(parts["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.42)
    box = ax.boxplot(
        data,
        widths=0.10,
        showfliers=False,
        patch_artist=True,
        medianprops={"color": "#1f4e79", "linewidth": 2},
        boxprops={"facecolor": "white", "edgecolor": "#1f4e79", "linewidth": 1},
        whiskerprops={"color": "#1f4e79", "linewidth": 1},
        capprops={"color": "#1f4e79", "linewidth": 1},
    )
    for patch in box["boxes"]:
        patch.set_alpha(0.9)
    rng = __import__("numpy").random.default_rng(42)
    for index, values in enumerate(data, start=1):
        sample = values.to_numpy()
        jitter = rng.uniform(-0.055, 0.055, len(sample))
        ax.scatter(index + jitter, sample, s=3, color="#333333", alpha=0.12, linewidths=0, zorder=1)
        median = float(values.median())
        mean = float(values.mean())
        ax.text(index, 1.005, f"n={len(sample)}\nmed={median:.1%}\nmean={mean:.1%}",
                ha="center", va="bottom", fontsize=6.5, color="#222222")
    short_labels = ["Correct", "Reasoning /\nexec.", "Program right", "Alt. program", "Invalid", "Answer right"]
    ax.set_xticks(range(1, len(labels) + 1), short_labels[:len(labels)], rotation=25, ha="right")
    ax.set_ylabel("Mean token confidence")
    ax.set_ylim(0.5, 1.13)
    ax.set_title("Confidence by FinQA failure category")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    save(fig, "fig3_finqa_confidence_by_failure.png")

    bins = pd.read_csv(PHASE3_ANALYSIS / "E14_confidence_bins.csv")
    bins = bins[bins["dataset"] == "FinQA"].copy()
    bin_labels = ["Low", "Lower-mid", "Mid", "Upper-mid", "High"]
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    bars = ax.bar(bin_labels, bins["quality"], color="#4c78a8", width=0.65)
    ax.bar_label(bars, labels=[f"{value:.1%}" for value in bins["quality"]], padding=3, fontsize=8)
    ax.set_xlabel("Confidence bin (ordered)")
    ax.set_ylabel("Execution accuracy")
    ax.set_ylim(0, 1)
    ax.set_title("Accuracy by self-reported confidence")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    save(fig, "fig4_finqa_confidence_bins_accuracy.png")

    magnitude = pd.read_csv(PHASE3_ANALYSIS / "E12_finqa_failure_cases.csv")
    magnitude = magnitude[magnitude["failure_type"].isin([
        "program_right_answer_wrong",
        "reasoning_execution_error",
    ])].copy()
    magnitude["pred_answer"] = pd.to_numeric(magnitude["pred_answer"], errors="coerce")
    magnitude["gold_answer"] = pd.to_numeric(magnitude["gold_answer"], errors="coerce")
    magnitude["relative_error"] = (
        (magnitude["pred_answer"] - magnitude["gold_answer"]).abs()
        / magnitude["gold_answer"].abs().clip(lower=1e-12)
    )
    magnitude_order = ["program_right_answer_wrong", "reasoning_execution_error"]
    magnitude_labels = ["Program right,\nanswer wrong", "Reasoning /\nexecution error"]
    error_data = [
        magnitude.loc[magnitude["failure_type"] == failure_type, "relative_error"].dropna().to_numpy()
        for failure_type in magnitude_order
    ]
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    box = ax.boxplot(
        error_data,
        vert=False,
        patch_artist=True,
        widths=0.48,
        showfliers=False,
        tick_labels=magnitude_labels,
        medianprops={"color": "#222222", "linewidth": 2},
        whiskerprops={"color": "#666666"},
        capprops={"color": "#666666"},
    )
    for patch, color in zip(box["boxes"], ["#f4a261", "#e76f6a"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    rng = __import__("numpy").random.default_rng(42)
    for position, values in enumerate(error_data, start=1):
        jitter = rng.uniform(-0.06, 0.06, len(values))
        ax.scatter(values, position + jitter, s=8, color="#999999", alpha=0.28, zorder=1)
        median = float(__import__("numpy").median(values))
        ax.annotate(
            f"median {median:.1%}",
            xy=(median, position),
            xytext=(median, position + 0.34),
            textcoords="data",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#333333",
            arrowprops={"arrowstyle": "-", "color": "#555555", "linewidth": 0.8},
        )
    ax.set_xscale("log")
    ax.set_xlabel("Relative absolute error |pred - gold| / |gold| (log scale)")
    ax.set_title("Failure magnitude: near-misses versus catastrophic misreads")
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    save(fig, "fig5_finqa_error_magnitude.png")

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.4), constrained_layout=True)
    failure_plot = failure.sort_values("count")
    axes[0, 0].barh(failure_plot["label"], failure_plot["count"], color="#4c78a8")
    axes[0, 0].set_title("Failure taxonomy")
    axes[0, 0].set_xlabel("Records")
    axes[0, 0].tick_params(axis="y", labelsize=6)
    axes[0, 0].grid(axis="x", alpha=0.2)
    axes[0, 1].bar(bin_labels, bins["quality"], color="#59a14f")
    axes[0, 1].set_title("Accuracy by confidence bin")
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].tick_params(axis="x", labelrotation=25)
    axes[0, 1].grid(axis="y", alpha=0.2)
    axes[1, 0].violinplot(data, showmeans=True, showextrema=False)
    axes[1, 0].set_title("Confidence by failure category")
    axes[1, 0].set_ylim(0.5, 1.01)
    axes[1, 0].set_xticks(range(1, len(labels) + 1), short_labels[:len(labels)], fontsize=6)
    axes[1, 0].tick_params(axis="x", labelrotation=20)
    axes[1, 0].grid(axis="y", alpha=0.2)
    fiqa = pd.read_csv(PHASE3_ANALYSIS / "E13_fiqa_failure_cases.csv")
    fiqa = fiqa.dropna(subset=["absolute_error", "error_band"])
    error_order = ["<=0.10", "0.10-0.25", "0.25-0.50", "0.50-1.00", ">1.00"]
    error_data = [fiqa.loc[fiqa["error_band"] == label, "absolute_error"] for label in error_order]
    axes[1, 1].boxplot(error_data, tick_labels=error_order, showfliers=False)
    axes[1, 1].set_title("FiQA absolute-error bands")
    axes[1, 1].set_ylabel("Absolute error")
    axes[1, 1].tick_params(axis="x", labelrotation=25)
    axes[1, 1].grid(axis="y", alpha=0.2)
    fig.suptitle("Phase 3 validation failure overview", fontsize=12)
    save(fig, "fig6_finqa_failure_overview.png")


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    phase3_baselines()
    phase4_fiqa_aggregation()
    phase4_selection()
    phase4_risk_coverage()
    phase3_diagnostics()


if __name__ == "__main__":
    main()