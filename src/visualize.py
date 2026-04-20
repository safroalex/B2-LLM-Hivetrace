"""
Visualization module — generates all plots for the report.
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"

sns.set_theme(style="whitegrid", font_scale=1.1)
PALETTE = sns.color_palette("Set2")


def load_data():
    df = pd.read_csv(RESULTS_DIR / "raw_results.csv")
    with open(RESULTS_DIR / "analysis.json", "r", encoding="utf-8") as f:
        analysis = json.load(f)
    return df, analysis


# ──────────────────────────────────────────────────────────────────
# 1. Baseline accuracy per category (bar chart)
# ──────────────────────────────────────────────────────────────────

def plot_baseline_per_category(analysis: dict):
    cats = analysis["baseline_per_category"]
    names = list(cats.keys())
    acc = [cats[n]["accuracy"] for n in names]
    tpr = [cats[n]["tpr"] for n in names]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(names))
    w = 0.35
    ax.bar(x - w / 2, acc, w, label="Accuracy", color=PALETTE[0])
    ax.bar(x + w / 2, tpr, w, label="TPR (Recall)", color=PALETTE[1])
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("Baseline Performance per Category")
    ax.legend()
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "baseline_per_category.png", dpi=150)
    plt.close()


# ──────────────────────────────────────────────────────────────────
# 2. ASR heatmap: perturbation × category
# ──────────────────────────────────────────────────────────────────

def plot_asr_heatmap(df: pd.DataFrame):
    unsafe_df = df[df["expected_label"] == "unsafe"]
    pivot = unsafe_df.pivot_table(
        index="perturbation",
        columns="category",
        values="predicted_label",
        aggfunc=lambda x: (x == "safe").mean(),
    )
    if pivot.empty:
        return

    fig, ax = plt.subplots(figsize=(12, max(8, len(pivot) * 0.45)))
    sns.heatmap(
        pivot, annot=True, fmt=".2f", cmap="YlOrRd", vmin=0, vmax=1,
        linewidths=0.5, ax=ax, cbar_kws={"label": "ASR (Attack Success Rate)"}
    )
    ax.set_title("Attack Success Rate: Perturbation × Category")
    ax.set_ylabel("Perturbation")
    ax.set_xlabel("Attack Category")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "asr_heatmap.png", dpi=150)
    plt.close()


# ──────────────────────────────────────────────────────────────────
# 3. ASR bar chart sorted by effectiveness
# ──────────────────────────────────────────────────────────────────

def plot_asr_by_perturbation(analysis: dict):
    perts = analysis["perturbation_asr"]
    names = sorted(perts.keys(), key=lambda k: perts[k]["asr"], reverse=True)
    asr_vals = [perts[n]["asr"] for n in names]
    ci_low = [perts[n].get("asr_ci_lower", 0) for n in names]
    ci_high = [perts[n].get("asr_ci_upper", 0) for n in names]
    errors = [[a - l for a, l in zip(asr_vals, ci_low)],
              [h - a for a, h in zip(asr_vals, ci_high)]]

    fig, ax = plt.subplots(figsize=(10, max(6, len(names) * 0.4)))
    y = np.arange(len(names))
    colors = ["#e74c3c" if v > 0.15 else "#f39c12" if v > 0.05 else "#27ae60"
              for v in asr_vals]
    ax.barh(y, asr_vals, xerr=errors, color=colors, edgecolor="white", capsize=3)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlabel("ASR (Attack Success Rate)")
    ax.set_title("Perturbation Effectiveness (ASR with 95% CI)")
    ax.set_xlim(0, max(asr_vals) * 1.3 + 0.05)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "asr_by_perturbation.png", dpi=150)
    plt.close()


# ──────────────────────────────────────────────────────────────────
# 4. Language comparison
# ──────────────────────────────────────────────────────────────────

def plot_language_comparison(df: pd.DataFrame):
    baseline = df[df["perturbation"] == "none"]
    lang_cat = baseline.groupby(["language", "category"]).apply(
        lambda x: (x["expected_label"] == x["predicted_label"]).mean()
    ).unstack(fill_value=0)

    if lang_cat.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    lang_cat.T.plot(kind="bar", ax=ax, color=[PALETTE[0], PALETTE[1]])
    ax.set_ylabel("Accuracy")
    ax.set_title("Baseline Accuracy: English vs. Russian per Category")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.set_ylim(0, 1.1)
    ax.legend(title="Language")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "language_comparison.png", dpi=150)
    plt.close()


# ──────────────────────────────────────────────────────────────────
# 5. Perturbation impact delta chart
# ──────────────────────────────────────────────────────────────────

def plot_perturbation_delta(analysis: dict):
    baseline_acc = analysis["baseline_overall"]["accuracy"]
    perts = analysis["perturbation_asr"]
    names = [k for k in perts if k != "none"]
    deltas = [perts[n]["accuracy"] - baseline_acc for n in names]

    order = np.argsort(deltas)
    names = [names[i] for i in order]
    deltas = [deltas[i] for i in order]

    fig, ax = plt.subplots(figsize=(10, max(6, len(names) * 0.4)))
    colors = ["#e74c3c" if d < -0.05 else "#f39c12" if d < 0 else "#27ae60" for d in deltas]
    ax.barh(range(len(names)), deltas, color=colors, edgecolor="white")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel("Δ Accuracy (vs. Baseline)")
    ax.set_title("Impact of Perturbations on Model Accuracy")
    ax.axvline(x=0, color="black", linewidth=0.8)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "perturbation_delta.png", dpi=150)
    plt.close()


# ──────────────────────────────────────────────────────────────────
# 6. Multi-task analysis plots
# ──────────────────────────────────────────────────────────────────

def plot_adversarial_tags(analysis: dict):
    tags = analysis.get("adversarial_tag_distribution", {})
    if not tags:
        return
    names = sorted(tags.keys(), key=lambda k: tags[k], reverse=True)
    counts = [tags[n] for n in names]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(range(len(names)), counts, color=PALETTE[3])
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel("Count")
    ax.set_title("Adversarial Tag Distribution (Baseline)")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "adversarial_tags.png", dpi=150)
    plt.close()


def plot_intent_tone(analysis: dict):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Intent
    intent = analysis.get("intent_distribution", {})
    if intent:
        names = list(intent.keys())
        vals = list(intent.values())
        axes[0].barh(names, vals, color=PALETTE[0])
        axes[0].set_xlabel("Count")
        axes[0].set_title("Intent Distribution (Baseline)")
        axes[0].invert_yaxis()

    # Tone
    tone = analysis.get("tone_distribution", {})
    if tone:
        names = list(tone.keys())
        vals = list(tone.values())
        axes[1].barh(names, vals, color=PALETTE[1])
        axes[1].set_xlabel("Count")
        axes[1].set_title("Tone of Voice Distribution (Baseline)")
        axes[1].invert_yaxis()

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "intent_tone.png", dpi=150)
    plt.close()


def generate_all_plots():
    print("Loading data...")
    df, analysis = load_data()

    print("Generating plots...")
    plot_baseline_per_category(analysis)
    print("  + baseline_per_category.png")

    plot_asr_heatmap(df)
    print("  + asr_heatmap.png")

    plot_asr_by_perturbation(analysis)
    print("  + asr_by_perturbation.png")

    plot_language_comparison(df)
    print("  + language_comparison.png")

    plot_perturbation_delta(analysis)
    print("  + perturbation_delta.png")

    plot_adversarial_tags(analysis)
    print("  + adversarial_tags.png")

    plot_intent_tone(analysis)
    print("  + intent_tone.png")

    print(f"\nAll figures saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    generate_all_plots()
