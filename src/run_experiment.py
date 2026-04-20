"""
Main experiment runner: HiveTrace Guard (ModernBERT) adversarial robustness evaluation.

Methodology:
1. Baseline evaluation — test model on clean (unperturbed) prompts
2. Adversarial evaluation — apply each perturbation technique and re-test
3. Compute metrics: TPR, FPR, Accuracy, F1, ASR (Attack Success Rate)
4. Statistical analysis — confidence intervals, per-category breakdown
5. Export results to CSV + JSON analysis
"""

import os
import sys
import json
import random
import datetime
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.classifier import HiveTraceGuardClassifier
from src.dataset import build_dataset
from src.perturbations import PERTURBATIONS

warnings.filterwarnings("ignore")

RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

HF_TOKEN = os.environ.get("HF_TOKEN", "")


def compute_metrics(df: pd.DataFrame) -> dict:
    """Compute classification metrics from a results DataFrame."""
    tp = ((df["expected_label"] == "unsafe") & (df["predicted_label"] == "unsafe")).sum()
    tn = ((df["expected_label"] == "safe") & (df["predicted_label"] == "safe")).sum()
    fp = ((df["expected_label"] == "safe") & (df["predicted_label"] == "unsafe")).sum()
    fn = ((df["expected_label"] == "unsafe") & (df["predicted_label"] == "safe")).sum()

    total = len(df)
    accuracy = (tp + tn) / total if total > 0 else 0
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * precision * tpr / (precision + tpr) if (precision + tpr) > 0 else 0

    # ASR: fraction of unsafe prompts misclassified as safe
    unsafe_total = (df["expected_label"] == "unsafe").sum()
    asr = fn / unsafe_total if unsafe_total > 0 else 0

    return {
        "accuracy": round(accuracy, 4),
        "tpr": round(tpr, 4),
        "fpr": round(fpr, 4),
        "precision": round(precision, 4),
        "f1": round(f1, 4),
        "asr": round(asr, 4),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        "total": int(total),
    }


def wilson_ci(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score confidence interval for a proportion."""
    if n == 0:
        return (0.0, 0.0)
    denom = 1 + z**2 / n
    centre = p + z**2 / (2 * n)
    spread = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n)
    lower = max(0, (centre - spread) / denom)
    upper = min(1, (centre + spread) / denom)
    return (round(lower, 4), round(upper, 4))


def run_experiment(classifier: HiveTraceGuardClassifier) -> pd.DataFrame:
    """Run the full experiment: baseline + all perturbations."""
    dataset = build_dataset()
    all_results = []

    # ── Phase 1: Baseline evaluation ──
    print("\n" + "=" * 70)
    print("PHASE 1: BASELINE EVALUATION (no perturbations)")
    print("=" * 70)

    for item in tqdm(dataset, desc="Baseline"):
        result = classifier.classify(item["text"])
        all_results.append({
            **item,
            "perturbed_text": item["text"],
            "predicted_label": result["label"],
            "confidence": result["score"],
        })

    # ── Phase 2: Adversarial perturbations ──
    print("\n" + "=" * 70)
    print("PHASE 2: ADVERSARIAL PERTURBATION EVALUATION")
    print("=" * 70)

    text_perturbations = [
        "homoglyph", "zero_width", "leetspeak", "char_swap",
        "char_delete", "char_noise", "extra_spaces", "dot_split",
        "markdown_wrap", "html_comment", "base64_hint",
        "academic_framing", "roleplay_framing", "negation_framing",
    ]
    ru_only_perturbations = ["reverse_homoglyph", "transliteration", "code_switch"]

    for pert_name in text_perturbations + ru_only_perturbations:
        pert_fn = PERTURBATIONS[pert_name]
        applicable_items = dataset

        if pert_name in ru_only_perturbations:
            applicable_items = [d for d in dataset if d["language"] == "ru"]
        if pert_name == "homoglyph":
            applicable_items = [d for d in dataset if d["language"] == "en"]

        if not applicable_items:
            continue

        print(f"\n  Perturbation: {pert_name} ({len(applicable_items)} samples)")

        for item in tqdm(applicable_items, desc=f"  {pert_name}", leave=False):
            try:
                perturbed = pert_fn(item["text"])
            except Exception:
                perturbed = item["text"]

            result = classifier.classify(perturbed)
            all_results.append({
                "text": item["text"],
                "expected_label": item["expected_label"],
                "category": item["category"],
                "language": item["language"],
                "perturbation": pert_name,
                "perturbed_text": perturbed,
                "predicted_label": result["label"],
                "confidence": result["score"],
            })

    return pd.DataFrame(all_results)


def analyze_results(df: pd.DataFrame) -> dict:
    """Perform full analysis and return structured results."""
    analysis = {}

    # 1. Overall baseline metrics
    baseline = df[df["perturbation"] == "none"]
    analysis["baseline_overall"] = compute_metrics(baseline)

    # 2. Baseline per category
    analysis["baseline_per_category"] = {}
    for cat in sorted(baseline["category"].unique()):
        cat_df = baseline[baseline["category"] == cat]
        analysis["baseline_per_category"][cat] = compute_metrics(cat_df)

    # 3. Baseline per language
    analysis["baseline_per_language"] = {}
    for lang in sorted(baseline["language"].unique()):
        lang_df = baseline[baseline["language"] == lang]
        analysis["baseline_per_language"][lang] = compute_metrics(lang_df)

    # 4. Per-perturbation metrics
    analysis["perturbation_asr"] = {}
    for pert in df["perturbation"].unique():
        pert_df = df[df["perturbation"] == pert]
        metrics = compute_metrics(pert_df)
        unsafe_df = pert_df[pert_df["expected_label"] == "unsafe"]
        n = len(unsafe_df)
        ci = wilson_ci(metrics["asr"], n)
        metrics["asr_ci_lower"] = ci[0]
        metrics["asr_ci_upper"] = ci[1]
        metrics["n_samples"] = int(n)
        analysis["perturbation_asr"][pert] = metrics

    # 5. Per-perturbation × per-category breakdown
    analysis["perturbation_category"] = {}
    for pert in df["perturbation"].unique():
        analysis["perturbation_category"][pert] = {}
        for cat in df["category"].unique():
            sub = df[(df["perturbation"] == pert) & (df["category"] == cat)]
            if len(sub) > 0:
                analysis["perturbation_category"][pert][cat] = compute_metrics(sub)

    # 6. Confidence score statistics
    baseline = df[df["perturbation"] == "none"]
    conf_stats = {}
    for label in ["safe", "unsafe"]:
        sub = baseline[baseline["expected_label"] == label]
        correct = sub[sub["predicted_label"] == sub["expected_label"]]
        wrong = sub[sub["predicted_label"] != sub["expected_label"]]
        conf_stats[label] = {
            "correct_mean_conf": round(correct["confidence"].mean(), 4) if len(correct) > 0 else 0,
            "correct_std_conf": round(correct["confidence"].std(), 4) if len(correct) > 0 else 0,
            "wrong_mean_conf": round(wrong["confidence"].mean(), 4) if len(wrong) > 0 else 0,
            "wrong_std_conf": round(wrong["confidence"].std(), 4) if len(wrong) > 0 else 0,
        }
    analysis["confidence_stats"] = conf_stats

    return analysis


def print_summary(analysis: dict):
    """Print a human-readable summary of results."""
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    bl = analysis["baseline_overall"]
    print(f"\n{'BASELINE METRICS':^70}")
    print(f"  Accuracy: {bl['accuracy']:.4f}  |  TPR: {bl['tpr']:.4f}  |  "
          f"FPR: {bl['fpr']:.4f}  |  F1: {bl['f1']:.4f}  |  ASR: {bl['asr']:.4f}")

    print(f"\n{'BASELINE PER CATEGORY':^70}")
    print(f"  {'Category':<25} {'Accuracy':>8} {'TPR':>8} {'FPR':>8} {'ASR':>8} {'N':>6}")
    print("  " + "-" * 60)
    for cat, m in analysis["baseline_per_category"].items():
        print(f"  {cat:<25} {m['accuracy']:>8.4f} {m['tpr']:>8.4f} {m['fpr']:>8.4f} "
              f"{m['asr']:>8.4f} {m['total']:>6}")

    print(f"\n{'BASELINE PER LANGUAGE':^70}")
    for lang, m in analysis["baseline_per_language"].items():
        print(f"  {lang}: Accuracy={m['accuracy']:.4f}  TPR={m['tpr']:.4f}  "
              f"FPR={m['fpr']:.4f}  ASR={m['asr']:.4f}")

    print(f"\n{'CONFIDENCE SCORES':^70}")
    for label, stats in analysis["confidence_stats"].items():
        print(f"  {label}: correct={stats['correct_mean_conf']:.3f}±{stats['correct_std_conf']:.3f}  "
              f"wrong={stats['wrong_mean_conf']:.3f}±{stats['wrong_std_conf']:.3f}")

    print(f"\n{'PERTURBATION IMPACT (ASR on unsafe prompts)':^70}")
    print(f"  {'Perturbation':<22} {'ASR':>7} {'95% CI':>18} {'Acc':>7} {'TPR':>7} {'N':>6}")
    print("  " + "-" * 68)
    sorted_perts = sorted(
        analysis["perturbation_asr"].items(),
        key=lambda x: x[1]["asr"], reverse=True
    )
    for pert, m in sorted_perts:
        ci = f"[{m.get('asr_ci_lower', 0):.3f}, {m.get('asr_ci_upper', 0):.3f}]"
        print(f"  {pert:<22} {m['asr']:>7.4f} {m['accuracy']:>7.4f} "
              f"{m['tpr']:>7.4f} {m.get('n_samples', m['total']):>6}")


def main():
    print("=" * 70)
    print("HiveTrace Guard (ModernBERT) — Adversarial Robustness Evaluation")
    print(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Model: hivetrace/hivetrace-guard-base-2025-10-23")
    print("=" * 70)

    # Initialize classifier
    print("\nLoading model...")
    classifier = HiveTraceGuardClassifier(token=HF_TOKEN)
    print("Model loaded.")

    # Run experiment
    df = run_experiment(classifier)

    # Save raw results
    csv_path = RESULTS_DIR / "raw_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nRaw results saved to {csv_path}")

    # Analyze
    analysis = analyze_results(df)

    # Save analysis JSON
    json_path = RESULTS_DIR / "analysis.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)
    print(f"Analysis saved to {json_path}")

    # Print summary
    print_summary(analysis)

    return df, analysis


if __name__ == "__main__":
    main()
