"""
Correlate per-cluster quality measures with per-cluster labeling scores.

Loads:
  results/{dataset}_cluster_quality.json   (from cluster_quality.py)
  results/{dataset}_results.json           (from run_pipeline.py)

For each dataset that has both files, joins on cluster label and computes
Pearson r and Spearman rho between every quality measure and every labeling metric.

Outputs:
  analysis/correlation_table.csv     – full correlation results
  analysis/correlation_heatmap.png   – heatmap of Spearman rho across datasets
  analysis/scatter_{dataset}.png     – scatter grid per dataset

Usage:
    python cluster_analysis/correlation_analysis.py
    python cluster_analysis/correlation_analysis.py --results-dir results --output-dir analysis
"""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DATASETS = ["bbc_news", "banking77", "20newsgroups", "ag_news"]

QUALITY_MEASURES = ["voorhees_nn", "overlap", "kl_divergence", "density", "nmrd"]
LABELING_METRICS = ["sem_sim", "tok_f1"]

# kl_divergence is inverted (lower = better), negate for intuitive correlation direction
NEGATE = {"kl_divergence"}

MEASURE_LABELS = {
    "voorhees_nn":    "Voorhees NN↑",
    "overlap":        "Overlap↑",
    "kl_divergence":  "KL div↓ (neg.)",
    "density":        "Density↑",
    "nmrd":           "nMRD↑",
}
METRIC_LABELS = {
    "sem_sim": "Semantic Similarity",
    "tok_f1":  "Token F1",
}

sns.set_theme(style="whitegrid", font_scale=1.05)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_quality(results_dir: str, dataset: str) -> pd.DataFrame | None:
    path = Path(results_dir) / f"{dataset}_cluster_quality.json"
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    rows = [{"cluster": label, **scores} for label, scores in data.items()]
    df = pd.DataFrame(rows)
    # Negate KL divergence so all measures are "higher = better"
    if "kl_divergence" in df.columns:
        df["kl_divergence"] = -df["kl_divergence"]
    return df


def load_scores(results_dir: str, dataset: str) -> pd.DataFrame | None:
    path = Path(results_dir) / f"{dataset}_results.json"
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    metrics = data["test_result"]["metrics"]
    sem = {k: v for k, v in metrics["semantic_similarity"].items() if k != "__mean__"}
    tok = {k: v for k, v in metrics["token_overlap_f1"].items() if k != "__mean__"}
    clusters = sorted(sem.keys())
    rows = [
        {"cluster": c, "sem_sim": sem[c], "tok_f1": tok.get(c)}
        for c in clusters
    ]
    return pd.DataFrame(rows)


def load_dataset(results_dir: str, dataset: str) -> pd.DataFrame | None:
    quality = load_quality(results_dir, dataset)
    scores = load_scores(results_dir, dataset)
    if quality is None or scores is None:
        return None
    merged = quality.merge(scores, on="cluster", how="inner")
    if merged.empty:
        return None
    merged.insert(0, "dataset", dataset)
    return merged


# ---------------------------------------------------------------------------
# Correlation
# ---------------------------------------------------------------------------

def compute_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (quality_measure, labeling_metric) pair, compute Pearson r and
    Spearman rho (with p-values) on the rows with no missing values.
    """
    rows = []
    for measure in QUALITY_MEASURES:
        if measure not in df.columns:
            continue
        for metric in LABELING_METRICS:
            if metric not in df.columns:
                continue
            sub = df[[measure, metric]].dropna()
            n = len(sub)
            if n < 3:
                continue
            x, y = sub[measure].values, sub[metric].values
            if x.std() == 0 or y.std() == 0:
                continue  # correlation undefined for constant input
            pr, pp = stats.pearsonr(x, y)
            sr, sp = stats.spearmanr(x, y)
            rows.append({
                "dataset":  df["dataset"].iloc[0],
                "measure":  measure,
                "metric":   metric,
                "n":        n,
                "pearson_r":  round(pr, 3),
                "pearson_p":  round(pp, 4),
                "spearman_r": round(sr, 3),
                "spearman_p": round(sp, 4),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_scatter(df: pd.DataFrame, dataset: str, output_dir: str) -> None:
    """Scatter grid: rows = quality measures, cols = labeling metrics."""
    measures = [m for m in QUALITY_MEASURES if m in df.columns]
    metrics  = [m for m in LABELING_METRICS if m in df.columns]
    n_rows, n_cols = len(measures), len(metrics)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5 * n_cols, 4 * n_rows),
        squeeze=False,
    )
    fig.suptitle(f"Quality measures vs. labeling scores — {dataset}", fontsize=14, y=1.01)

    for r, measure in enumerate(measures):
        for c, metric in enumerate(metrics):
            ax = axes[r][c]
            sub = df[[measure, metric, "cluster"]].dropna()
            if sub.empty:
                ax.set_visible(False)
                continue

            x, y = sub[measure].values, sub[metric].values
            ax.scatter(x, y, s=60, alpha=0.8, color=sns.color_palette("tab10")[c])

            # Annotate cluster names
            for _, row in sub.iterrows():
                ax.annotate(
                    row["cluster"],
                    xy=(row[measure], row[metric]),
                    xytext=(4, 4), textcoords="offset points",
                    fontsize=6, alpha=0.8,
                )

            # Regression line + correlation annotation
            if len(x) >= 3 and x.std() > 0 and y.std() > 0:
                m_fit, b_fit = np.polyfit(x, y, 1)
                x_line = np.linspace(x.min(), x.max(), 100)
                ax.plot(x_line, m_fit * x_line + b_fit, "k--", linewidth=1, alpha=0.5)

                pr, pp = stats.pearsonr(x, y)
                sr, sp = stats.spearmanr(x, y)
                ax.set_title(
                    f"r={pr:+.2f} (p={pp:.3f})  ρ={sr:+.2f} (p={sp:.3f})",
                    fontsize=9,
                )
            elif x.std() == 0 or y.std() == 0:
                ax.set_title("(constant input — correlation undefined)", fontsize=8)

            ax.set_xlabel(
                MEASURE_LABELS.get(measure, measure)
                + (" [negated]" if measure in NEGATE else ""),
                fontsize=9,
            )
            ax.set_ylabel(METRIC_LABELS.get(metric, metric), fontsize=9)

    fig.tight_layout()
    path = Path(output_dir) / f"scatter_{dataset}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_correlation_heatmap(corr_df: pd.DataFrame, output_dir: str) -> None:
    """
    One heatmap per labeling metric showing Spearman rho for each
    (dataset × quality_measure) cell.
    """
    for metric in LABELING_METRICS:
        sub = corr_df[corr_df["metric"] == metric]
        if sub.empty:
            continue

        pivot = sub.pivot(index="dataset", columns="measure", values="spearman_r")
        # Reorder columns to canonical order
        pivot = pivot.reindex(
            columns=[m for m in QUALITY_MEASURES if m in pivot.columns]
        )
        pivot.columns = [MEASURE_LABELS.get(c, c) for c in pivot.columns]

        fig, ax = plt.subplots(figsize=(max(6, len(pivot.columns) * 1.5), max(3, len(pivot) * 1.0)))
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".2f",
            cmap="RdYlGn",
            vmin=-1, vmax=1,
            center=0,
            linewidths=0.5,
            ax=ax,
        )
        ax.set_title(
            f"Spearman ρ: quality measure vs. {METRIC_LABELS[metric]}",
            fontsize=12,
        )
        ax.set_xlabel("")
        ax.set_ylabel("")
        fig.tight_layout()
        safe_metric = metric.replace(" ", "_")
        path = Path(output_dir) / f"correlation_heatmap_{safe_metric}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_table(corr_df: pd.DataFrame) -> None:
    print(f"\n{'='*90}")
    print("Correlation: cluster quality measures vs. labeling scores (test split, per cluster)")
    print(f"{'='*90}")
    print("Note: kl_divergence is negated so all measures are 'higher = better'.")
    print(f"{'='*90}")
    header = f"{'Dataset':<16} {'Measure':<18} {'Metric':<12} {'n':>4}  "
    header += f"{'Pearson r':>10}  {'p':>7}  {'Spearman ρ':>11}  {'p':>7}"
    print(header)
    print("-" * 90)

    prev_ds = None
    for _, row in corr_df.sort_values(["dataset", "metric", "measure"]).iterrows():
        ds = row["dataset"] if row["dataset"] != prev_ds else ""
        prev_ds = row["dataset"]
        sig_p = " *" if row["pearson_p"]  < 0.05 else "  "
        sig_s = " *" if row["spearman_p"] < 0.05 else "  "
        print(
            f"{ds:<16} {row['measure']:<18} {row['metric']:<12} {int(row['n']):>4}  "
            f"{row['pearson_r']:>+10.3f}{sig_p}  {row['pearson_p']:>7.4f}  "
            f"{row['spearman_r']:>+11.3f}{sig_s}  {row['spearman_p']:>7.4f}"
        )

    print("-" * 90)
    print("* p < 0.05")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Correlate cluster quality measures with labeling scores."
    )
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output-dir", default="analysis")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    all_data: list[pd.DataFrame] = []
    all_corr: list[pd.DataFrame] = []

    print(f"\n=== Cluster Quality ↔ Labeling Score Correlation ===\n")

    for dataset in DATASETS:
        df = load_dataset(args.results_dir, dataset)
        if df is None:
            print(f"  [{dataset}] skipped — missing quality or results file")
            continue

        print(f"  [{dataset}] {len(df)} clusters loaded")
        all_data.append(df)

        corr = compute_correlations(df)
        all_corr.append(corr)

        plot_scatter(df, dataset, args.output_dir)

    if not all_corr:
        print("\nNo datasets with both quality and results files found.")
        return

    corr_df = pd.concat(all_corr, ignore_index=True)

    _print_table(corr_df)

    # Save CSV
    csv_path = Path(args.output_dir) / "correlation_table.csv"
    corr_df.to_csv(csv_path, index=False)
    print(f"\nSaved {csv_path}")

    plot_correlation_heatmap(corr_df, args.output_dir)


if __name__ == "__main__":
    main()
