"""
Analyze hyperparameter sweep results.

Usage:
    python analyze_results.py
    python analyze_results.py --results-dir results --output-dir analysis
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

# ── config ────────────────────────────────────────────────────────────────────
DATASETS = ["bbc_news", "banking77", "20newsgroups", "ag_news"]
METRICS = ["semantic_similarity", "token_overlap_f1"]
PALETTE = sns.color_palette("tab10")
sns.set_theme(style="whitegrid", font_scale=1.05)


# ── loaders ───────────────────────────────────────────────────────────────────

def load_sweep(results_dir: str, dataset: str) -> pd.DataFrame:
    """Load a dataset's sweep JSON into a flat DataFrame (one row per run)."""
    path = Path(results_dir) / f"{dataset}_sweep.json"
    with open(path) as f:
        runs = json.load(f)

    rows = []
    for run in runs:
        cfg = run["config"]
        sem = run["metrics"]["semantic_similarity"]
        tok = run["metrics"]["token_overlap_f1"]
        row = {
            "dataset": dataset,
            "threshold": cfg["similarity_threshold"],
            "top_k": cfg["top_k"],
            "alpha": cfg["pagerank_alpha"],
            "sem_sim": sem["__mean__"],
            "tok_f1": tok["__mean__"],
        }
        # per-cluster scores
        for k, v in sem.items():
            if k != "__mean__":
                row[f"sem_{k}"] = v
        rows.append(row)

    return pd.DataFrame(rows)


def load_results(results_dir: str, dataset: str) -> dict:
    path = Path(results_dir) / f"{dataset}_results.json"
    with open(path) as f:
        return json.load(f)


# ── helpers ───────────────────────────────────────────────────────────────────

def savefig(fig, output_dir: str, name: str):
    os.makedirs(output_dir, exist_ok=True)
    path = Path(output_dir) / name
    fig.savefig(path, bbox_inches="tight", dpi=150)
    print(f"  Saved: {path}")
    plt.close(fig)


# ── plots ─────────────────────────────────────────────────────────────────────

def plot_test_scores(all_results: dict, output_dir: str):
    """Bar chart: test semantic similarity and token-F1 per dataset."""
    rows = []
    for ds, r in all_results.items():
        tm = r["test_result"]["metrics"]
        rows.append({
            "dataset": ds,
            "Semantic Similarity": tm["semantic_similarity"]["__mean__"],
            "Token Overlap F1": tm["token_overlap_f1"]["__mean__"],
        })
    df = pd.DataFrame(rows).set_index("dataset")

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(df))
    w = 0.35
    bars1 = ax.bar(x - w / 2, df["Semantic Similarity"], w, label="Semantic Similarity", color=PALETTE[0])
    bars2 = ax.bar(x + w / 2, df["Token Overlap F1"], w, label="Token Overlap F1", color=PALETTE[1])
    ax.set_xticks(x)
    ax.set_xticklabels(df.index, rotation=15, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Test Set Performance by Dataset (Best Hyperparameters)")
    ax.set_ylim(0, 1)
    ax.legend()

    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f"{h:.3f}",
                ha="center", va="bottom", fontsize=8)

    savefig(fig, output_dir, "test_scores.png")


def plot_sweep_distributions(dfs: dict[str, pd.DataFrame], output_dir: str):
    """Box plots: distribution of dev semantic similarity across all sweep runs."""
    fig, ax = plt.subplots(figsize=(8, 4))
    data = [df["sem_sim"].values for df in dfs.values()]
    bp = ax.boxplot(data, tick_labels=list(dfs.keys()), patch_artist=True, notch=False)
    for patch, color in zip(bp["boxes"], PALETTE):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel("Dev Semantic Similarity")
    ax.set_title("Sweep Score Distribution (all 27 runs per dataset)")
    ax.set_xticklabels(list(dfs.keys()), rotation=15, ha="right")
    savefig(fig, output_dir, "sweep_distributions.png")


def plot_hyperparam_effects(dfs: dict[str, pd.DataFrame], output_dir: str):
    """
    Three subplots: mean dev sem-sim vs each hyperparameter,
    one line per dataset.
    """
    params = [
        ("threshold", "Similarity Threshold"),
        ("top_k", "Top-k Documents"),
        ("alpha", "PageRank Alpha"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=False)

    for ax, (param, label) in zip(axes, params):
        for (ds, df), color in zip(dfs.items(), PALETTE):
            grouped = df.groupby(param)["sem_sim"].mean()
            ax.plot(grouped.index, grouped.values, marker="o", label=ds, color=color)
        ax.set_xlabel(label)
        ax.set_ylabel("Mean Dev Semantic Similarity")
        ax.set_title(f"Effect of {label}")
        ax.legend(fontsize=7)

    fig.suptitle("Hyperparameter Sensitivity (dev set, averaged over other params)", y=1.02)
    fig.tight_layout()
    savefig(fig, output_dir, "hyperparam_effects.png")


def plot_combined_heatmap(all_results: dict, output_dir: str):
    """
    Single PNG with all 4 dataset heatmaps stacked vertically.

    Small datasets (4-5 clusters) get thin rows with large readable cells.
    Large datasets (20, 77 clusters) get taller rows to accommodate
    rotated labels. Clusters are sorted by score descending so the
    colour gradient is immediately readable.
    """
    ds_order = ["bbc_news", "ag_news", "20newsgroups", "banking77"]

    # Height ratios: heatmap strip height only (label space is outside axes).
    # Give larger datasets more strip height so per-cell text isn't squashed.
    height_ratios = [1, 1, 1.5, 2]

    fig, axes = plt.subplots(
        4, 1, figsize=(22, 24),
        gridspec_kw={"height_ratios": height_ratios},
    )
    fig.subplots_adjust(hspace=1.8)

    im_ref = None
    for ax, ds, in zip(axes, ds_order):
        sem = {
            k: v
            for k, v in all_results[ds]["test_result"]["metrics"]["semantic_similarity"].items()
            if k != "__mean__"
        }
        n = len(sem)

        # Sort clusters by score descending for a clean colour gradient
        labels, scores = zip(*sorted(sem.items(), key=lambda x: x[1], reverse=True))
        scores = np.array(scores)

        im_ref = ax.imshow(scores.reshape(1, -1), cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(n))
        ax.set_yticks([])

        # Label font size and rotation scale with cluster count
        if n <= 5:
            fs_label, rot = 9, 40
        elif n <= 20:
            fs_label, rot = 7, 60
        else:
            fs_label, rot = 5, 90

        ax.set_xticklabels(labels, rotation=rot, ha="right", fontsize=fs_label)
        ax.tick_params(axis="x", pad=2)

        mean = all_results[ds]["test_result"]["metrics"]["semantic_similarity"]["__mean__"]
        ax.set_title(
            f"{ds}  —  {n} clusters  —  mean = {mean:.3f}",
            fontsize=10, fontweight="bold", loc="left", pad=5,
        )

        # Annotate score values inside cells (skip for banking77: too crowded)
        if n <= 20:
            fs_ann = 8 if n <= 5 else 6
            for j, v in enumerate(scores):
                txt_color = "black" if 0.25 < v < 0.82 else "white"
                ax.text(j, 0, f"{v:.2f}", ha="center", va="center",
                        fontsize=fs_ann, color=txt_color)

    # Single shared colourbar on the right
    fig.colorbar(im_ref, ax=axes.tolist(), fraction=0.008, pad=0.01,
                 label="Semantic Similarity")
    fig.suptitle("Test Semantic Similarity per Cluster — All Datasets",
                 fontsize=13, fontweight="bold", y=0.99)

    savefig(fig, output_dir, "combined_cluster_heatmap.png")


def plot_per_cluster_heatmap(all_results: dict, output_dir: str):
    """
    Heatmap of test semantic similarity per cluster, for each dataset separately
    (datasets have different cluster sets so we can't merge them).
    """
    for ds, r in all_results.items():
        sem = r["test_result"]["metrics"]["semantic_similarity"]
        clusters = {k: v for k, v in sem.items() if k != "__mean__"}
        if not clusters:
            continue

        labels = list(clusters.keys())
        scores = [clusters[l] for l in labels]

        fig, ax = plt.subplots(figsize=(max(4, len(labels) * 0.5 + 2), 2.5))
        data = np.array(scores).reshape(1, -1)
        im = ax.imshow(data, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticks([])
        ax.set_title(f"{ds} — Test Semantic Similarity per Cluster (mean={sem['__mean__']:.3f})")
        plt.colorbar(im, ax=ax, fraction=0.02, pad=0.04)
        for j, v in enumerate(scores):
            ax.text(j, 0, f"{v:.2f}", ha="center", va="center", fontsize=7,
                    color="black" if 0.3 < v < 0.8 else "white")
        fig.tight_layout()
        savefig(fig, output_dir, f"cluster_heatmap_{ds}.png")


def plot_dev_vs_test(all_results: dict, output_dir: str):
    """Scatter: best dev score vs test score per dataset."""
    # Per-point offsets (points, direction) to avoid overlapping labels.
    # bbc_news and ag_news are very close; push them apart manually.
    label_offsets = {
        "bbc_news":      (-68, 6),
        "banking77":     (6,   2),
        "20newsgroups":  (6,   2),
        "ag_news":       (6,  -13),
    }

    fig, ax = plt.subplots(figsize=(5, 4))
    for (ds, r), color in zip(all_results.items(), PALETTE):
        dev = r["best_dev_metrics"]["semantic_similarity"]["__mean__"]
        test = r["test_result"]["metrics"]["semantic_similarity"]["__mean__"]
        ax.scatter(dev, test, color=color, s=100, zorder=3, label=ds)

    lim_min = min(ax.get_xlim()[0], ax.get_ylim()[0])
    lim_max = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([lim_min, lim_max], [lim_min, lim_max], "k--", alpha=0.4, label="dev = test")
    ax.set_xlabel("Best Dev Semantic Similarity")
    ax.set_ylabel("Test Semantic Similarity")
    ax.set_title("Dev vs Test Generalisation")
    ax.legend(fontsize=7)
    savefig(fig, output_dir, "dev_vs_test.png")
    


def plot_param_variance(dfs: dict[str, pd.DataFrame], output_dir: str):
    """Bar chart: which hyperparameter explains most variance in dev score per dataset."""
    params = ["threshold", "top_k", "alpha"]
    records = []
    for ds, df in dfs.items():
        total_var = df["sem_sim"].var()
        if total_var == 0:
            continue
        for p in params:
            group_means = df.groupby(p)["sem_sim"].mean()
            # variance of group means (between-group variance)
            between_var = group_means.var()
            records.append({"dataset": ds, "param": p, "variance_ratio": between_var / total_var})

    vdf = pd.DataFrame(records)
    fig, ax = plt.subplots(figsize=(8, 4))
    pivot = vdf.pivot(index="dataset", columns="param", values="variance_ratio")
    pivot.plot(kind="bar", ax=ax, colormap="tab10", width=0.6)
    ax.set_ylabel("Between-group Variance / Total Variance")
    ax.set_title("Which Hyperparameter Drives Performance? (higher = more impact)")
    ax.set_xticklabels(pivot.index, rotation=15, ha="right")
    ax.legend(title="Param")
    savefig(fig, output_dir, "param_variance.png")


# ── label comparison ──────────────────────────────────────────────────────────

def print_label_comparison(all_results: dict):
    """
    For each dataset print the selected hyperparameters and a side-by-side
    table of ground-truth cluster name | dev generated label | test generated label.
    """
    sep = "=" * 60
    for ds, r in all_results.items():
        bc = r["best_config"]

        # Find the dev run that matches the best config
        best_dev_run = next(
            run for run in r["dev_sweep"]
            if run["config"]["similarity_threshold"] == bc["similarity_threshold"]
            and run["config"]["top_k"] == bc["top_k"]
            and run["config"]["pagerank_alpha"] == bc["pagerank_alpha"]
        )
        dev_labels = best_dev_run["labels"]
        test_labels = r["test_result"]["labels"]

        print(f"\n{sep}")
        print(f"  {ds}")
        print(f"  Best config: threshold={bc['similarity_threshold']}  "
              f"top_k={bc['top_k']}  alpha={bc['pagerank_alpha']}")
        print(sep)

        clusters = sorted(test_labels.keys())
        rows = []
        for cluster in clusters:
            rows.append({
                "Ground Truth": cluster,
                "Dev Label": dev_labels.get(cluster, {}).get("generated", "—"),
                "Test Label": test_labels.get(cluster, {}).get("generated", "—"),
            })

        df = pd.DataFrame(rows)
        print(df.to_string(index=False))
        print()


# ── sweep tables ──────────────────────────────────────────────────────────────

def print_sweep_tables(dfs: dict[str, pd.DataFrame]):
    """Print all sweep runs as a table for each dataset."""
    sep = "=" * 60
    for ds, df in dfs.items():
        print(f"\n{sep}")
        print(f"  SWEEP RESULTS: {ds}")
        print(sep)
        display = df[["threshold", "top_k", "alpha", "sem_sim", "tok_f1"]].copy()
        display.columns = ["threshold", "top_k", "alpha", "sem_sim", "tok_f1"]
        display = display.sort_values("sem_sim", ascending=False).reset_index(drop=True)
        display.index += 1  # 1-based rank
        display["sem_sim"] = display["sem_sim"].map("{:.4f}".format)
        display["tok_f1"] = display["tok_f1"].map("{:.4f}".format)
        print(display.to_string())
        print()


# ── insights ──────────────────────────────────────────────────────────────────

def print_insights(dfs: dict[str, pd.DataFrame], all_results: dict):
    sep = "=" * 60

    print(f"\n{sep}")
    print("  RESULTS SUMMARY")
    print(sep)

    rows = []
    for ds, r in all_results.items():
        bc = r["best_config"]
        dev_sim = r["best_dev_metrics"]["semantic_similarity"]["__mean__"]
        dev_f1 = r["best_dev_metrics"]["token_overlap_f1"]["__mean__"]
        test_sim = r["test_result"]["metrics"]["semantic_similarity"]["__mean__"]
        test_f1 = r["test_result"]["metrics"]["token_overlap_f1"]["__mean__"]
        rows.append({
            "Dataset": ds,
            "Best threshold": bc["similarity_threshold"],
            "Best top_k": bc["top_k"],
            "Best alpha": bc["pagerank_alpha"],
            "Dev sim": f"{dev_sim:.3f}",
            "Test sim": f"{test_sim:.3f}",
            "Dev F1": f"{dev_f1:.3f}",
            "Test F1": f"{test_f1:.3f}",
        })
    print(pd.DataFrame(rows).to_string(index=False))

    print(f"\n{sep}")
    print("  INSIGHTS")
    print(sep)

    # 1. Best/worst datasets
    test_sims = {ds: r["test_result"]["metrics"]["semantic_similarity"]["__mean__"]
                 for ds, r in all_results.items()}
    best_ds = max(test_sims, key=test_sims.get)
    worst_ds = min(test_sims, key=test_sims.get)
    print(f"\n[1] Easiest dataset to label: {best_ds} ({test_sims[best_ds]:.3f})")
    print(f"    Hardest dataset to label:  {worst_ds} ({test_sims[worst_ds]:.3f})")
    print(f"    Gap: {test_sims[best_ds] - test_sims[worst_ds]:.3f}")

    # 2. Dev/test gap (generalisation)
    print(f"\n[2] Dev → Test generalisation gap (best dev - test):")
    for ds, r in all_results.items():
        dev = r["best_dev_metrics"]["semantic_similarity"]["__mean__"]
        test = r["test_result"]["metrics"]["semantic_similarity"]["__mean__"]
        direction = "↓ overfit" if dev > test else "↑ underfit"
        print(f"    {ds:20s}: dev={dev:.3f}  test={test:.3f}  gap={dev-test:+.3f}  {direction}")

    # 3. Hyperparameter consistency across datasets
    print(f"\n[3] Best hyperparameters per dataset:")
    param_votes = {"threshold": {}, "top_k": {}, "alpha": {}}
    for ds, r in all_results.items():
        bc = r["best_config"]
        for p, key in [("threshold", "similarity_threshold"), ("top_k", "top_k"), ("alpha", "pagerank_alpha")]:
            v = bc[key]
            param_votes[p][v] = param_votes[p].get(v, 0) + 1
    for p, votes in param_votes.items():
        most_common = max(votes, key=votes.get)
        print(f"    {p:12s}: most common best value = {most_common}  (chosen by {votes[most_common]}/{len(all_results)} datasets)")

    # 4. Sweep range and stability
    print(f"\n[4] Dev score range across 27 sweep runs:")
    for ds, df in dfs.items():
        lo, hi, std = df["sem_sim"].min(), df["sem_sim"].max(), df["sem_sim"].std()
        print(f"    {ds:20s}: min={lo:.3f}  max={hi:.3f}  std={std:.3f}  range={hi-lo:.3f}")

    # 5. Which hyperparameter matters most (by between-group variance)
    print(f"\n[5] Most impactful hyperparameter per dataset (by group-mean variance):")
    params = ["threshold", "top_k", "alpha"]
    for ds, df in dfs.items():
        total_var = df["sem_sim"].var()
        if total_var == 0:
            print(f"    {ds:20s}: no variance in results")
            continue
        variances = {}
        for p in params:
            group_means = df.groupby(p)["sem_sim"].mean()
            variances[p] = group_means.var() / total_var
        top_p = max(variances, key=variances.get)
        print(f"    {ds:20s}: {top_p} (ratio={variances[top_p]:.3f})")

    # 6. Per-cluster hardest/easiest
    print(f"\n[6] Hardest and easiest clusters per dataset (test set):")
    for ds, r in all_results.items():
        sem = {k: v for k, v in r["test_result"]["metrics"]["semantic_similarity"].items()
               if k != "__mean__"}
        if not sem:
            continue
        easiest = max(sem, key=sem.get)
        hardest = min(sem, key=sem.get)
        print(f"    {ds}:")
        print(f"      easiest: {easiest} ({sem[easiest]:.3f})")
        print(f"      hardest: {hardest} ({sem[hardest]:.3f})")

    # 7. Token F1 observations
    all_test_f1 = [r["test_result"]["metrics"]["token_overlap_f1"]["__mean__"]
                   for r in all_results.values()]
    avg_f1 = np.mean(all_test_f1)
    print(f"\n[7] Token Overlap F1 avg across all datasets: {avg_f1:.3f}")
    if avg_f1 < 0.1:
        print("    → Very low F1: LLM paraphrases rather than copying ground-truth tokens.")
        print("      Semantic similarity is the more informative metric for this task.")

    print(f"\n{sep}\n")


# ── dataset statistics ────────────────────────────────────────────────────────

def print_dataset_statistics():
    """Load each dataset and print corpus-level and per-cluster statistics."""
    from run_pipeline import load_dataset

    sep = "=" * 60
    print(f"\n{sep}")
    print("  DATASET STATISTICS")
    print(sep)

    for ds in DATASETS:
        print(f"\nLoading {ds}...")
        df = load_dataset(ds)
        df["n_words"] = df["text"].str.split().str.len()
        df["n_chars"] = df["text"].str.len()

        n_docs = len(df)
        n_clusters = df["label"].nunique()
        balance = df["label"].value_counts()
        imbalance_ratio = balance.max() / balance.min()

        print(f"\n{sep}")
        print(f"  {ds}")
        print(sep)

        # ── overall ──────────────────────────────────────────────────────────
        print(f"\n  {'Total documents:':<30} {n_docs:,}")
        print(f"  {'Number of clusters:':<30} {n_clusters}")
        print(f"  {'Imbalance ratio (max/min):':<30} {imbalance_ratio:.2f}")
        print(f"\n  Document length (words):")
        print(f"    {'Mean:':<10} {df['n_words'].mean():.1f}")
        print(f"    {'Std:':<10} {df['n_words'].std():.1f}")
        print(f"    {'Median:':<10} {df['n_words'].median():.1f}")
        print(f"    {'Min:':<10} {df['n_words'].min()}")
        print(f"    {'Max:':<10} {df['n_words'].max()}")
        print(f"\n  Document length (chars):")
        print(f"    {'Mean:':<10} {df['n_chars'].mean():.1f}")
        print(f"    {'Std:':<10} {df['n_chars'].std():.1f}")
        print(f"    {'Median:':<10} {df['n_chars'].median():.1f}")
        print(f"    {'Min:':<10} {df['n_chars'].min()}")
        print(f"    {'Max:':<10} {df['n_chars'].max()}")

        # ── per-cluster ───────────────────────────────────────────────────────
        #print(f"\n  Per-cluster breakdown:")
        # cluster_stats = (
        #     df.groupby("label")
        #     .agg(
        #         docs=("text", "count"),
        #         words_mean=("n_words", "mean"),
        #         words_std=("n_words", "std"),
        #         words_min=("n_words", "min"),
        #         words_max=("n_words", "max"),
        #         chars_mean=("n_chars", "mean"),
        #     )
        #     .sort_values("docs", ascending=False)
        # )
        # cluster_stats["docs_%"] = (cluster_stats["docs"] / n_docs * 100).map("{:.1f}%".format)
        # cluster_stats["words_mean"] = cluster_stats["words_mean"].map("{:.1f}".format)
        # cluster_stats["words_std"]  = cluster_stats["words_std"].map("{:.1f}".format)
        # cluster_stats["chars_mean"] = cluster_stats["chars_mean"].map("{:.1f}".format)

        # display = cluster_stats[["docs", "docs_%", "words_mean", "words_std", "words_min", "words_max", "chars_mean"]]
        # display.columns = ["docs", "docs_%", "avg_words", "std_words", "min_words", "max_words", "avg_chars"]
        #print(display.to_string())
        print()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output-dir", default="analysis")
    args = parser.parse_args()

    print("Loading results...")
    dfs = {ds: load_sweep(args.results_dir, ds) for ds in DATASETS}
    all_results = {ds: load_results(args.results_dir, ds) for ds in DATASETS}

    print("Generating charts...")
    plot_test_scores(all_results, args.output_dir)
    plot_sweep_distributions(dfs, args.output_dir)
    plot_hyperparam_effects(dfs, args.output_dir)
    plot_combined_heatmap(all_results, args.output_dir)
    plot_per_cluster_heatmap(all_results, args.output_dir)
    plot_dev_vs_test(all_results, args.output_dir)
    plot_param_variance(dfs, args.output_dir)

    print_dataset_statistics()
    print_sweep_tables(dfs)
    print_label_comparison(all_results)
    print_insights(dfs, all_results)


if __name__ == "__main__":
    #main()
    print_dataset_statistics()