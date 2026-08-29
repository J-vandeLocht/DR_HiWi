"""
Plot average ROC_AUC_Direct and PR_AUC_Direct per model, averaged over splits.

Expects a CSV with (at least) these columns:
    Split, Model, Best_Thresh, F1, ROC_AUC_Direct, ROC_AUC_Integral,
    PR_AUC_Direct, PR_AUC_Integral

Produces a figure with 2 subplots (1 row, 2 columns):
    - ROC_AUC_Direct averaged over splits, one bar per model
    - PR_AUC_Direct averaged over splits, one bar per model

Usage:
    python plot_model_metrics.py --csv_file results.csv --out_file model_metrics.png
"""

import argparse
import csv
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


METRICS = ['ROC_AUC_Direct', 'PR_AUC_Direct']


def load_csv(path):
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return rows


def compute_model_averages(rows):
    """Group rows by Model and average each metric in METRICS over splits.

    Returns:
        model_names: list of model names in first-seen order
        averages: dict {model_name: {metric: avg_value}}
    """
    model_names = []
    sums = defaultdict(lambda: defaultdict(float))
    counts = defaultdict(int)

    for row in rows:
        model = row['Model']
        if model not in model_names:
            model_names.append(model)
        counts[model] += 1
        for metric in METRICS:
            sums[model][metric] += float(row[metric])

    averages = {
        model: {metric: sums[model][metric] / counts[model] for metric in METRICS}
        for model in model_names
    }
    return model_names, averages


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_file', required=True,
                         help="Path to the CSV file containing per-split, per-model results")
    parser.add_argument('--out_file', default='model_metrics.png')
    parser.add_argument('--y_min', type=float, default=None,
                         help="Optional y-axis minimum, to zoom in on differences (e.g. 0.6)")
    parser.add_argument('--dpi', type=int, default=150)
    args = parser.parse_args()

    rows = load_csv(args.csv_file)
    model_names, averages = compute_model_averages(rows)

    # Assign each model a distinct, consistent color
    cmap = plt.get_cmap('tab10' if len(model_names) <= 10 else 'tab20')
    model_colors = {name: cmap(i % cmap.N) for i, name in enumerate(model_names)}

    n_cols = len(METRICS)
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 5), squeeze=False)
    axes = axes[0]

    # Shared y-limit across both subplots so they're directly comparable
    all_values = [averages[m][metric] for m in model_names for metric in METRICS]
    y_max = max(all_values) * 1.05
    y_min = args.y_min if args.y_min is not None else 0

    for col_idx, metric in enumerate(METRICS):
        ax = axes[col_idx]
        values = [averages[m][metric] for m in model_names]
        colors = [model_colors[m] for m in model_names]

        bars = ax.bar(model_names, values, color=colors)
        ax.bar_label(bars, fmt='%.4f', fontsize=8, padding=2)

        ax.set_title(metric, fontsize=13, fontweight='bold')
        ax.set_ylim(y_min, y_max)
        ax.set_xticklabels([])  # model identity shown via color + legend instead
        ax.tick_params(axis='x', bottom=False)
        ax.tick_params(axis='y', labelsize=9)

    # Single shared legend mapping color -> model
    legend_handles = [
        mpatches.Patch(color=model_colors[name], label=name)
        for name in model_names
    ]
    fig.legend(
        handles=legend_handles,
        loc='lower center',
        ncol=min(len(model_names), 6),
        bbox_to_anchor=(0.5, -0.05),
        fontsize=9,
        frameon=False,
    )

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(args.out_file, dpi=args.dpi, bbox_inches='tight')
    print(f"Saved plot to {args.out_file}")


if __name__ == '__main__':
    main()