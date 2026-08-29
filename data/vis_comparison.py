"""
Plot average ROC_AUC_Direct and PR_AUC_Direct per model, averaged over splits,
for two datasets side by side (one row per dataset).

Expects two CSVs, each with (at least) these columns:
    Split, Model, Best_Thresh, F1, ROC_AUC_Direct, ROC_AUC_Integral,
    PR_AUC_Direct, PR_AUC_Integral

Produces a figure with 2 rows x 2 columns:
    - Row 1: --name1 dataset -> ROC_AUC_Direct, PR_AUC_Direct
    - Row 2: --name2 dataset -> ROC_AUC_Direct, PR_AUC_Direct

The y-axis is shared and normalized across BOTH rows so the bars are
directly comparable across datasets.

Usage:
    python plot_model_metrics.py \
        --csv_file1 results_a.csv --name1 "Dataset A" \
        --csv_file2 results_b.csv --name2 "Dataset B" \
        --out_file model_metrics.png
"""

import argparse
from collections import defaultdict
import csv

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
    parser.add_argument('--csv_file1', required=True, help="Path to the first dataset's CSV")
    parser.add_argument('--name1', default='Dataset 1', help="Display name for the first dataset (row 1)")
    parser.add_argument('--csv_file2', required=True, help="Path to the second dataset's CSV")
    parser.add_argument('--name2', default='Dataset 2', help="Display name for the second dataset (row 2)")
    parser.add_argument('--out_file', default='model_metrics.png')
    parser.add_argument('--y_min', type=float, default=None,
                         help="Optional y-axis minimum, to zoom in on differences (e.g. 0.6)")
    parser.add_argument('--dpi', type=int, default=150)
    args = parser.parse_args()

    csv_paths = [args.csv_file1, args.csv_file2]
    labels = [args.name1, args.name2]

    # Load + average each file independently
    per_file_model_names = []
    per_file_averages = []
    for path in csv_paths:
        rows = load_csv(path)
        model_names, averages = compute_model_averages(rows)
        per_file_model_names.append(model_names)
        per_file_averages.append(averages)

    # Consistent color per model across BOTH files (union of model names, first-seen order)
    all_model_names = []
    for model_names in per_file_model_names:
        for m in model_names:
            if m not in all_model_names:
                all_model_names.append(m)

    cmap = plt.get_cmap('tab10' if len(all_model_names) <= 10 else 'tab20')
    model_colors = {name: cmap(i % cmap.N) for i, name in enumerate(all_model_names)}

    n_cols = len(METRICS)

    # Extra vertical gap between the two rows for clearer visual separation,
    # plus dedicated row space for a bold dataset-name banner above each row.
    fig, axes = plt.subplots(
        2, n_cols, figsize=(6 * n_cols, 11),
        gridspec_kw={'hspace': 0.55},
    )
    # Reserve margins explicitly instead of using tight_layout, since we add
    # figure-level artists (banners, divider line) after the axes are laid
    # out, which tight_layout can't account for.
    fig.subplots_adjust(left=0.06, right=0.97, top=0.90, bottom=0.09)

    # Shared y-limit across BOTH files and both metrics so everything
    # is directly comparable
    all_values = [
        averages[m][metric]
        for averages, model_names in zip(per_file_averages, per_file_model_names)
        for m in model_names
        for metric in METRICS
    ]
    y_max = max(all_values) * 1.05
    y_min = args.y_min if args.y_min is not None else 0

    for row_idx, (label, model_names, averages) in enumerate(
        zip(labels, per_file_model_names, per_file_averages)
    ):
        for col_idx, metric in enumerate(METRICS):
            ax = axes[row_idx][col_idx]
            values = [averages[m][metric] for m in model_names]
            colors = [model_colors[m] for m in model_names]

            bars = ax.bar(model_names, values, color=colors)
            ax.bar_label(bars, fmt='%.4f', fontsize=8, padding=2)

            ax.set_title(metric, fontsize=12, fontweight='bold')
            ax.set_ylim(y_min, y_max)
            ax.set_xticklabels([])  # model identity shown via color + legend instead
            ax.tick_params(axis='x', bottom=False)
            ax.tick_params(axis='y', labelsize=9)

        # Bold dataset-name banner spanning the row, placed above its two subplots
        row_top = axes[row_idx][0].get_position().y1
        fig.text(
            0.5, row_top + 0.045, label,
            ha='center', va='bottom', fontsize=15, fontweight='bold',
        )

    # Horizontal divider line between the two rows for extra visual separation
    row0_bottom = axes[0][0].get_position().y0
    row1_top = axes[1][0].get_position().y1
    divider_y = (row0_bottom + row1_top) / 2
    fig.add_artist(plt.Line2D([0.03, 0.97], [divider_y, divider_y],
                               color='gray', linewidth=1, linestyle='--',
                               transform=fig.transFigure))

    # Single shared legend mapping color -> model
    legend_handles = [
        mpatches.Patch(color=model_colors[name], label=name)
        for name in all_model_names
    ]
    fig.legend(
        handles=legend_handles,
        loc='lower center',
        ncol=min(len(all_model_names), 6),
        bbox_to_anchor=(0.5, -0.02),
        fontsize=9,
        frameon=False,
    )

    fig.savefig(args.out_file, dpi=args.dpi, bbox_inches='tight')
    print(f"Saved plot to {args.out_file}")


if __name__ == '__main__':
    main()