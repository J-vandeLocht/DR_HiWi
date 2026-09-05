import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score, precision_recall_curve,
    auc, cohen_kappa_score, confusion_matrix
)

NUM_CLASSES = 5
CLASS_NAMES = ['0', '1', '2', '3', '4']

HEADLINE_METRICS = [
    'acc', 'qwk', 'referable_recall',
    'f1_macro', 'f1_weighted',
    'precision_macro', 'recall_macro',
    'roc_auc_macro', 'pr_auc_macro',
]

PER_CLASS_METRIC_PREFIXES = ['f1_class', 'precision_class', 'recall_class', 'pr_auc_class', 'roc_auc_class']


def generate_analysis(df, split_name, output_dir, model_name="Transformer"):
    y_true = df['grade'].values.astype(int)
    y_pred = df['pred_grade'].values.astype(int)
    prob_cols = [f"prob_{c}" for c in range(NUM_CLASSES)]
    y_probs = df[prob_cols].values.astype(float)

    metrics = multiclass_metrics(y_true, y_pred, y_probs, num_classes=NUM_CLASSES)

    # Confusion matrix (counts + row-normalized), saved as one PNG per split
    cm_path, _ = save_confusion_matrix(y_true, y_pred, split_name, output_dir, class_names=CLASS_NAMES)

    row = {"Split": split_name, "Model": model_name}
    row.update(metrics)
    return row


def multiclass_metrics(y_true, y_pred, y_probs, num_classes):
    labels = list(range(num_classes))
    y_true_onehot = np.eye(num_classes)[y_true]

    qwk = cohen_kappa_score(y_true, y_pred, weights='quadratic')

    y_true_referable = (y_true >= 2).astype(int)
    y_pred_referable = (y_pred >= 2).astype(int)
    referable_recall = recall_score(y_true_referable, y_pred_referable, zero_division=0)

    present = np.unique(y_true)
    roc_auc_macro = (
        roc_auc_score(y_true, y_probs, labels=labels, multi_class='ovr', average='macro')
        if len(present) > 1 else 0.5
    )
    roc_auc_weighted = (
        roc_auc_score(y_true, y_probs, labels=labels, multi_class='ovr', average='weighted')
        if len(present) > 1 else 0.5
    )

    metrics = {
        'acc': accuracy_score(y_true, y_pred),
        'qwk': qwk,
        'referable_recall': referable_recall,
        'f1_macro': f1_score(y_true, y_pred, labels=labels, average='macro', zero_division=0),
        'f1_weighted': f1_score(y_true, y_pred, labels=labels, average='weighted', zero_division=0),
        'precision_macro': precision_score(y_true, y_pred, labels=labels, average='macro', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, labels=labels, average='macro', zero_division=0),
        'roc_auc_macro': roc_auc_macro,
        'roc_auc_weighted': roc_auc_weighted,
        'pr_auc_macro': average_precision_score(y_true_onehot, y_probs, average='macro'),
        'pr_auc_weighted': average_precision_score(y_true_onehot, y_probs, average='weighted'),
    }

    f1_per_class = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    prec_per_class = precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    rec_per_class = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    ap_per_class = average_precision_score(y_true_onehot, y_probs, average=None)

    for c in labels:
        metrics[f'f1_class{c}'] = f1_per_class[c]
        metrics[f'precision_class{c}'] = prec_per_class[c]
        metrics[f'recall_class{c}'] = rec_per_class[c]
        metrics[f'pr_auc_class{c}'] = ap_per_class[c]
        if len(present) > 1 and c in present:
            metrics[f'roc_auc_class{c}'] = roc_auc_score((y_true == c).astype(int), y_probs[:, c])

    metrics['pr_auc'] = metrics['pr_auc_macro']
    return metrics


def save_confusion_matrix(y_true, y_pred, tag, run_dir, class_names=None):
    num_classes = len(class_names) if class_names is not None else (int(max(y_true.max(), y_pred.max())) + 1)
    labels = list(range(num_classes))
    if class_names is None:
        class_names = [str(l) for l in labels]

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)  # row-normalized, avoid div/0

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=class_names, yticklabels=class_names, ax=axes[0])
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')
    axes[0].set_title(f'Counts - {tag}')

    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', cbar=False, vmin=0, vmax=1,
                xticklabels=class_names, yticklabels=class_names, ax=axes[1])
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Actual')
    axes[1].set_title(f'Row-normalized (Recall) - {tag}')

    plt.tight_layout()
    save_path = os.path.join(run_dir, f"cm_{tag}.png")
    plt.savefig(save_path)
    plt.close(fig)
    return save_path, cm


def summarize_across_splits(summary_df):
    """Mean +/- std across splits, separately for headline metrics and
    per-class metrics. Non-numeric / list-like columns (Split, Model) are
    excluded automatically."""
    numeric_cols = [c for c in summary_df.columns if c not in ("Split", "Model")]

    agg = summary_df[numeric_cols].agg(['mean', 'std']).T
    agg.columns = ['Mean', 'Std']
    agg.index.name = 'Metric'
    return agg.reset_index()


def print_headline_table(summary_df):
    cols = ["Split"] + [c for c in HEADLINE_METRICS if c in summary_df.columns]
    print("\n=== Per-Split Headline Metrics ===")
    print(summary_df[cols].round(4).to_string(index=False))


def print_per_class_table(summary_df):
    print("\n=== Per-Split, Per-Class Metrics ===")
    for prefix in PER_CLASS_METRIC_PREFIXES:
        class_cols = [f"{prefix}{c}" for c in range(NUM_CLASSES) if f"{prefix}{c}" in summary_df.columns]
        if not class_cols:
            continue
        print(f"\n-- {prefix.rstrip('_class')} per class --")
        print(summary_df[["Split"] + class_cols].round(4).to_string(index=False))


def print_averaged_summary(avg_df):
    print("\n=== Averaged Across Splits (Mean +/- Std) ===")

    headline_rows = avg_df[avg_df['Metric'].isin(HEADLINE_METRICS)]
    print("\n-- Headline --")
    for _, r in headline_rows.iterrows():
        print(f"{r['Metric']:<20} {r['Mean']:.4f} +/- {r['Std']:.4f}")

    for prefix in PER_CLASS_METRIC_PREFIXES:
        class_rows = avg_df[avg_df['Metric'].str.startswith(prefix)]
        if class_rows.empty:
            continue
        print(f"\n-- {prefix.rstrip('_class')} per class (avg over splits) --")
        for _, r in class_rows.iterrows():
            cls = r['Metric'].replace(prefix, '')
            print(f"  class {cls:<3} {r['Mean']:.4f} +/- {r['Std']:.4f}")


def main():
    INPUT_FOLDER = "."
    OUTPUT_FOLDER = "split_analysis_results"
    CM_FOLDER = os.path.join(OUTPUT_FOLDER, "confusion_matrices")
    os.makedirs(CM_FOLDER, exist_ok=True)

    csv_files = sorted(glob.glob(os.path.join(INPUT_FOLDER, "*.csv")))
    csv_files = [f for f in csv_files if "summary" not in os.path.basename(f)]

    if not csv_files:
        print(f"No CSV files found in {INPUT_FOLDER}.")
        return

    summary_list = []

    for f in csv_files:
        df = pd.read_csv(f)
        split_name = os.path.basename(f).replace('.csv', '')

        required_cols = {'grade', 'pred_grade'} | {f"prob_{c}" for c in range(NUM_CLASSES)}
        if not required_cols.issubset(df.columns):
            print(f"Skipping {f}: missing required columns {required_cols - set(df.columns)}")
            continue

        row = generate_analysis(df, split_name, CM_FOLDER)
        summary_list.append(row)
        print(f"Processed {split_name}")

    if not summary_list:
        print("No valid metrics were computed. Ensure CSVs have grade/pred_grade/prob_0..4 columns.")
        return

    summary_df = pd.DataFrame(summary_list).sort_values(by='Split')
    drop_cols = [c for c in ('y_true', 'y_pred') if c in summary_df.columns]
    summary_df = summary_df.drop(columns=drop_cols)

    summary_df.to_csv(os.path.join(OUTPUT_FOLDER, "analysis_summary_full.csv"), index=False)

    avg_df = summarize_across_splits(summary_df)
    avg_df.to_csv(os.path.join(OUTPUT_FOLDER, "analysis_summary_averaged.csv"), index=False)

    all_y_true = np.concatenate([pd.read_csv(f)['grade'].values.astype(int) for f in csv_files])
    all_y_pred = np.concatenate([pd.read_csv(f)['pred_grade'].values.astype(int) for f in csv_files])
    save_confusion_matrix(all_y_true, all_y_pred, "all_splits_combined", CM_FOLDER, class_names=CLASS_NAMES)

    print_headline_table(summary_df)
    print_per_class_table(summary_df)
    print_averaged_summary(avg_df)

    print(f"\nConfusion matrices saved to: {CM_FOLDER}")


if __name__ == "__main__":
    main()