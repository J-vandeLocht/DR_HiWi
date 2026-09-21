import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score, cohen_kappa_score,
    confusion_matrix
)

NUM_CLASSES = 5
CLASS_NAMES = ['0', '1', '2', '3', '4']

HEADLINE_METRICS = [
    'acc', 'qwk', 'referable_recall', 'f1_macro', 'f1_weighted',
    'precision_macro', 'recall_macro', 'roc_auc_macro', 'pr_auc_macro'
]

PER_CLASS_METRIC_PREFIXES = [
    'f1_class', 'precision_class', 'recall_class',
    'pr_auc_class', 'roc_auc_class'
]


def detect_models_and_columns(df):
    """Detect frame classifier, video classifier, and transformer."""
    models = []

    pred_cols = [c for c in df.columns if 'pred_grade' in c]

    for pred_col in pred_cols:
        if pred_col.startswith("Clf_frame"):
            model_name = "Classifier Frame"
            prob_prefix = "Clf_frame_"
        elif pred_col.startswith("Clf_video"):
            model_name = "Classifier Video"
            prob_prefix = "Clf_video_"
        elif pred_col.startswith("Trans"):
            model_name = "Transformer"
            prob_prefix = "Trans_"
        else:
            continue

        prob_cols = [f"{prob_prefix}prob_{c}" for c in range(NUM_CLASSES)]

        if not set(prob_cols).issubset(df.columns):
            raise ValueError(
                f"Missing probability columns for {model_name}: {prob_cols}"
            )

        models.append({
            "model_name": model_name,
            "pred_col": pred_col,
            "prob_cols": prob_cols
        })

    return models


def generate_analysis(df, split_name, output_dir, model_info):
    y_true = df['grade'].values.astype(int)
    y_pred = df[model_info["pred_col"]].values.astype(int)
    y_probs = df[model_info["prob_cols"]].values.astype(float)

    metrics = multiclass_metrics(
        y_true, y_pred, y_probs, num_classes=NUM_CLASSES
    )

    model_name = model_info["model_name"]
    tag = f"{split_name}_{model_name.replace(' ', '_')}"

    save_confusion_matrix(
        y_true, y_pred, tag, output_dir, class_names=CLASS_NAMES
    )

    row = {"Split": split_name, "Model": model_name}
    row.update(metrics)
    return row


def multiclass_metrics(y_true, y_pred, y_probs, num_classes):
    labels = list(range(num_classes))
    y_true_onehot = np.eye(num_classes)[y_true]

    qwk = cohen_kappa_score(y_true, y_pred, weights='quadratic')

    y_true_referable = (y_true >= 2).astype(int)
    y_pred_referable = (y_pred >= 2).astype(int)
    referable_recall = recall_score(
        y_true_referable, y_pred_referable, zero_division=0
    )

    present = np.unique(y_true)

    if len(present) > 1:
        roc_auc_macro = roc_auc_score(
            y_true, y_probs, labels=labels,
            multi_class='ovr', average='macro'
        )
        roc_auc_weighted = roc_auc_score(
            y_true, y_probs, labels=labels,
            multi_class='ovr', average='weighted'
        )
    else:
        roc_auc_macro = 0.5
        roc_auc_weighted = 0.5

    metrics = {
        'acc': accuracy_score(y_true, y_pred),
        'qwk': qwk,
        'referable_recall': referable_recall,
        'f1_macro': f1_score(
            y_true, y_pred, labels=labels,
            average='macro', zero_division=0
        ),
        'f1_weighted': f1_score(
            y_true, y_pred, labels=labels,
            average='weighted', zero_division=0
        ),
        'precision_macro': precision_score(
            y_true, y_pred, labels=labels,
            average='macro', zero_division=0
        ),
        'recall_macro': recall_score(
            y_true, y_pred, labels=labels,
            average='macro', zero_division=0
        ),
        'roc_auc_macro': roc_auc_macro,
        'roc_auc_weighted': roc_auc_weighted,
        'pr_auc_macro': average_precision_score(
            y_true_onehot, y_probs, average='macro'
        ),
        'pr_auc_weighted': average_precision_score(
            y_true_onehot, y_probs, average='weighted'
        )
    }

    f1_per_class = f1_score(
        y_true, y_pred, labels=labels,
        average=None, zero_division=0
    )
    prec_per_class = precision_score(
        y_true, y_pred, labels=labels,
        average=None, zero_division=0
    )
    rec_per_class = recall_score(
        y_true, y_pred, labels=labels,
        average=None, zero_division=0
    )
    ap_per_class = average_precision_score(
        y_true_onehot, y_probs, average=None
    )

    for c in labels:
        metrics[f'f1_class{c}'] = f1_per_class[c]
        metrics[f'precision_class{c}'] = prec_per_class[c]
        metrics[f'recall_class{c}'] = rec_per_class[c]
        metrics[f'pr_auc_class{c}'] = ap_per_class[c]

        if len(present) > 1 and c in present:
            metrics[f'roc_auc_class{c}'] = roc_auc_score(
                (y_true == c).astype(int), y_probs[:, c]
            )
        else:
            metrics[f'roc_auc_class{c}'] = 0.5

    metrics['pr_auc'] = metrics['pr_auc_macro']
    return metrics


def save_confusion_matrix(y_true, y_pred, tag, run_dir, class_names=None):
    num_classes = (
        len(class_names)
        if class_names is not None
        else int(max(y_true.max(), y_pred.max())) + 1
    )
    labels = list(range(num_classes))

    if class_names is None:
        class_names = [str(l) for l in labels]

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues', cbar=False,
        xticklabels=class_names, yticklabels=class_names, ax=axes[0]
    )
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')
    axes[0].set_title(f'Counts - {tag}')

    sns.heatmap(
        cm_norm, annot=True, fmt='.2f', cmap='Blues',
        cbar=False, vmin=0, vmax=1,
        xticklabels=class_names, yticklabels=class_names, ax=axes[1]
    )
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Actual')
    axes[1].set_title(f'Row-normalized (Recall) - {tag}')

    plt.tight_layout()

    save_path = os.path.join(run_dir, f"cm_{tag}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    return save_path, cm


def plot_metric_summary(summary_df, output_dir):
    metrics_to_plot = [
        ('acc', 'Accuracy'),
        ('qwk', 'Quadratic Weighted Kappa (QWK)'),
        ('referable_recall', 'Referable Recall (Grade >= 2)'),
        ('f1_macro', 'F1 Macro'),
        ('roc_auc_macro', 'ROC-AUC Macro'),
        ('pr_auc_macro', 'PR-AUC Macro')
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    sns.set_theme(style="whitegrid")

    for idx, (metric, title) in enumerate(metrics_to_plot):
        if metric not in summary_df.columns:
            continue

        ax = axes[idx]

        sns.barplot(
            data=summary_df,
            x='Model',
            y=metric,
            hue='Model',
            ax=ax,
            palette="Set2",
            width=0.5,
            alpha=0.7,
            legend=False,
            errorbar='sd',
            capsize=0.15,
            err_kws={'linewidth': 1.5, 'color': '#333333'}
        )

        sns.stripplot(
            data=summary_df,
            x='Model',
            y=metric,
            ax=ax,
            color='#111111',
            jitter=0.08,
            size=6,
            alpha=0.85,
            linewidth=0
        )

        vals = summary_df[metric]
        lo, hi = vals.min(), vals.max()
        val_range = hi - lo
        margin = (
            val_range * 0.2
            if val_range > 0
            else max(abs(hi) * 0.1, 0.02)
        )

        ax.set_ylim(lo - margin, hi + margin)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel('Score')

    plt.suptitle(
        'Model Performance Across 5 Splits (Mean ± SD)',
        fontsize=16, fontweight='bold', y=0.98
    )
    plt.tight_layout()

    save_path = os.path.join(
        output_dir, "model_comparison_summary.png"
    )
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"\nSummary comparison saved to: {save_path}")


def print_comparison_overview(summary_df, output_dir):
    """Print and save mean +/- SD for all three models."""

    numeric_cols = [
        c for c in HEADLINE_METRICS
        if c in summary_df.columns
    ]

    grouped = summary_df.groupby('Model')[numeric_cols]
    means = grouped.mean()
    stds = grouped.std()

    model_order = [
        'Classifier Frame',
        'Classifier Video',
        'Transformer'
    ]

    comparison_data = []

    for metric in numeric_cols:
        row = {'Metric': metric}

        for model in model_order:
            if model in means.index:
                mean = means.loc[model, metric]
                std = stds.loc[model, metric]
                row[model] = f"{mean:.4f} ± {std:.4f}"
            else:
                row[model] = "N/A"

        comparison_data.append(row)

    comp_df = pd.DataFrame(comparison_data)

    print("\n" + "=" * 20 + " MODEL COMPARISON OVERVIEW " + "=" * 20)
    print(comp_df.to_string(index=False))

    output_file = os.path.join(
        output_dir, "model_comparison_summary.csv"
    )
    comp_df.to_csv(output_file, index=False)

    print(f"\nComparison table saved to: {output_file}")


def main():
    INPUT_FOLDER = "."
    OUTPUT_FOLDER = "split_analysis_results"
    CM_FOLDER = os.path.join(OUTPUT_FOLDER, "confusion_matrices")

    os.makedirs(CM_FOLDER, exist_ok=True)

    csv_files = sorted(
        glob.glob(os.path.join(INPUT_FOLDER, "*.csv"))
    )
    csv_files = [
        f for f in csv_files
        if "summary" not in os.path.basename(f)
    ]

    if not csv_files:
        print(f"No CSV files found in {INPUT_FOLDER}.")
        return

    summary_list = []
    combined_data = {}

    for f in csv_files:
        df = pd.read_csv(f)
        split_name = os.path.basename(f).replace('.csv', '')

        if 'grade' not in df.columns:
            continue

        models_info = detect_models_and_columns(df)

        if not models_info:
            continue

        for model_info in models_info:
            row = generate_analysis(
                df,
                split_name,
                CM_FOLDER,
                model_info
            )
            summary_list.append(row)

            model_name = model_info["model_name"]

            if model_name not in combined_data:
                combined_data[model_name] = {
                    'y_true': [],
                    'y_pred': []
                }

            combined_data[model_name]['y_true'].extend(
                df['grade'].values.astype(int)
            )
            combined_data[model_name]['y_pred'].extend(
                df[model_info["pred_col"]].values.astype(int)
            )

        processed_names = [
            m["model_name"] for m in models_info
        ]
        print(
            f"Processed {split_name} for models: "
            f"{', '.join(processed_names)}"
        )

    if not summary_list:
        print("No valid metrics were computed.")
        return

    summary_df = pd.DataFrame(summary_list).sort_values(
        by=['Split', 'Model']
    )

    summary_df.to_csv(
        os.path.join(
            OUTPUT_FOLDER,
            "analysis_summary_full.csv"
        ),
        index=False
    )

    # Combined confusion matrices across all 5 splits
    for model_name, data in combined_data.items():
        save_confusion_matrix(
            np.array(data['y_true']),
            np.array(data['y_pred']),
            f"all_splits_combined_{model_name.replace(' ', '_')}",
            CM_FOLDER,
            class_names=CLASS_NAMES
        )

    plot_metric_summary(summary_df, OUTPUT_FOLDER)
    print_comparison_overview(summary_df, OUTPUT_FOLDER)


if __name__ == "__main__":
    main()