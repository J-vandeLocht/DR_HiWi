import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
from sklearn.metrics import (
    precision_recall_curve, roc_curve, auc,
    average_precision_score, roc_auc_score, confusion_matrix
)
from sklearn.calibration import calibration_curve


def find_best_threshold(y_true, y_probs):
    prec, rec, thresholds = precision_recall_curve(y_true, y_probs)
    f1_scores = np.divide(2 * (prec * rec), (prec + rec),
                          out=np.zeros_like(prec), where=(prec + rec) != 0)
    best_idx = np.argmax(f1_scores)
    return thresholds[min(best_idx, len(thresholds) - 1)], f1_scores[best_idx]


def generate_analysis(csv_path, output_dir):
    df = pd.read_csv(csv_path)
    split_name = os.path.basename(csv_path).replace('.csv', '')

    # Auto-detect score column
    score_col = [c for c in df.columns if c not in ['video', 'label']][0]
    y_true, y_probs = df['label'].values, df[score_col].values

    # --- 1. Metrics Computation ---
    # Method A: Direct Scores
    roc_auc_direct = roc_auc_score(y_true, y_probs)
    pr_auc_direct = average_precision_score(y_true, y_probs)

    # Method B: Area via Integration (Trapezoidal)
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    prec, rec, _ = precision_recall_curve(y_true, y_probs)
    roc_auc_integral = auc(fpr, tpr)
    pr_auc_integral = auc(rec, prec)

    best_thresh, best_f1 = find_best_threshold(y_true, y_probs)

    # --- 2. Plotting (5 Panels: Diagnostic + Curves) ---
    fig, axes = plt.subplots(1, 5, figsize=(28, 5))

    # Panel 1: Ranking
    sort_idx = np.argsort(-y_probs)
    colors = ['#ff4d4d' if y == 1 else '#1a75ff' for y in y_true[sort_idx]]
    axes[0].scatter(range(len(y_probs)), y_probs[sort_idx], c=colors, s=10, alpha=0.5)
    axes[0].axhline(best_thresh, color='green', linestyle='--', label=f'Thresh: {best_thresh:.2f}')
    axes[0].set_title(f"Ranking\nF1: {best_f1:.3f}")
    axes[0].legend()

    # Panel 2: Confusion Matrix
    y_pred = (y_probs >= best_thresh).astype(int)
    sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d', cmap='Blues', ax=axes[1], cbar=False)
    axes[1].set_title(f"CM @ Opt. Thresh")

    # Panel 3: ROC Curve
    axes[2].plot(fpr, tpr, color='darkorange', lw=2,
                 label=f'Direct: {roc_auc_direct:.3f}\nIntegral: {roc_auc_integral:.3f}')
    axes[2].plot([0, 1], [0, 1], color='navy', linestyle='--')
    axes[2].set_title("ROC Curve")
    axes[2].set_xlabel("FPR")
    axes[2].set_ylabel("TPR")
    axes[2].legend(loc="lower right", fontsize='small')

    # Panel 4: Precision-Recall Curve
    axes[3].plot(rec, prec, color='green', lw=2, label=f'Direct: {pr_auc_direct:.3f}\nIntegral: {pr_auc_integral:.3f}')
    axes[3].set_title("PR Curve")
    axes[3].set_xlabel("Recall")
    axes[3].set_ylabel("Precision")
    axes[3].legend(loc="lower left", fontsize='small')

    # Panel 5: Calibration
    p_true, p_pred = calibration_curve(y_true, y_probs, n_bins=10)
    axes[4].plot(p_pred, p_true, marker='s')
    axes[4].plot([0, 1], [0, 1], linestyle='--', color='gray')
    axes[4].set_title("Calibration")

    plt.suptitle(f"Split Analysis: {split_name}", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_dir, f"full_diag_{split_name}.png"), dpi=150)
    plt.close()

    return {
        "Split": split_name,
        "Best_Thresh": best_thresh,
        "F1": best_f1,
        "ROC_AUC_Direct": roc_auc_direct,
        "ROC_AUC_Integral": roc_auc_integral,
        "PR_AUC_Direct": pr_auc_direct,
        "PR_AUC_Integral": pr_auc_integral
    }


def main():
    INPUT_FOLDER = "."  # <--- UPDATE THIS
    OUTPUT_FOLDER = "split_analysis_results"

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    csv_files = glob.glob(os.path.join(INPUT_FOLDER, "*.csv"))

    if not csv_files:
        print("No CSV files found.")
        return

    summary_list = []
    for f in csv_files:
        metrics = generate_analysis(f, OUTPUT_FOLDER)
        summary_list.append(metrics)
        print(f"Processed {metrics['Split']}")

    # Save summary table
    summary_df = pd.DataFrame(summary_list)
    summary_df.to_csv(os.path.join(OUTPUT_FOLDER, "analysis_summary.csv"), index=False)

    # Terminal display
    print("\n=== Summary Stats ===")
    cols_to_show = ["Split", "F1", "ROC_AUC_Direct", "PR_AUC_Direct"]
    print(summary_df[cols_to_show].to_string(index=False))


if __name__ == "__main__":
    main()