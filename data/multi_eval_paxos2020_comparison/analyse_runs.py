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
    f1_scores = np.divide(
        2 * (prec * rec),
        (prec + rec),
        out=np.zeros_like(prec),
        where=(prec + rec) != 0
    )
    best_idx = np.argmax(f1_scores)
    return thresholds[min(best_idx, len(thresholds) - 1)], f1_scores[best_idx]


def generate_analysis(df, split_name, score_col, model_name, output_dir):
    y_true, y_probs = df['label'].values, df[score_col].values

    # --- 1. Metrics Computation ---
    roc_auc_direct = roc_auc_score(y_true, y_probs)
    pr_auc_direct = average_precision_score(y_true, y_probs)

    fpr, tpr, _ = roc_curve(y_true, y_probs)
    prec, rec, _ = precision_recall_curve(y_true, y_probs)
    roc_auc_integral = auc(fpr, tpr)
    pr_auc_integral = auc(rec, prec)

    best_thresh, best_f1 = find_best_threshold(y_true, y_probs)

    # --- 2. Plotting (4 Panels: Diagnostic + Curves) ---
    fig, axes = plt.subplots(1, 4, figsize=(24, 5))

    # Panel 1: Ranking (Using Grade Color Scheme)
    sort_idx = np.argsort(-y_probs)
    grade_colors = {
        0: "green",
        1: "yellow",
        2: "orange",
        3: "red",
        4: "black"
    }
    grades_sorted = df["grade"].values[sort_idx]
    colors = [grade_colors.get(g, "gray") for g in grades_sorted]

    axes[0].scatter(
        range(len(y_probs)),
        y_probs[sort_idx],
        c=colors,
        s=15,
        alpha=0.8,
        edgecolors='black',
        linewidths=0.5
    )
    axes[0].axhline(best_thresh, color='black', linestyle='--', label=f'Thresh: {best_thresh:.2f}')
    axes[0].set_title(f"Ranking\nF1: {best_f1:.3f}")
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].set_ylabel("Score")
    axes[0].legend()

    # Panel 2: Confusion Matrix
    y_pred = (y_probs >= best_thresh).astype(int)
    sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d', cmap='Blues', ax=axes[1], cbar=False)
    axes[1].set_title("CM @ Opt. Thresh")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Actual")

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

    # Master Title & Layout Setup
    plt.suptitle(f"Split Analysis: {split_name} | Model: {model_name}", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    save_path = os.path.join(output_dir, f"full_diag_{split_name}_{model_name}.png")
    plt.savefig(save_path, dpi=150)
    plt.close()

    return {
        "Split": split_name,
        "Model": model_name,
        "Best_Thresh": best_thresh,
        "F1": best_f1,
        "ROC_AUC_Direct": roc_auc_direct,
        "ROC_AUC_Integral": roc_auc_integral,
        "PR_AUC_Direct": pr_auc_direct,
        "PR_AUC_Integral": pr_auc_integral
    }


def main():
    INPUT_FOLDER = "."
    BASE_OUTPUT_FOLDER = "split_analysis_results"

    # Separate diagnostic output folders per archetype
    CLF_2020_DIR = os.path.join(BASE_OUTPUT_FOLDER, "Classifier_2020")
    CLF_2025_DIR = os.path.join(BASE_OUTPUT_FOLDER, "Classifier_2025")
    CLF_FUSED_DIR = os.path.join(BASE_OUTPUT_FOLDER, "Classifier_Fused")
    TRANS_2020_DIR = os.path.join(BASE_OUTPUT_FOLDER, "Transformer_2020")
    TRANS_2025_DIR = os.path.join(BASE_OUTPUT_FOLDER, "Transformer_2025")
    TRANS_FUSED_DIR = os.path.join(BASE_OUTPUT_FOLDER, "Transformer_Fused")

    os.makedirs(CLF_2020_DIR, exist_ok=True)
    os.makedirs(CLF_2025_DIR, exist_ok=True)
    os.makedirs(CLF_FUSED_DIR, exist_ok=True)
    os.makedirs(TRANS_2020_DIR, exist_ok=True)
    os.makedirs(TRANS_2025_DIR, exist_ok=True)
    os.makedirs(TRANS_FUSED_DIR, exist_ok=True)

    csv_files = sorted(glob.glob(os.path.join(INPUT_FOLDER, "*.csv")))

    if not csv_files:
        print(f"No CSV files found in {INPUT_FOLDER}.")
        return

    summary_list = []

    # Process individual runs independently
    for f in csv_files:
        if "summary" in f:
            continue

        df = pd.read_csv(f)
        split_name = os.path.basename(f).replace('.csv', '')

        # Isolate tracking columns dynamically
        clf_2020_col = next((col for col in df.columns if "Clf_2020" in col), None)
        clf_2025_col = next((col for col in df.columns if "Clf_2025" in col), None)
        clf_fused_col = next((col for col in df.columns if "Clf_fused" in col), None)
        trans_2020_col = next((col for col in df.columns if "Trans_2020" in col), None)
        trans_2025_col = next((col for col in df.columns if "Trans_2025" in col), None)
        trans_fused_col = next((col for col in df.columns if "Trans_fused" in col), None)

        if clf_2020_col is not None:
            clf_metrics = generate_analysis(df, split_name, clf_2020_col, "Classifier_2020", CLF_2020_DIR)
            summary_list.append(clf_metrics)
            print(f"Processed Classifier for {split_name}")
        if clf_2025_col is not None:
            clf_metrics = generate_analysis(df, split_name, clf_2025_col, "Classifier_2025", CLF_2025_DIR)
            summary_list.append(clf_metrics)
            print(f"Processed Classifier for {split_name}")
        if clf_fused_col is not None:
            clf_metrics = generate_analysis(df, split_name, clf_fused_col, "Classifier_Fused", CLF_FUSED_DIR)
            summary_list.append(clf_metrics)
            print(f"Processed Classifier for {split_name}")

        if trans_2020_col is not None:
            trans_metrics = generate_analysis(df, split_name, trans_2020_col, "Transformer_2020", TRANS_2020_DIR)
            summary_list.append(trans_metrics)
            print(f"Processed Transformer for {split_name}")
        if trans_2025_col is not None:
            trans_metrics = generate_analysis(df, split_name, trans_2025_col, "Transformer_2025", TRANS_2025_DIR)
            summary_list.append(trans_metrics)
            print(f"Processed Transformer for {split_name}")
        if trans_fused_col is not None:
            trans_metrics = generate_analysis(df, split_name, trans_fused_col, "Transformer_Fused", TRANS_FUSED_DIR)
            summary_list.append(trans_metrics)
            print(f"Processed Transformer for {split_name}")

    if not summary_list:
        print("No valid metrics were computed. Ensure CSV headers contain the right name")
        return

    # Compile global tracking sheet
    summary_df = pd.DataFrame(summary_list)
    summary_df = summary_df.sort_values(by=['Split', 'Model'])
    summary_df.to_csv(os.path.join(BASE_OUTPUT_FOLDER, "analysis_summary.csv"), index=False)

    # Output log tables straight to terminal context
    print("\n=== Summary Stats ===")
    cols_to_show = ["Split", "Model", "F1", "ROC_AUC_Direct", "PR_AUC_Direct"]
    print(summary_df[cols_to_show].to_string(index=False))


if __name__ == "__main__":
    main()