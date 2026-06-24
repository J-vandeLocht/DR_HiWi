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


def find_best_threshold(y_true, y_probs):
    prec, rec, thresholds = precision_recall_curve(y_true, y_probs)

    f1_scores = np.divide(
        2 * (prec * rec),
        (prec + rec),
        out=np.zeros_like(prec),
        where=(prec + rec) != 0
    )

    best_idx = np.argmax(f1_scores)
    best_idx = min(best_idx, len(thresholds) - 1)

    return thresholds[best_idx], f1_scores[best_idx]


def generate_analysis(csv_path, output_dir):
    df = pd.read_csv(csv_path)
    split_name = os.path.basename(csv_path).replace(".csv", "")

    # Detect score column
    score_col = [c for c in df.columns if c not in ["video", "label", "grade"]][0]

    y_true = df["label"].values
    y_probs = df[score_col].values

    # Metrics
    roc_auc_direct = roc_auc_score(y_true, y_probs)
    pr_auc_direct = average_precision_score(y_true, y_probs)

    fpr, tpr, _ = roc_curve(y_true, y_probs)
    prec, rec, _ = precision_recall_curve(y_true, y_probs)

    roc_auc_integral = auc(fpr, tpr)
    pr_auc_integral = auc(rec, prec)

    best_thresh, best_f1 = find_best_threshold(y_true, y_probs)

    # Plot
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))

    # -------------------------
    # Panel 1: Ranking plot
    # -------------------------
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
        s=10,
        alpha=0.6
    )

    axes[0].axhline(
        best_thresh,
        color="black",
        linestyle="--",
        label=f"Thresh: {best_thresh:.2f}"
    )

    axes[0].set_title(f"Ranking\nF1: {best_f1:.3f}")
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Score")
    axes[0].legend()

    # -------------------------
    # Panel 2: Confusion matrix
    # -------------------------
    y_pred = (y_probs >= best_thresh).astype(int)

    sns.heatmap(
        confusion_matrix(y_true, y_pred),
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=axes[1],
        cbar=False
    )

    axes[1].set_title("CM @ Opt. Thresh")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Actual")

    # -------------------------
    # Panel 3: ROC
    # -------------------------
    axes[2].plot(
        fpr,
        tpr,
        color="darkorange",
        lw=2,
        label=f"Direct: {roc_auc_direct:.3f}\nIntegral: {roc_auc_integral:.3f}"
    )

    axes[2].plot([0, 1], [0, 1], color="navy", linestyle="--")
    axes[2].set_title("ROC Curve")
    axes[2].set_xlabel("FPR")
    axes[2].set_ylabel("TPR")
    axes[2].legend(loc="lower right", fontsize="small")

    # -------------------------
    # Panel 4: PR
    # -------------------------
    axes[3].plot(
        rec,
        prec,
        color="green",
        lw=2,
        label=f"Direct: {pr_auc_direct:.3f}\nIntegral: {pr_auc_integral:.3f}"
    )

    axes[3].set_title("PR Curve")
    axes[3].set_xlabel("Recall")
    axes[3].set_ylabel("Precision")
    axes[3].legend(loc="lower left", fontsize="small")

    plt.suptitle(f"Split Analysis: {split_name}", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    plt.savefig(
        os.path.join(output_dir, f"full_diag_{split_name}.png"),
        dpi=150
    )
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


def build_ensemble(csv_files, output_dir):
    dfs = []

    for i, f in enumerate(csv_files):
        df = pd.read_csv(f)

        score_col = [c for c in df.columns if c not in ["video", "label", "grade"]][0]

        df = df[["video", "label", "grade", score_col]].rename(
            columns={score_col: f"score_{i}"}
        )

        dfs.append(df)

    merged = dfs[0]

    for df in dfs[1:]:
        merged = merged.merge(df, on=["video", "label", "grade"])

    score_cols = [c for c in merged.columns if c.startswith("score_")]

    merged["ensemble_score"] = merged[score_cols].mean(axis=1)

    out_path = os.path.join(output_dir, "ensemble.csv")

    merged[["video", "label", "grade", "ensemble_score"]].to_csv(
        out_path,
        index=False
    )

    return out_path


def main():
    INPUT_FOLDER = "."
    OUTPUT_FOLDER = "split_analysis_results"

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    csv_files = sorted(glob.glob(os.path.join(INPUT_FOLDER, "*.csv")))

    if not csv_files:
        print("No CSV files found.")
        return

    summary = []

    # Individual splits
    for f in csv_files:
        metrics = generate_analysis(f, OUTPUT_FOLDER)
        summary.append(metrics)
        print(f"Processed {metrics['Split']}")

    # Ensemble (6th plot)
    ensemble_csv = build_ensemble(csv_files, OUTPUT_FOLDER)

    ensemble_metrics = generate_analysis(ensemble_csv, OUTPUT_FOLDER)
    ensemble_metrics["Split"] = "ENSEMBLE"

    summary.append(ensemble_metrics)
    print("Processed ENSEMBLE")

    # Save summary
    summary_df = pd.DataFrame(summary)

    summary_df.to_csv(
        os.path.join(OUTPUT_FOLDER, "analysis_summary.csv"),
        index=False
    )

    print("\n=== Summary Stats ===")
    print(summary_df[["Split", "F1", "ROC_AUC_Direct", "PR_AUC_Direct"]].to_string(index=False))


if __name__ == "__main__":
    main()