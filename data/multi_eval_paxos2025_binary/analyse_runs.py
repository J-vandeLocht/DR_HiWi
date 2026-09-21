import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import re
from collections import defaultdict
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
    thresh_idx = min(best_idx, len(thresholds) - 1)
    return thresholds[thresh_idx], f1_scores[best_idx]


def get_video_frame_stats(video_name, txt_dir="../ensemble_results_paxos2025/txt_files"):
    """
    Parses frame-level txt file to extract:
      1. Count of informative frames
      2. Mean probability among informative frames
    """
    txt_filename = os.path.splitext(str(video_name))[0] + ".txt"
    txt_path = os.path.join(txt_dir, txt_filename)

    if not os.path.exists(txt_path):
        return 0, 0.0

    try:
        # Load space/tab-separated txt file with columns: frame, prob, informative
        txt_df = pd.read_csv(txt_path, sep=r"\s+")

        # Filter for informative frames
        inf_df = txt_df[txt_df["informative"] == True]

        num_inf = len(inf_df)
        mean_score = inf_df["prob"].mean() if num_inf > 0 else 0.0

        return num_inf, mean_score
    except Exception:
        return 0, 0.0


def generate_analysis(df, split_name, score_col, model_name, output_dir):
    y_true = df["label"].values.astype(int)
    y_probs = df[score_col].values.astype(float)

    roc_auc_direct = roc_auc_score(y_true, y_probs)
    pr_auc_direct = average_precision_score(y_true, y_probs)

    fpr, tpr, _ = roc_curve(y_true, y_probs)
    prec, rec, _ = precision_recall_curve(y_true, y_probs)
    roc_auc_integral = auc(fpr, tpr)
    pr_auc_integral = auc(rec, prec)

    best_thresh, best_f1 = find_best_threshold(y_true, y_probs)

    # 1 row, 6 columns layout
    fig, axes = plt.subplots(1, 6, figsize=(36, 5))

    # Identify videos & predictions
    sort_idx = np.argsort(-y_probs)
    y_pred = (y_probs >= best_thresh).astype(int)
    is_correct = (y_true == y_pred)

    video_col = "video" if "video" in df.columns else df.columns[0]
    sorted_videos = df[video_col].values[sort_idx]
    sorted_correct = is_correct[sort_idx]

    grade_colors = {
        0: "green",
        1: "yellow",
        2: "orange",
        3: "red",
        4: "black"
    }
    grade_col = "grade" if "grade" in df.columns else "label"
    grades_sorted = df[grade_col].values[sort_idx]
    colors = [grade_colors.get(g, "gray") for g in grades_sorted]

    # Panel 1: Ranking
    axes[0].scatter(
        range(len(y_probs)),
        y_probs[sort_idx],
        c=colors,
        s=15,
        alpha=0.8,
        edgecolors="black",
        linewidths=0.5
    )
    axes[0].axhline(
        best_thresh,
        color="black",
        linestyle="--",
        label=f"Thresh: {best_thresh:.2f}"
    )
    axes[0].set_title(f"Ranking\nF1: {best_f1:.3f}")
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].set_ylabel("Score")
    axes[0].legend()

    # Extract informativeness metrics for videos (ordered by ranking score)
    inf_counts = []
    mean_scores = []
    bar_colors = []

    for vid, correct in zip(sorted_videos, sorted_correct):
        cnt, mean_s = get_video_frame_stats(vid)
        inf_counts.append(cnt)
        mean_scores.append(mean_s)
        # Green for correctly classified, Red for wrongly classified
        bar_colors.append("#2ecc71" if correct else "#e74c3c")

    x_indices = np.arange(len(sorted_videos))

    # Panel 2: Count of Informative Frames per Video
    axes[1].bar(x_indices, inf_counts, color=bar_colors, width=0.8, edgecolor="none")
    axes[1].set_title("Informative Frame Count\n(Green = Correct, Red = Wrong)")
    axes[1].set_xlabel("Video Index (Ranked)")
    axes[1].set_ylabel("Frame Count")

    # Panel 3: Mean Informativeness Score per Video
    axes[2].bar(x_indices, mean_scores, color=bar_colors, width=0.8, edgecolor="none")
    axes[2].set_title("Mean Informative Frame Score\n(Green = Correct, Red = Wrong)")
    axes[2].set_xlabel("Video Index (Ranked)")
    axes[2].set_ylabel("Mean Prob Score")
    axes[2].set_ylim(-0.05, 1.05)

    # Panel 4: Confusion Matrix
    sns.heatmap(
        confusion_matrix(y_true, y_pred),
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=axes[3],
        cbar=False
    )
    axes[3].set_title("CM @ Opt. Thresh")
    axes[3].set_xlabel("Predicted")
    axes[3].set_ylabel("Actual")

    # Panel 5: ROC Curve
    axes[4].plot(
        fpr,
        tpr,
        color="darkorange",
        lw=2,
        label=(
            f"Direct: {roc_auc_direct:.3f}\n"
            f"Integral: {roc_auc_integral:.3f}"
        )
    )
    axes[4].plot(
        [0, 1],
        [0, 1],
        color="navy",
        linestyle="--"
    )
    axes[4].set_title("ROC Curve")
    axes[4].set_xlabel("FPR")
    axes[4].set_ylabel("TPR")
    axes[4].legend(loc="lower right", fontsize="small")

    # Panel 6: Precision-Recall Curve
    axes[5].plot(
        rec,
        prec,
        color="green",
        lw=2,
        label=(
            f"Direct: {pr_auc_direct:.3f}\n"
            f"Integral: {pr_auc_integral:.3f}"
        )
    )
    axes[5].set_title("PR Curve")
    axes[5].set_xlabel("Recall")
    axes[5].set_ylabel("Precision")
    axes[5].legend(loc="lower left", fontsize="small")

    plt.suptitle(
        f"Split Analysis: {split_name} | Model: {model_name}",
        fontsize=16
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    save_path = os.path.join(
        output_dir,
        f"full_diag_{split_name}_{model_name.replace(' ', '_')}.png"
    )
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Extract wrongly classified video list
    wrong_videos = list(df[video_col].values[~is_correct])

    metrics = {
        "Split": split_name,
        "Model": model_name,
        "Best_Thresh": best_thresh,
        "F1": best_f1,
        "ROC_AUC_Direct": roc_auc_direct,
        "ROC_AUC_Integral": roc_auc_integral,
        "PR_AUC_Direct": pr_auc_direct,
        "PR_AUC_Integral": pr_auc_integral
    }

    return metrics, wrong_videos


def print_cross_model_error_analysis(misclassified_tracker):
    """
    Analyzes and prints misclassified videos across all models per split.
    """
    print("\n" + "=" * 50)
    print("      CROSS-MODEL MISCLASSIFICATION ANALYSIS      ")
    print("=" * 50)

    for split_name, model_errors in misclassified_tracker.items():
        print(f"\n--- {split_name.upper()} ---")

        # Count errors per video across all models in this split
        video_error_counts = defaultdict(list)
        for model_name, wrong_vids in model_errors.items():
            for v in wrong_vids:
                video_error_counts[v].append(model_name)

        if not video_error_counts:
            print("No errors found across any models.")
            continue

        # Group videos by the number of models that misclassified them
        multi_model_errors = {
            vid: models for vid, models in video_error_counts.items() if len(models) > 1
        }
        single_model_errors = {
            vid: models for vid, models in video_error_counts.items() if len(models) == 1
        }

        print(f"Total Unique Wrong Videos: {len(video_error_counts)}")
        print(f"Videos Wrong by MULTIPLE Models ({len(multi_model_errors)}):")

        if multi_model_errors:
            for vid, models in sorted(multi_model_errors.items(), key=lambda x: len(x[1]), reverse=True):
                print(f"  • {vid:<15} -> Failed in {len(models)}/3 models: {', '.join(models)}")
        else:
            print("  None! No video was misclassified by more than one model.")

        print(f"\nVideos Wrong by ONLY ONE Model ({len(single_model_errors)}):")
        if single_model_errors:
            for vid, models in sorted(single_model_errors.items()):
                print(f"  • {vid:<15} -> Failed in: {models[0]}")
        else:
            print("  None!")


def plot_metric_summary(summary_df, output_dir):
    metrics_to_plot = [
        ("F1", "Optimal F1 Score"),
        ("ROC_AUC_Direct", "ROC-AUC"),
        ("PR_AUC_Direct", "PR-AUC"),
        ("Best_Thresh", "Optimal Threshold")
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    sns.set_theme(style="whitegrid")

    model_order = [
        "Classifier Frame",
        "Classifier Video",
        "Transformer"
    ]

    for idx, (metric, title) in enumerate(metrics_to_plot):
        if metric not in summary_df.columns:
            continue

        ax = axes[idx]

        sns.barplot(
            data=summary_df,
            x="Model",
            y=metric,
            hue="Model",
            order=model_order,
            hue_order=model_order,
            ax=ax,
            palette="Set2",
            width=0.5,
            alpha=0.7,
            legend=False,
            errorbar="sd",
            capsize=0.15,
            err_kws={"linewidth": 1.5, "color": "#333333"}
        )

        sns.stripplot(
            data=summary_df,
            x="Model",
            y=metric,
            order=model_order,
            ax=ax,
            color="#111111",
            jitter=0.08,
            size=7,
            alpha=0.9,
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

        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("Score")

    plt.suptitle(
        "Binary Performance Across Splits (Mean ± SD)",
        fontsize=15,
        fontweight="bold",
        y=0.98
    )
    plt.tight_layout()

    save_path = os.path.join(
        output_dir,
        "model_comparison_summary.png"
    )
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSummary plot saved to: {save_path}")


def main():
    INPUT_FOLDER = "."
    BASE_OUTPUT_FOLDER = "split_analysis_results"

    FRAME_DIR = os.path.join(BASE_OUTPUT_FOLDER, "Classifier_Frame")
    VIDEO_DIR = os.path.join(BASE_OUTPUT_FOLDER, "Classifier_Video")
    TRANS_DIR = os.path.join(BASE_OUTPUT_FOLDER, "Transformer")

    os.makedirs(FRAME_DIR, exist_ok=True)
    os.makedirs(VIDEO_DIR, exist_ok=True)
    os.makedirs(TRANS_DIR, exist_ok=True)

    csv_files = sorted(
        glob.glob(os.path.join(INPUT_FOLDER, "*.csv"))
    )

    if not csv_files:
        print(f"No CSV files found in {INPUT_FOLDER}.")
        return

    summary_list = []
    # Nested dict structure: misclassified_tracker[split_name][model_name] = [wrong_video_names]
    misclassified_tracker = defaultdict(dict)

    model_configs = [
        ("Clf_frame", "Classifier Frame", FRAME_DIR),
        ("Clf_video", "Classifier Video", VIDEO_DIR),
        ("Trans", "Transformer", TRANS_DIR)
    ]

    for f in csv_files:
        if "summary" in os.path.basename(f):
            continue

        df = pd.read_csv(f)
        split_name = os.path.basename(f).replace(".csv", "")

        # Extract split number from filename, e.g. split_1
        match = re.search(r"split_(\d+)", split_name)
        if not match:
            print(f"Could not determine split number from {split_name}")
            continue

        split_num = match.group(1)
        processed_models = []

        for prefix, model_name, output_dir in model_configs:
            score_col = f"{prefix}_split_{split_num}"

            if score_col not in df.columns:
                print(
                    f"Missing column '{score_col}' "
                    f"for {model_name} in {split_name}"
                )
                continue

            metrics, wrong_vids = generate_analysis(
                df,
                split_name,
                score_col,
                model_name,
                output_dir
            )

            summary_list.append(metrics)
            misclassified_tracker[split_name][model_name] = wrong_vids
            processed_models.append(model_name)

        if processed_models:
            print(
                f"Processed {split_name} for models: "
                f"{', '.join(processed_models)}"
            )

    if not summary_list:
        print("No valid metrics were computed.")
        return

    summary_df = pd.DataFrame(summary_list)
    summary_df = summary_df.sort_values(
        by=["Split", "Model"]
    )

    summary_df.to_csv(
        os.path.join(
            BASE_OUTPUT_FOLDER,
            "analysis_summary.csv"
        ),
        index=False
    )

    plot_metric_summary(
        summary_df,
        BASE_OUTPUT_FOLDER
    )

    # Output cross-model error overview
    print_cross_model_error_analysis(misclassified_tracker)

    print("\n=== Summary Stats ===")
    cols_to_show = [
        "Split",
        "Model",
        "F1",
        "ROC_AUC_Direct",
        "PR_AUC_Direct"
    ]
    print(
        summary_df[cols_to_show].to_string(index=False)
    )


if __name__ == "__main__":
    main()