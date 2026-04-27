import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import (
    precision_recall_curve, f1_score, confusion_matrix,
    classification_report
)
from sklearn.calibration import calibration_curve
from matplotlib.lines import Line2D


def find_best_threshold(y_true, y_probs):
    prec, rec, thresholds = precision_recall_curve(y_true, y_probs)
    f1_scores = 2 * (prec * rec) / (prec + rec + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[min(best_idx, len(thresholds) - 1)]
    return best_threshold, f1_scores[best_idx]


def plot_diagnostic_plots(df, run_col, output_dir):
    """Generates a multi-panel diagnostic figure for a single run."""
    y_true = df['label'].values
    y_probs = df[run_col].values
    best_thresh, best_f1 = find_best_threshold(y_true, y_probs)
    y_pred = (y_probs >= best_thresh).astype(int)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # --- 1. Ranking Scatter Plot ---
    sort_idx = np.argsort(-y_probs)
    ax = axes[0]
    colors = ['#ff4d4d' if y == 1 else '#1a75ff' for y in y_true[sort_idx]]
    ax.scatter(range(len(y_probs)), y_probs[sort_idx], c=colors, s=15, alpha=0.5)
    ax.axhline(best_thresh, color='green', linestyle='--', label=f'Best Thresh: {best_thresh:.2f}')
    ax.set_title(f"Confidence Ranking\n(F1: {best_f1:.3f})")
    ax.set_ylabel("Probability")

    # --- 2. Confusion Matrix (at Best Threshold) ---
    ax = axes[1]
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
    ax.set_title(f"Confusion Matrix @ {best_thresh:.2f}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    # --- 3. Calibration Curve (Reliability Diagram) ---
    ax = axes[2]
    prob_true, prob_pred = calibration_curve(y_true, y_probs, n_bins=10)
    ax.plot(prob_pred, prob_true, marker='s', label=run_col)
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray')
    ax.set_title("Calibration Curve")
    ax.set_xlabel("Mean Predicted Prob")
    ax.set_ylabel("Fraction of Positives")

    plt.suptitle(f"Run Analysis: {run_col}", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_dir, f"diagnostic_{run_col}.png"), dpi=150)
    plt.close()


def plot_ensemble_agreement(df, run_cols, output_dir):
    """Plots how much the 4 runs agree with each other."""
    plt.figure(figsize=(10, 8))

    # Calculate variance across runs
    std_devs = df[run_cols].std(axis=1)
    means = df[run_cols].mean(axis=1)

    sc = plt.scatter(means, std_devs, c=df['label'], cmap='coolwarm', alpha=0.6, edgecolor='w')
    plt.colorbar(sc, label="True Label")
    plt.xlabel("Mean Probability (Ensemble)")
    plt.ylabel("Standard Deviation (Disagreement)")
    plt.title("Model Agreement vs. Confidence\n(High Y-axis = Models disagree on this video)")

    plt.savefig(os.path.join(output_dir, "ensemble_agreement.png"))
    plt.close()


def print_detailed_errors_to_cmd(df, run_cols):
    """
    Analyzes misclassifications using a strict 0.5 threshold
    and prints the results to the terminal.
    """
    # 1. Identify errors per run (Hard threshold 0.5)
    error_mask_df = pd.DataFrame(index=df.index)
    for col in run_cols:
        # A video is wrong if (prob > 0.5) != label
        preds = (df[col] >= 0.5).astype(int)
        error_mask_df[col] = (preds != df['label'])

    # 2. Count failures per video
    df['error_count'] = error_mask_df.sum(axis=1)

    # 3. Filter for videos with at least 1 error
    error_cases = df[df['error_count'] > 0].copy()
    error_cases = error_cases.sort_values(by=['error_count', 'video'], ascending=[False, True])

    print("\n" + "=" * 90)
    print(f"{'VIDEO NAME':<30} | {'LAB':<3} | {'ERRS':<4} | {'PROBABILITIES (Runs 1-4)':<40}")
    print("-" * 90)

    for _, row in error_cases.iterrows():
        # Format the probabilities for clean terminal viewing
        probs_str = " | ".join([f"{row[col]:.4f}" for col in run_cols])

        # Color or highlight the "Total Failures" (4/4 models wrong)
        prefix = ">>> " if row['error_count'] == len(run_cols) else "    "

        print(f"{prefix}{row['video'][:26]:<26} | {int(row['label']):<3} | {int(row['error_count']):<4} | {probs_str}")

    print("-" * 90)

    # Summary Statistics
    print(f"Total Videos Analyzed: {len(df)}")
    print(f"Videos with 0 errors: {len(df[df['error_count'] == 0])}")
    for i in range(1, len(run_cols) + 1):
        num = len(df[df['error_count'] == i])
        print(f"Videos with exactly {i} error(s): {num}")
    print("=" * 90 + "\n")


def main(csv_path, output_dir):
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    os.makedirs(output_dir, exist_ok=True)

    run_cols = [c for c in df.columns if c not in ['video', 'label']]
    summary = []

    print(f"Analyzing {len(run_cols)} runs...")

    for col in run_cols:
        # Generate the detailed 3-panel plot
        plot_diagnostic_plots(df, col, output_dir)

        # Calculate stats
        thresh, f1 = find_best_threshold(df['label'], df[col])
        summary.append({
            "Run": col,
            "Best_Threshold": round(thresh, 4),
            "Best_F1": round(f1, 4)
        })

    # Generate the Agreement Plot
    if len(run_cols) > 1:
        plot_ensemble_agreement(df, run_cols, output_dir)

    # Save summary
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(os.path.join(output_dir, "analysis_summary.csv"), index=False)

    print("\n=== Analysis Complete ===")
    print(summary_df.to_string(index=False))
    print(f"\nAll plots saved to: {output_dir}")

    print_detailed_errors_to_cmd(df, run_cols)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, default='predictions_master.csv')
    parser.add_argument('--out', type=str, default='mil_analysis_results')
    args = parser.parse_args()
    main(args.csv, args.out)