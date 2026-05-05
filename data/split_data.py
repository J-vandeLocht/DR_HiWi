import json
import os
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import StratifiedKFold


def create_kfold_splits(matching_path, clip_labels_path, n_splits=5, output_dir="stratified_splits", random_seed=42):
    # 1. Setup Environment
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 2. Load data
    with open(matching_path, "r") as f:
        data = json.load(f)
        matches = data.get("matches", data)

    with open(clip_labels_path, "r") as f:
        clip_labels = json.load(f)

    # 3. Sync IDs and prepare for K-Fold
    labeled_clip_ids = []
    stratify_labels = []

    for cid in matches.keys():
        clip_filename = f"{cid}.mp4"
        if clip_filename in clip_labels:
            labeled_clip_ids.append(cid)
            stratify_labels.append(clip_labels[clip_filename])

    # Convert to numpy for easy indexing
    import numpy as np
    labeled_clip_ids = np.array(labeled_clip_ids)
    stratify_labels = np.array(stratify_labels)

    # 4. Initialize K-Fold and Stats Tracking
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    stats_log = []

    print(f"--- Starting {n_splits}-Fold Stratified Split ---")

    for fold, (train_idx, val_idx) in enumerate(skf.split(labeled_clip_ids, stratify_labels), 1):
        fold_dir = os.path.join(output_dir, f"split_{fold}")
        os.makedirs(fold_dir, exist_ok=True)

        train_ids, val_ids = labeled_clip_ids[train_idx], labeled_clip_ids[val_idx]

        # Structures for this fold
        structures = {
            "mil_train.json": {}, "mil_val.json": {},
            "frame_train.json": {}, "frame_val.json": {}
        }

        # Helper to populate and calculate binary distribution
        # Assuming Binary 1 (Referable) is Grade >= 2, else 0
        def process_subset(ids, clip_dict, image_dict):
            grades = []
            for cid in ids:
                filename = f"{cid}.mp4"
                grade = clip_labels[filename]
                grades.append(grade)
                clip_dict[filename] = grade
                if cid in matches:
                    for img_name in matches[cid]:
                        image_dict[img_name] = grade
            return grades

        train_grades = process_subset(train_ids, structures["mil_train.json"], structures["frame_train.json"])
        val_grades = process_subset(val_ids, structures["mil_val.json"], structures["frame_val.json"])

        # Save the 4 JSON files for this fold
        for filename, content in structures.items():
            with open(os.path.join(fold_dir, filename), "w") as f:
                json.dump(content, f, indent=4)

        # 5. Collect Metadata/Stats
        for name, g_list in [("train", train_grades), ("val", val_grades)]:
            counts = Counter(g_list)
            binary_counts = Counter([1 if g >= 2 else 0 for g in g_list])

            entry = {
                "fold": fold,
                "set": name,
                "total": len(g_list),
                **{f"grade_{i}": counts.get(i, 0) for i in range(5)},
                "binary_0": binary_counts.get(0, 0),
                "binary_1": binary_counts.get(1, 0)
            }
            stats_log.append(entry)

    # 6. Save Metadata CSV
    df_stats = pd.DataFrame(stats_log)
    df_stats.to_csv(os.path.join(output_dir, "split_metadata.csv"), index=False)

    # 7. Generate Distribution Plot
    generate_plots(df_stats, output_dir)

    print(f"Success! Splits and metadata saved to: {output_dir}")


def generate_plots(df, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: 0-4 Grades (Train sets only for clarity, or can be adjusted)
    df_train = df[df['set'] == 'train']
    grade_cols = [f"grade_{i}" for i in range(5)]
    df_train.set_index('fold')[grade_cols].plot(kind='bar', stacked=True, ax=axes[0])
    axes[0].set_title("Grade Distribution (0-4) per Fold (Train)")
    axes[0].set_ylabel("Count")

    # Plot 2: Binary Distribution (Train sets)
    df_train.set_index('fold')[['binary_0', 'binary_1']].plot(kind='bar', ax=axes[1])
    axes[1].set_title("Binary Distribution (Referable vs Not) per Fold")
    axes[1].set_ylabel("Count")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "distribution_plot.png"))
    plt.close()


if __name__ == '__main__':
    create_kfold_splits("matching_results_hd.json", "final_clip_labels_hd.json")