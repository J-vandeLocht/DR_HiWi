import json
import os
from sklearn.model_selection import train_test_split


def create_unified_splits(matching_path, clip_labels_path, val_size=0.2, random_seed=42):
    # 1. Load data
    with open(matching_path, "r") as f:
        data = json.load(f)
        matches = data.get("matches", data)  # Handle if "matches" key exists or if it's raw dict

    with open(clip_labels_path, "r") as f:
        clip_labels = json.load(f)

    # 2. Sync IDs: Only use IDs present in BOTH labels and matches
    # matches keys = "12345", clip_labels keys = "12345.mp4"
    labeled_clip_ids = []
    stratify_labels = []

    for cid in matches.keys():
        clip_filename = f"{cid}.mp4"
        if clip_filename in clip_labels:
            labeled_clip_ids.append(cid)
            # --- THE FIX: Use the raw Grade (0-4) for stratification ---
            stratify_labels.append(clip_labels[clip_filename])

    # 3. Perform the Split (Stratified by Grade 0-4)
    # This ensures Grade 3s and 4s are evenly split between Train/Val
    try:
        train_ids, val_ids = train_test_split(
            labeled_clip_ids,
            test_size=val_size,
            stratify=stratify_labels,  # Stratifying on raw grades now
            random_state=random_seed
        )
    except ValueError as e:
        print(f"Stratification failed (likely a class with only 1 member): {e}")
        print("Falling back to random split.")
        train_ids, val_ids = train_test_split(
            labeled_clip_ids,
            test_size=val_size,
            random_state=random_seed
        )

    # 4. Create the 4 JSON structures
    # We still SAVE the binary label (Referable/Non-Referable) for training if that's what you need,
    # OR we can save the Grade. Usually, for binary training, you convert at runtime.
    # Below, I save the RAW GRADE so you have flexibility later.

    train_clips, val_clips = {}, {}
    train_images, val_images = {}, {}

    # Helper to map IDs to Dicts
    def populate_splits(id_list, clip_dict, image_dict):
        for cid in id_list:
            clip_filename = f"{cid}.mp4"
            grade = clip_labels[clip_filename]

            # Save Clip Label
            clip_dict[clip_filename] = grade

            # Save Image Labels (All frames inherit the clip's grade)
            if cid in matches:
                for img_name in matches[cid]:
                    image_dict[img_name] = grade

    populate_splits(train_ids, train_clips, train_images)
    populate_splits(val_ids, val_clips, val_images)

    # 5. Save all 4 files
    output_files = {
        "mil_train.json": train_clips,
        "mil_val.json": val_clips,
        "frame_train.json": train_images,
        "frame_val.json": val_images
    }

    for filename, content in output_files.items():
        with open(filename, "w") as f:
            json.dump(content, f, indent=4)

    print(f"--- Unified Split Complete (Stratified by Original Grades) ---")
    print(f"Total Patients: {len(labeled_clip_ids)}")
    print(f"Training: {len(train_ids)} patients")
    print(f"Validation: {len(val_ids)} patients")

    # Print distribution check
    from collections import Counter
    train_grades = [clip_labels[f"{cid}.mp4"] for cid in train_ids]
    val_grades = [clip_labels[f"{cid}.mp4"] for cid in val_ids]
    print(f"Train Grade Dist: {dict(Counter(train_grades))}")
    print(f"Val Grade Dist:   {dict(Counter(val_grades))}")

    return output_files.keys()


if __name__ == '__main__':
    # Update paths if necessary
    create_unified_splits("matching_results_hd.json", "final_clip_labels_hd.json")