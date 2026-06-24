import json
import pandas as pd

# Load grading JSON
with open("final_clip_labels_hd.json", "r") as f:
    grading_json = json.load(f)

# Convert JSON mapping {video_name: grade} to DataFrame
grading_subset = pd.DataFrame(
    grading_json.items(),
    columns=["video_name", "grade"]
)

for split in [1, 2, 3, 4, 5]:
    # Load predictions
    predictions = pd.read_csv(
        f"predictions_split_{split}.csv"
    )

    # Merge grades into predictions
    predictions_augmented = predictions.merge(
        grading_subset,
        left_on="video",
        right_on="video_name",
        how="left"
    )

    # Remove the extra merge key column
    predictions_augmented = predictions_augmented.drop(
        columns=["video_name"]
    )

    # Save result
    predictions_augmented.to_csv(
        f"predictions_split_{split}.csv",
        index=False
    )

    print(f"Split {split}:")
    print(f"  Added grades to {len(predictions_augmented)} rows.")
    print(
        f"  Rows with missing grades: "
        f"{predictions_augmented['grade'].isna().sum()}"
    )