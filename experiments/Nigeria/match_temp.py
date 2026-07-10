import pandas as pd

for split in [1, 2, 3, 4, 5]:
    # Load files
    grading = pd.read_csv("Grading_Final.csv")
    predictions = pd.read_csv(f"mil_multi_eval/predictions_split_{split}.csv")

    # Keep only the columns needed for the merge
    grading_subset = grading[["video_name", "grade"]]

    # Merge and add the grade column
    predictions_augmented = predictions.merge(
        grading_subset,
        left_on="video",
        right_on="video_name",
        how="left"
    )

    # Remove the extra merge key column if desired
    predictions_augmented = predictions_augmented.drop(columns=["video_name"])

    # Save result
    predictions_augmented.to_csv(
        f"predictions_split_{split}.csv",
        index=False
    )

    print(f"Added grades to {len(predictions_augmented)} rows.")
    print(f"Rows with missing grades: {predictions_augmented['grade'].isna().sum()}")