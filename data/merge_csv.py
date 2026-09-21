import argparse
import re
from pathlib import Path

import pandas as pd


def merge_csv_folders(folder1, folder2, output_folder):
    folder1 = Path(folder1)
    folder2 = Path(folder2)
    output_folder = Path(output_folder)

    output_folder.mkdir(parents=True, exist_ok=True)

    csvs1 = {f.name: f for f in folder1.glob("*.csv")}
    csvs2 = {f.name: f for f in folder2.glob("*.csv")}

    common_files = sorted(set(csvs1) & set(csvs2))

    if len(common_files) != 5:
        raise ValueError(
            f"Expected 5 matching CSV files, found {len(common_files)}."
        )

    for filename in common_files:
        file1 = csvs1[filename]
        file2 = csvs2[filename]

        # Get split number from filename, e.g.
        # predictions_split_3_2025.csv -> 3
        match = re.search(r"split_(\d+)", filename)

        if not match:
            raise ValueError(
                f"Could not determine split number from '{filename}'"
            )

        split = match.group(1)

        print(f"Processing {filename} (split {split})")

        df_frame = pd.read_csv(file1)
        df_video = pd.read_csv(file2)

        # Columns shared between both files
        base_cols = [
            "video",
            "label",
            "grade",
        ]

        # Classifier columns -- duplicated between frame/video
        clf_cols = [
            f"Clf_pred_grade_split_{split}",
            "Clf_prob_0",
            "Clf_prob_1",
            "Clf_prob_2",
            "Clf_prob_3",
            "Clf_prob_4",
        ]

        # Transformer columns -- only one copy
        trans_cols = [
            f"Trans_pred_grade_split_{split}",
            "Trans_prob_0",
            "Trans_prob_1",
            "Trans_prob_2",
            "Trans_prob_3",
            "Trans_prob_4",
        ]

        # Make sure everything exists
        required = base_cols + clf_cols + trans_cols

        for col in required:
            if col not in df_frame.columns:
                raise ValueError(
                    f"'{col}' missing from {file1}"
                )

            if col not in df_video.columns:
                raise ValueError(
                    f"'{col}' missing from {file2}"
                )

        # Make sure shared metadata and Transformer values
        # are identical in both files
        for col in base_cols + trans_cols:
            if not df_frame[col].equals(df_video[col]):
                raise ValueError(
                    f"'{col}' differs between {file1} and {file2}"
                )

        # Rename classifier columns
        frame_rename = {
            col: col.replace("Clf_", "Clf_frame_", 1)
            for col in clf_cols
        }

        video_rename = {
            col: col.replace("Clf_", "Clf_video_", 1)
            for col in clf_cols
        }

        df_frame = df_frame.rename(columns=frame_rename)
        df_video = df_video.rename(columns=video_rename)

        frame_clf_cols = list(frame_rename.values())
        video_clf_cols = list(video_rename.values())

        # Construct output
        result = pd.concat(
            [
                df_frame[base_cols],
                df_frame[frame_clf_cols],
                df_video[video_clf_cols],
                df_frame[trans_cols],
            ],
            axis=1,
        )

        output_file = output_folder / filename
        result.to_csv(output_file, index=False)

        print(f"  -> {output_file}")

    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("folder1")
    parser.add_argument("folder2")
    parser.add_argument("output_folder")

    args = parser.parse_args()

    merge_csv_folders(
        args.folder1,
        args.folder2,
        args.output_folder,
    )