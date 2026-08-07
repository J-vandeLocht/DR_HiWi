import json
from pathlib import Path


def create_subset_json(source_json, video_dir, output_json):
    source_json = Path(source_json)
    video_dir = Path(video_dir)
    output_json = Path(output_json)

    # Load original json
    with open(source_json, "r") as f:
        data = json.load(f)

    # Get all filenames in the video directory
    video_files = {p.name for p in video_dir.iterdir() if p.is_file()}

    # Keep only matching entries
    subset = {k: v for k, v in data.items() if k in video_files}

    # Save
    with open(output_json, "w") as f:
        json.dump(subset, f, indent=4)

    print(
        f"{output_json}: kept {len(subset)} of {len(data)} entries."
    )


if __name__ == "__main__":

    for split in range(1, 6):
        jobs = [
            (
                f"split_{split}/mil_val.json",
                "../ensemble_results_paxos2020/cleaned_videos",
                f"split_{split}/mil_val_paxos2020.json",
            ),
            (
                f"split_{split}/mil_val.json",
                "../ensemble_results/cleaned_videos",
                f"split_{split}/mil_val_paxos2025.json",
            ),
        ]

        for source_json, video_dir, output_json in jobs:
            create_subset_json(source_json, video_dir, output_json)