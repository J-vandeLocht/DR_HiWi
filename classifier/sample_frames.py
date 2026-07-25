"""
Sample 1-4 informative, well-spaced frames per video for classifier training.

For each video, we have a companion txt file with a per-frame confidence
score (`prob`). We select frames as follows:

1. Take all frames with prob >= CONFIDENCE_THRESHOLD as candidates.
2. Greedily pick the highest-confidence candidate, then suppress (remove)
   all other candidates within MIN_DISTANCE_FRACTION * total_frames of it.
3. Repeat until either no candidates remain or MAX_FRAMES have been picked.
4. If no frame ever clears the threshold, fall back to the single
   highest-confidence frame in the whole video.

Selected frames are padded to a square (black padding, no stretching) and
saved as PNGs, mirroring the original annotation-extraction script.
"""

import cv2
import argparse
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
VIDEO_DIRS = [Path('data/dr_videos'), Path('data/Paxos_2020_videos')]
SCORES_DIRS = [Path('data/ensemble_results/txt_files'), Path('data/ensemble_results_paxos2020/txt_files')]  # folders containing the {video_stem}.txt files
OUTPUT_DIR = Path('classifier/frames_sampled')

CONFIDENCE_THRESHOLD = 0.95
MIN_DISTANCE_FRACTION = 0.25   # min gap between picks, as fraction of total_frames
MAX_FRAMES = 4
VIDEO_EXTS = ['*.mp4', '*.mov', '*.MP4', '*.MOV']

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def pad_to_square(image):
    """Adds black padding to make the image 1:1 without stretching."""
    h, w = image.shape[:2]
    max_side = max(h, w)
    squared_img = np.zeros((max_side, max_side, 3), dtype=np.uint8)

    y_offset = (max_side - h) // 2
    x_offset = (max_side - w) // 2

    squared_img[y_offset:y_offset + h, x_offset:x_offset + w] = image
    return squared_img


def find_scores_file(video_stem, scores_dirs):
    """Looks for {video_stem}.txt across all provided scores directories.
    Returns the Path if found (first match wins), else None."""
    for d in scores_dirs:
        candidate = d / f"{video_stem}.txt"
        if candidate.exists():
            return candidate
    return None


def parse_scores_file(scores_path):
    """
    Parses a txt file structured as:
        frame   prob   informative
        0   0.007791206240653992   False
        ...
    Returns a list of (frame_idx: int, prob: float) tuples.
    """
    entries = []
    with open(scores_path, 'r') as f:
        lines = f.readlines()

    # Skip header line (first line, non-numeric first token)
    start_idx = 0
    if lines and not lines[0].strip().split()[0].isdigit():
        start_idx = 1

    for line in lines[start_idx:]:
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        frame_idx = int(parts[0])
        prob = float(parts[1])
        entries.append((frame_idx, prob))

    return entries


def select_frames_greedy(entries, total_frames, threshold, min_dist_fraction, max_frames):
    """
    Greedy peak-picking with suppression.

    entries: list of (frame_idx, prob)
    Returns: list of (frame_idx, prob), sorted by frame_idx, and a bool
             indicating whether the threshold-based path was used
             (False => fallback to single global max frame).
    """
    min_distance = min_dist_fraction * total_frames

    candidates = [(f, p) for f, p in entries if p >= threshold]

    if not candidates:
        # Fallback: single highest-confidence frame overall
        best = max(entries, key=lambda x: x[1])
        return [best], False

    pool = candidates.copy()
    selected = []

    while pool and len(selected) < max_frames:
        # Pick current global max in remaining pool
        pick = max(pool, key=lambda x: x[1])
        selected.append(pick)

        # Suppress all candidates within min_distance of the pick
        pool = [
            (f, p) for (f, p) in pool
            if abs(f - pick[0]) >= min_distance
        ]

    selected.sort(key=lambda x: x[0])
    return selected, True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def extract_sampled_frames(num_gpus=1, gpu_id=0):
    print(f"Searching for videos in: {[str(d.resolve()) for d in VIDEO_DIRS]}")
    video_files = []
    for d in VIDEO_DIRS:
        for ext in VIDEO_EXTS:
            video_files.extend(d.glob(ext))
    video_files = sorted(set(video_files))

    if not video_files:
        print(f"Error: No videos found in {VIDEO_DIRS}")
        return

    # Shard videos across GPUs: simple round-robin indexing, e.g. for
    # num_gpus=4, gpu_id=0 gets videos [0, 4, 8, ...], gpu_id=1 gets
    # [1, 5, 9, ...], etc. Sorting above ensures every GPU sees the same
    # ordering before slicing, so the split is deterministic and disjoint.
    video_files = video_files[gpu_id::num_gpus]
    print(f"GPU {gpu_id}/{num_gpus - 1}: assigned {len(video_files)} video(s).")

    if not video_files:
        print(f"GPU {gpu_id}: no videos assigned, exiting.")
        return

    summary = []
    n_mismatched = 0

    for video_path in video_files:
        scores_path = find_scores_file(video_path.stem, SCORES_DIRS)
        if scores_path is None:
            print(f"[{video_path.name}] Skipping: no scores file found in {SCORES_DIRS}")
            continue

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"[{video_path.name}] Skipping: error opening video.")
            continue

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            print(f"[{video_path.name}] Skipping: invalid frame count.")
            cap.release()
            continue

        entries = parse_scores_file(scores_path)
        if not entries:
            print(f"[{video_path.name}] Skipping: scores file empty or unparsable.")
            cap.release()
            continue

        # Sanity check: does CAP_PROP_FRAME_COUNT match the number of scored frames?
        n_entries = len(entries)
        if n_entries != total_frames:
            n_mismatched += 1
            print(f"[{video_path.name}] WARNING: frame count mismatch -- "
                  f"cv2 reports {total_frames} frames, scores file has {n_entries} entries. "
                  f"Using cv2's count ({total_frames}) for spacing/extraction; "
                  f"double-check this video/scores pair.")

        selected, used_threshold = select_frames_greedy(
            entries, total_frames, CONFIDENCE_THRESHOLD, MIN_DISTANCE_FRACTION, MAX_FRAMES
        )

        mode = "threshold" if used_threshold else "fallback (max-confidence only)"
        print(f"\n[GPU {gpu_id}] Processing: {video_path.name} ({total_frames} frames) -- mode: {mode}")
        print(f"  Selected {len(selected)} frame(s): "
              + ", ".join(f"[{f} | p={p:.3f}]" for f, p in selected))

        targets = {f: p for f, p in selected}
        max_target = max(targets.keys())

        current_idx = 0
        saved_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if current_idx in targets:
                prob = targets[current_idx]
                square_frame = pad_to_square(frame)
                filename = f"{video_path.stem}_frame_{current_idx}_p{prob:.3f}.png"
                save_path = OUTPUT_DIR / filename

                cv2.imwrite(str(save_path), square_frame, [cv2.IMWRITE_PNG_COMPRESSION, 3])
                print(f"  -> Saved frame {current_idx} (prob={prob:.3f}) -> {filename}")
                saved_count += 1

            current_idx += 1
            if current_idx > max_target:
                break

        cap.release()

        if saved_count != len(selected):
            print(f"  WARNING: expected to save {len(selected)} frames but only saved "
                  f"{saved_count}. Some target frame indices may be missing/unreadable.")

        summary.append((video_path.name, len(selected), mode))

    print("\n" + "=" * 60)
    print(f"GPU {gpu_id} done. Processed {len(summary)} video(s). "
          f"Frames saved in: '{OUTPUT_DIR.resolve()}'")
    n_fallback = sum(1 for _, _, m in summary if "fallback" in m)
    if n_fallback:
        print(f"Note: {n_fallback} video(s) used the fallback (no frame cleared "
              f"the {CONFIDENCE_THRESHOLD} threshold) -- worth reviewing these.")
    if n_mismatched:
        print(f"Note: {n_mismatched} video(s) had a mismatch between cv2 frame count "
              f"and the number of entries in the scores file -- see WARNINGs above.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample high-confidence, well-spaced frames per video.")
    parser.add_argument("--num-gpus", type=int, default=1,
                         help="Total number of GPUs/processes splitting the video list.")
    parser.add_argument("--gpu-id", type=int, default=0,
                         help="Index of this GPU/process (0-indexed, must be < num-gpus).")
    args = parser.parse_args()

    if args.gpu_id >= args.num_gpus:
        raise ValueError(f"--gpu-id ({args.gpu_id}) must be < --num-gpus ({args.num_gpus})")

    extract_sampled_frames(num_gpus=args.num_gpus, gpu_id=args.gpu_id)