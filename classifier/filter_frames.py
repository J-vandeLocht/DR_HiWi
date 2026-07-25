"""
Removes previously-saved frames whose confidence score is below a threshold.

Expects frame files produced by sample_frames.py, named like:
    {video_stem}_frame_{frame_idx}_p{prob}.png

Per video, keeps every frame with prob >= min_confidence. If that would
remove ALL of a video's frames, keeps the single highest-confidence frame
instead, so no video ever ends up with zero frames.

Defaults to a dry run (prints what would be deleted, deletes nothing).
Pass --no-dry-run to actually delete.
"""

import argparse
import re
from pathlib import Path

OUTPUT_DIR = Path('classifier/frames_sampled')
MIN_CONFIDENCE = 0.95

# Matches filenames produced by sample_frames.py, e.g.
# "videoA_frame_142_p0.873.png" -> stem="videoA", frame_idx=142, prob=0.873
FRAME_FILENAME_RE = re.compile(r"^(?P<stem>.+)_frame_(?P<frame_idx>\d+)_p(?P<prob>[0-9.]+)\.png$")


def scan_output_frames(output_dir):
    """
    Scans output_dir for saved frame files and groups them by video stem.
    Returns dict: video_stem -> list of (path, frame_idx, prob).
    Files that don't match the expected naming pattern are skipped with a
    warning (e.g. leftover files from a different run/convention).
    """
    groups = {}
    for path in sorted(output_dir.glob("*.png")):
        m = FRAME_FILENAME_RE.match(path.name)
        if not m:
            print(f"  Skipping unrecognized file (doesn't match expected pattern): {path.name}")
            continue
        stem = m.group("stem")
        frame_idx = int(m.group("frame_idx"))
        prob = float(m.group("prob"))
        groups.setdefault(stem, []).append((path, frame_idx, prob))
    return groups


def cleanup_low_confidence_frames(output_dir, min_confidence, dry_run=True):
    """
    Removes saved frames with prob < min_confidence, per video, while always
    keeping at least one frame per video (the highest-confidence one, if all
    of that video's frames fall below the threshold).
    """
    print(f"Scanning existing frames in: {output_dir.resolve()}")
    groups = scan_output_frames(output_dir)

    if not groups:
        print("No recognized frame files found. Nothing to do.")
        return

    total_to_delete = 0
    total_kept = 0

    for stem, frames in sorted(groups.items()):
        frames_sorted_by_prob = sorted(frames, key=lambda x: x[2], reverse=True)
        below_threshold = [f for f in frames_sorted_by_prob if f[2] < min_confidence]
        above_threshold = [f for f in frames_sorted_by_prob if f[2] >= min_confidence]

        if above_threshold:
            # Normal case: keep everything >= threshold, delete the rest.
            to_delete = below_threshold
        else:
            # All frames for this video are below threshold: keep the single
            # best one, delete the rest, so every video retains >= 1 frame.
            to_delete = below_threshold[1:]
            kept_fallback = below_threshold[0]
            print(f"[{stem}] All {len(frames)} frame(s) below {min_confidence} -- "
                  f"keeping best one: frame {kept_fallback[1]} (p={kept_fallback[2]:.3f})")

        kept_count = len(frames) - len(to_delete)
        total_kept += kept_count
        total_to_delete += len(to_delete)

        if to_delete:
            action = "Would delete" if dry_run else "Deleting"
            for path, frame_idx, prob in sorted(to_delete, key=lambda x: x[1]):
                print(f"  [{stem}] {action}: frame {frame_idx} (p={prob:.3f}) -> {path.name}")
                if not dry_run:
                    path.unlink()

    print("\n" + "=" * 60)
    mode_str = "DRY RUN -- no files deleted" if dry_run else "Deletion complete"
    print(f"{mode_str}. {total_to_delete} frame(s) would be/were removed, "
          f"{total_kept} frame(s) kept, across {len(groups)} video(s).")
    if dry_run:
        print("Re-run with --no-dry-run to actually delete these files.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Remove low-confidence frames from the sampled-frames output folder."
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR,
                         help=f"Folder containing the saved frame PNGs (default: {OUTPUT_DIR}).")
    parser.add_argument("--min-confidence", type=float, default=MIN_CONFIDENCE,
                         help=f"Frames with prob below this are removed (default: {MIN_CONFIDENCE}).")
    parser.add_argument("--no-dry-run", action="store_true",
                         help="Actually delete files. Without this flag, only prints what would be deleted.")
    args = parser.parse_args()

    cleanup_low_confidence_frames(
        args.output_dir, min_confidence=args.min_confidence, dry_run=not args.no_dry_run
    )