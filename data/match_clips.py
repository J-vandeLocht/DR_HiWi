import os
import json
import re

FRAME_DIRS = [
    "classifier/cropped_classifier_frames",
]

# CLIP_DIRS = [
#     "data/ensemble_results_paxos2020/cleaned_videos",
# ]

CLIP_DIRS = [
    "data/ensemble_results_paxos2025/cleaned_videos",
]


def get_core_id_from_video(filename):
    # Removes extension and 'CLEAN_' prefix to get the base ID.
    name = os.path.splitext(filename)[0]
    return re.sub(r'^CLEAN_', '', name)


def get_core_id_from_image(filename):
    # Extracts the base ID from frame names produced by sample_frames.py, e.g.:
    # 'B005L_frame_2121_p0.997.png' -> 'B005L'
    # 'IMG_2708 LE HEALTHY_frame_2614_p0.980.png' -> 'IMG_2708 LE HEALTHY'
    return re.sub(r'_frame_\d+_p[0-9.]+\.png$', '', filename)


# 1. Get all files across all provided dirs
# clips/images become (filename, source_dir) pairs so we know where each file lives
clips = []
for d in CLIP_DIRS:
    clips.extend((f, d) for f in os.listdir(d) if f.endswith('.mp4'))

images = []
for d in FRAME_DIRS:
    images.extend((f, d) for f in os.listdir(d) if f.endswith('.png'))

# Sanity check: warn if the same clip/image filename shows up in more than one dir
clip_names_seen = {}
for name, d in clips:
    clip_names_seen.setdefault(name, []).append(d)
duplicate_clips = {name: dirs for name, dirs in clip_names_seen.items() if len(dirs) > 1}
if duplicate_clips:
    print(f"WARNING: {len(duplicate_clips)} clip filename(s) found in multiple CLIP_DIRS: {duplicate_clips}")

image_names_seen = {}
for name, d in images:
    image_names_seen.setdefault(name, []).append(d)
duplicate_images = {name: dirs for name, dirs in image_names_seen.items() if len(dirs) > 1}
if duplicate_images:
    print(f"WARNING: {len(duplicate_images)} image filename(s) found in multiple FRAME_DIRS: {duplicate_images}")

# 2. Map Core IDs to their full video filename
# This creates a lookup like: {"R008R2": "CLEAN_R008R2.mp4"}
video_lookup = {get_core_id_from_video(c): c for c, _ in clips}

# 3. Initialize results
matches = {c: [] for c, _ in clips}
matched_images = set()

# 4. Perform Strict Matching
for img, _ in images:
    img_core = get_core_id_from_image(img)

    # Check for an exact match in the video dictionary
    if img_core in video_lookup:
        full_clip_name = video_lookup[img_core]
        matches[full_clip_name].append(img)
        matched_images.add(img)

# 5. Format for JSON output
# We use the filename without extension as the key to stay consistent
final_matches = {os.path.splitext(k)[0]: v for k, v in matches.items() if v}

results = {
    "matches": final_matches,
    "unmatched_clips": [os.path.splitext(c)[0] for c, _ in clips if not matches[c]],
    "unmatched_images": [img for img, _ in images if img not in matched_images],
}

# 6. Save
# output_path = "data/matching_results_2020.json"
output_path = "data/matching_results_2025.json"
with open(output_path, "w") as f:
    json.dump(results, f, indent=4)

print(f"--- Matching Complete ---")
print(f"Matched Clips:  {len(results['matches'])}")
print(f"Unmatched Clips: {len(results['unmatched_clips'])}")
print(f"Unmatched Imgs:  {len(results['unmatched_images'])}")
