import os
import json
import re

frame_dir = "data/2024_Paxos_Frames/cropped_frames_new"
clip_dir = "ensemble_results/cleaned_videos"


def get_core_id_from_video(filename):
    """Removes extension and 'CLEAN_' prefix to get the base ID."""
    name = os.path.splitext(filename)[0]
    return re.sub(r'^CLEAN_', '', name)


def get_core_id_from_image(filename):
    """
    Extracts the base ID from frame names like:
    'R008R.MOV-00001.png' -> 'R008R'
    '2024_IMG_5029.mp4-00040.png' -> '2024_IMG_5029'
    """
    # Split by common video extensions and take the first part
    base = re.split(r'\.MOV|\.mp4|\.avi|\.mpeg', filename, flags=re.IGNORECASE)[0]
    return base


# 1. Get all files
clips = [f for f in os.listdir(clip_dir) if f.endswith('.mp4')]
images = [f for f in os.listdir(frame_dir) if f.endswith('.png')]

# 2. Map Core IDs to their full video filename
# This creates a lookup like: {"R008R2": "CLEAN_R008R2.mp4"}
video_lookup = {get_core_id_from_video(c): c for c in clips}

# 3. Initialize results
matches = {c: [] for c in clips}
matched_images = set()

# 4. Perform Strict Matching
for img in images:
    img_core = get_core_id_from_image(img)

    # Check for an EXACT match in our video dictionary
    if img_core in video_lookup:
        full_clip_name = video_lookup[img_core]
        matches[full_clip_name].append(img)
        matched_images.add(img)

# 5. Format for JSON output
# We use the filename without extension as the key to stay consistent
final_matches = {os.path.splitext(k)[0]: v for k, v in matches.items() if v}

results = {
    "matches": final_matches,
    "unmatched_clips": [os.path.splitext(c)[0] for c in clips if not matches[c]],
    "unmatched_images": [img for img in images if img not in matched_images]
}

# 6. Save
output_path = "data/matching_results_hd_new.json"
with open(output_path, "w") as f:
    json.dump(results, f, indent=4)

print(f"--- Matching Complete ---")
print(f"Matched Clips:  {len(results['matches'])}")
print(f"Unmatched Clips: {len(results['unmatched_clips'])}")
print(f"Unmatched Imgs:  {len(results['unmatched_images'])}")