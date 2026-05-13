import os
import json

frame_dir = "2024_Paxos_Frames/frames"
clip_dir = "clips_hd"

clips = [f for f in os.listdir(clip_dir) if f.endswith('.mp4')]
images = [f for f in os.listdir(frame_dir) if f.endswith('.png')]

clip_map = {os.path.splitext(c)[0]: [] for c in clips}
image_to_clips = {img: [] for img in images}

for img in images:
    for cid in clip_map.keys():
        if cid in img:
            clip_map[cid].append(img)
            image_to_clips[img].append(cid)

results = {
    "matches": {k: v for k, v in clip_map.items() if v},
    "unmatched_clips": [k for k, v in clip_map.items() if not v],
    "unmatched_images": [img for img, matched in image_to_clips.items() if not matched],
    "multi_matched_images": [img for img, matched in image_to_clips.items() if len(matched) > 1]
}

with open("matching_results_hd.json", "w") as f:
    json.dump(results, f, indent=4)

print(f"Matched {len(results['matches'])} clips. Found {len(results['unmatched_images'])} unmatched images.")