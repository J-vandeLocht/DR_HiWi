import json
import csv
import os
import re

matching_file = "data/matching_results_hd_new.json"
csv_path = "data/DR_Grading_Summary_v3.csv"


def normalize_string(s):
    # Cleans a string to its core 'identity' for matching.
    if not s: return ""
    # 1. Remove "CLEAN_" prefix if it exists
    s = re.sub(r'^CLEAN_', '', s, flags=re.IGNORECASE)
    # 2. Get just the filename (handles both \ and /)
    s = s.replace('\\', '/').split('/')[-1]
    # 3. Strip extension
    s = os.path.splitext(s)[0]
    # 4. Lowercase and remove all non-alphanumeric (keeps only IMG2708LEHEALTHY)
    # This makes "IMG_2708" match "IMG 2708" or "IMG2708"
    s = re.sub(r'[^a-zA-Z0-9]', '', s).lower()
    return s


# 1. Load the matches
with open(matching_file, "r") as f:
    matches_data = json.load(f)["matches"]

# 2. Load CSV grades into a dict with normalized keys
# We store: { 'img2708lehealthy': grade }
csv_grades = {}
with open(csv_path, mode='r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        path_raw = row['Path'].strip()
        normalized_key = normalize_string(path_raw)
        csv_grades[normalized_key] = int(row['Grading'])

# 3. Create the two label maps
final_clip_labels = {}
final_image_labels = {}

for clip_id, image_list in matches_data.items():
    # Normalize the ID from the JSON
    query_id = normalize_string(clip_id)

    # Direct lookup in our normalized dictionary
    if query_id in csv_grades:
        grade = csv_grades[query_id]
        final_clip_labels[f"{clip_id}.mp4"] = grade
        for img_name in image_list:
            final_image_labels[img_name] = grade
    else:
        # Fallback: if total normalization was too aggressive, try substring
        found_fallback = False
        for path_key, grade in csv_grades.items():
            if query_id in path_key or path_key in query_id:
                final_clip_labels[f"{clip_id}.mp4"] = grade
                for img_name in image_list:
                    final_image_labels[img_name] = grade
                found_fallback = True
                break

        if not found_fallback:
            print(f"Warning: No grade found for: {clip_id} (Normalized as: {query_id})")

# 4. Save
with open("data/final_clip_labels_hd.json", "w") as f:
    json.dump(final_clip_labels, f, indent=4)

with open("data/final_image_labels_hd.json", "w") as f:
    json.dump(final_image_labels, f, indent=4)

print(f"\nSuccess!")
print(f"Mapped {len(final_clip_labels)} clips and {len(final_image_labels)} images.")