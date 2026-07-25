import json
import csv
import os
import re

matching_file = "data/matching_results.json"
csv_path = "data/DR_Grading_Summary_v3.csv"
graded_videos_json_path = "data/graded_videos.json"


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

# 2. Load groundtruth from BOTH sources into one combined, normalized dict.
# We track which source each key came from so we can warn on conflicts.
csv_grades = {}
grade_sources = {}  # normalized_key -> source label, for conflict reporting

# 2a. CSV source
with open(csv_path, mode='r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        path_raw = row['Path'].strip()
        normalized_key = normalize_string(path_raw)
        grade = int(row['Grading'])

        if normalized_key in csv_grades and csv_grades[normalized_key] != grade:
            print(f"WARNING: conflicting grade for {normalized_key!r} -- "
                  f"CSV says {grade}, but {grade_sources[normalized_key]} already "
                  f"set {csv_grades[normalized_key]}. Keeping the first value.")
            continue

        csv_grades[normalized_key] = grade
        grade_sources[normalized_key] = f"CSV ({path_raw!r})"

# 2b. graded_videos.json source, e.g. {"I019R.MOV": 0, ...}
with open(graded_videos_json_path, "r") as f:
    graded_videos = json.load(f)

for path_raw, grade in graded_videos.items():
    normalized_key = normalize_string(path_raw)
    grade = int(grade)

    if normalized_key in csv_grades and csv_grades[normalized_key] != grade:
        print(f"WARNING: conflicting grade for {normalized_key!r} -- "
              f"graded_videos.json says {grade}, but {grade_sources[normalized_key]} "
              f"already set {csv_grades[normalized_key]}. Keeping the first value.")
        continue

    if normalized_key not in csv_grades:
        csv_grades[normalized_key] = grade
        grade_sources[normalized_key] = f"graded_videos.json ({path_raw!r})"

print(f"Loaded {len(csv_grades)} unique groundtruth entries "
      f"(CSV + graded_videos.json combined).\n")

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
with open("data/clip_labels.json", "w") as f:
    json.dump(final_clip_labels, f, indent=4)

with open("data/image_labels.json", "w") as f:
    json.dump(final_image_labels, f, indent=4)

print(f"\nSuccess!")
print(f"Mapped {len(final_clip_labels)} clips and {len(final_image_labels)} images.")