import json
import csv
import os
import re

# Files
matching_results_file = "data/matching_results_hd_new.json"
csv_path = "data/DR_Grading_Summary_v3.csv"
output_file = "data/unmatched_clip_labels.json"


def normalize_string(s):
    """
    Standardizes names so 'CLEAN_IMG_5029' matches '09-05-2024\\IMG_5029.MOV'
    """
    if not s: return ""
    # 1. Remove "CLEAN_" prefix
    s = re.sub(r'^CLEAN_', '', s, flags=re.IGNORECASE)
    # 2. Get filename only, handle mixed slashes
    s = s.replace('\\', '/').split('/')[-1]
    # 3. Strip extension
    s = os.path.splitext(s)[0]
    # 4. Remove all non-alphanumeric and lowercase
    s = re.sub(r'[^a-zA-Z0-9]', '', s).lower()
    return s


# 1. Load the list of unmatched clips
if not os.path.exists(matching_results_file):
    print(f"Error: {matching_results_file} not found. Run the matching script first!")
    exit()

with open(matching_results_file, "r") as f:
    data = json.load(f)
    # This takes the list of strings from "unmatched_clips"
    unmatched_list = data.get("unmatched_clips", [])

# 2. Load CSV grades into a normalized lookup dictionary
csv_grades = {}
with open(csv_path, mode='r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        path_raw = row['Path'].strip()
        grade = int(row['Grading'])
        csv_grades[normalize_string(path_raw)] = grade

# 3. Match the clips
final_labels = {}
not_found = []

for clip_id in unmatched_list:
    query = normalize_string(clip_id)

    # Try exact match first
    if query in csv_grades:
        final_labels[f"{clip_id}.mp4"] = csv_grades[query]
    else:
        # Fallback: substring matching
        match_found = False
        for csv_key, grade in csv_grades.items():
            if query in csv_key or csv_key in query:
                final_labels[f"{clip_id}.mp4"] = grade
                match_found = True
                break

        if not match_found:
            not_found.append(clip_id)

# 4. Save results
with open(output_file, "w") as f:
    json.dump(final_labels, f, indent=4)

print(f"--- Processing Complete ---")
print(f"Successfully labeled: {len(final_labels)} unmatched clips.")
if not_found:
    print(f"Warning: {len(not_found)} clips still have no grade in CSV (e.g., {not_found[:2]})")
print(f"Saved labels to: {output_file}")