import json
import csv
import os
import re

from .match_grades import normalize_string

matching_results_file = "data/matching_results_hd_new.json"
csv_path = "data/DR_Grading_Summary_v3.csv"
output_file = "data/unmatched_clip_labels.json"


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