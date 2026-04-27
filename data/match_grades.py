import json
import csv
import os

# Files
matching_file = "matching_results_hd.json"
csv_path = "DR_Grading_Summary_v3.csv"

# 1. Load the matches (Only videos that have expert frames)
with open(matching_file, "r") as f:
    matches_data = json.load(f)["matches"]

# 2. Load CSV grades into a searchable dict
# We clean the keys to handle the 'IMG_XXXX' or 'R035R' logic
csv_grades = {}
with open(csv_path, mode='r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        # We store the raw path but also a 'clean' version for matching
        path_raw = row['Path'].strip()
        csv_grades[path_raw] = int(row['Grading'])


def get_grade_for_id(target_id, reference_dict):
    # Priority 1: Exact match
    if target_id in reference_dict:
        return reference_dict[target_id]
    # Priority 2: Substring match
    for path_key, grade in reference_dict.items():
        if target_id.lower() in path_key.lower() or path_key.lower() in target_id.lower():
            return grade
    return None


# 3. Create the two label maps
final_clip_labels = {}
final_image_labels = {}

for clip_id, image_list in matches_data.items():
    # Find the grade for this clip/video ID
    grade = get_grade_for_id(clip_id, csv_grades)
    if grade is not None:
        # Assign grade to the clip (add .mp4 extension for the dataloader)
        final_clip_labels[f"{clip_id}.mp4"] = grade

        # Assign the SAME grade to all corresponding expert images
        for img_name in image_list:
            final_image_labels[img_name] = grade
    else:
        print(f"Warning: No grade found in CSV for matched ID: {clip_id}")

# 4. Save the finalized labels
with open("final_clip_labels_hd.json", "w") as f:
    json.dump(final_clip_labels, f, indent=4)

with open("final_image_labels_hd.json", "w") as f:
    json.dump(final_image_labels, f, indent=4)

print(f"Success!")
print(f"Mapped {len(final_clip_labels)} clips and {len(final_image_labels)} images.")