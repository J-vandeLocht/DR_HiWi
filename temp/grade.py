import pandas as pd
import json

GRADE_MAP = {
    "No DR": 0,
    "Mild DR": 1,
    "Moderate DR": 2,
    "Severe DR": 3,
    "Proliferative DR": 4
}

def process_dr_grading(input_csv, output_csv, output_json):
    """
    Creates:
      1. CSV with columns:
         video_name, grade, referable_dr
      2. JSON with format:
         {
           "123.L.V1.MOV": 0,
           "123.R.V1.MOV": 2
         }
    """

    df = pd.read_csv(input_csv)

    output_rows = []
    json_labels = {}

    for _, row in df.iterrows():
        video_field = str(row["Video files"])
        left_grade = row.iloc[1]
        right_grade = row.iloc[2]

        video_field = video_field.strip().strip('"')
        videos = [v.strip() for v in video_field.split(",")]

        for v in videos:
            v_lower = v.lower()

            if ".l." in v_lower:
                grade_raw = left_grade
            elif ".r." in v_lower:
                grade_raw = right_grade
            else:
                continue

            if pd.isna(grade_raw):
                continue

            grade_raw = str(grade_raw).strip()

            if grade_raw == "Unable to grade":
                continue

            if grade_raw not in GRADE_MAP:
                continue

            grade = GRADE_MAP[grade_raw]
            referable = grade >= 2

            output_rows.append({
                "video_name": v,
                "grade": grade,
                "referable_dr": referable
            })

            # JSON entry
            json_labels[v] = grade

    # Save CSV
    out_df = pd.DataFrame(output_rows)
    out_df.to_csv(output_csv, index=False)

    # Save JSON
    with open(output_json, "w") as f:
        json.dump(json_labels, f, indent=2)

    print(f"Saved CSV: {output_csv}")
    print(f"Saved JSON: {output_json}")

if __name__ == "__main__":
    input_file_csv = "Grading_Short.csv"
    output_file_csv = "Grading_Final.csv"
    output_file_json = "Grading_Final.json"

    process_dr_grading(input_file_csv,output_file_csv, output_file_json)
