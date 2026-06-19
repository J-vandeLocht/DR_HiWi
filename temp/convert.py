import pandas as pd

def convert_xlsx_to_csv(input_file, output_file, drop_columns=None, sheet_name=0):
    """
    Convert an XLSX file to CSV while dropping specified columns.

    Parameters:
    - input_file (str): Path to the .xlsx file
    - output_file (str): Path to save the .csv file
    - drop_columns (list): List of column names to drop
    - sheet_name (str or int): Excel sheet name or index (default: first sheet)
    """

    # Load Excel file
    df = pd.read_excel(input_file, sheet_name=sheet_name)

    # Drop columns if provided
    if drop_columns:
        # Only drop columns that actually exist (avoids KeyError)
        existing_cols_to_drop = [col for col in drop_columns if col in df.columns]
        df = df.drop(columns=existing_cols_to_drop)

    # Save to CSV (without index column)
    df.to_csv(output_file, index=False)

    print(f"Saved cleaned CSV to: {output_file}")


def clean_grading_short_csv(file_path):
    """
    Removes rows where either of the last two columns is empty or NaN.

    Parameters:
    - file_path (str): Path to Grading_Short.csv (or any CSV file)
    """

    df = pd.read_csv(file_path)

    # Get last two column names
    last_two_cols = df.columns[-2:]

    # Drop rows where either of the last two columns is NaN or empty string
    df = df.dropna(subset=last_two_cols)

    # Also treat empty strings as missing
    mask = ~(
        (df[last_two_cols[0]].astype(str).str.strip() == "") |
        (df[last_two_cols[1]].astype(str).str.strip() == "")
    )
    df = df[mask]

    # Save back to same file (overwrite)
    df.to_csv(file_path, index=False)

    print(f"Cleaned file saved: {file_path}")


if __name__ == "__main__":
    input_xlsx = "DR_Screening_Anonymized.xlsx"
    output_csv = "Grading_Short.csv"

    columns_to_drop = ["Initials", "DOB", "Patient ID", "Camp site", "Camp date", "Maculopathy R (ophthal)",
                       "Maculopathy L (ophthal)", "Referral", "Advice (ophthal)"]

    convert_xlsx_to_csv(
        input_file=input_xlsx,
        output_file=output_csv,
        drop_columns=columns_to_drop
    )

    clean_grading_short_csv("Grading_Short.csv")