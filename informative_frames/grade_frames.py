import cv2
import os
import pandas as pd
from pathlib import Path

INPUT_DIR = 'data/informative_frames/frames_raw_extract_paxos2020'
WINDOW_NAME = 'Retina Labeler'

MODE_BINARY = "binary"
MODE_THREE_CLASS = "three_class"


def get_mode_config(mode):
    if mode == MODE_BINARY:
        return {
            "output_file": "manual_labels_binary_paxos2020.csv",
            "key_map": {
                ord('1'): ("INFORMATIVE", 1),
                ord('0'): ("JUNK", 0),
            },
            "instructions": "Controls: [1] Informative | [0] Junk | [S] Skip | [B] Back | [ESC] Save & Exit"
        }

    elif mode == MODE_THREE_CLASS:
        return {
            "output_file": "manual_labels_three_class.csv",
            "key_map": {
                ord('1'): ("UNUSABLE", 0),
                ord('2'): ("NOISY", 1),
                ord('3'): ("PERFECT", 2),
            },
            "instructions": "Controls: [1] Unusable | [2] Noisy | [3] Perfect | [S] Skip | [B] Back | [ESC] Save & Exit"
        }

    else:
        raise ValueError("Invalid mode selected.")


def start_labeling(folder_path, mode=MODE_BINARY):
    config = get_mode_config(mode)

    OUTPUT_FILE = config["output_file"]
    KEY_MAP = config["key_map"]
    INSTRUCTIONS = config["instructions"]

    folder = Path(folder_path)
    images = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # Load existing progress
    if os.path.exists(OUTPUT_FILE):
        df = pd.read_csv(OUTPUT_FILE)
        labeled_files = set(df['filename'].tolist())
    else:
        df = pd.DataFrame(columns=['filename', 'label'])
        labeled_files = set()

    print(f"\nMode: {mode}")
    print(f"Total images: {len(images)}")
    print(f"Already labeled: {len(labeled_files)}")
    print(INSTRUCTIONS)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 800, 800)

    results = []
    history = []  # <-- stores (img_name, index position)

    i = 0
    while i < len(images):
        img_name = images[i]

        if img_name in labeled_files:
            i += 1
            continue

        img_path = str(folder / img_name)
        img = cv2.imread(img_path)

        if img is None:
            print(f"Could not read {img_name}")
            i += 1
            continue

        display_img = img.copy()
        cv2.putText(display_img, f"{img_name}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow(WINDOW_NAME, display_img)

        key = cv2.waitKey(0) & 0xFF

        # --- LABEL ---
        if key in KEY_MAP:
            label_name, label_value = KEY_MAP[key]
            results.append({'filename': img_name, 'label': label_value})
            history.append((img_name, i))  # track position
            print(f"Labeled {img_name} as {label_name}")
            i += 1

        # --- SKIP ---
        elif key == ord('s'):
            print(f"Skipped {img_name}")
            i += 1

        # --- BACK / UNDO ---
        elif key == ord('b'):
            if history:
                last_img, last_index = history.pop()

                # Remove last result
                if results and results[-1]['filename'] == last_img:
                    results.pop()

                print(f"Undo: returning to {last_img}")

                i = last_index  # go back
            else:
                print("Nothing to undo.")

        # --- EXIT ---
        elif key == 27:
            print("Exiting and saving...")
            break

        else:
            print("Invalid key. Try again.")

    # Save progress
    if results:
        new_df = pd.DataFrame(results)
        df = pd.concat([df, new_df], ignore_index=True)
        df.to_csv(OUTPUT_FILE, index=False)
        print(f"Progress saved to {OUTPUT_FILE}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    start_labeling(INPUT_DIR, mode=MODE_BINARY)
