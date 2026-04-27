import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --- Configuration ---
csv_path = 'annotations.csv'
image_folder = Path('data/2024_Paxos_Frames/frames')


def verify_9_annotations():
    if not Path(csv_path).exists():
        print(f"Error: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)

    # Check if we have at least 9 images to sample
    num_to_sample = min(9, len(df))
    if num_to_sample == 0:
        print("No annotations found in CSV.")
        return

    # Pick 9 random samples (or all if less than 9)
    samples = df.sample(num_to_sample)

    # Create a 9x3 grid. We adjust figsize to be taller to accommodate 9 rows.
    fig, axes = plt.subplots(num_to_sample, 3, figsize=(18, 4 * num_to_sample))
    plt.subplots_adjust(wspace=0.1, hspace=0.4)

    # Ensure axes is 2D even if only 1 image is sampled
    if num_to_sample == 1:
        axes = np.expand_dims(axes, axis=0)

    for i, (idx, row) in enumerate(samples.iterrows()):
        img_path = image_folder / row['filename']
        img = cv2.imread(str(img_path))

        if img is None:
            print(f"Skipping {row['filename']}: Image not found.")
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 1. Parse Coordinates
        cx, cy = int(row['center_x']), int(row['center_y'])
        rx, ry = int(row['radius_x']), int(row['radius_y'])

        # 2. Create Binary Mask
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, 255, -1)

        # 3. Create Masked Image (Retina content only)
        masked_img = cv2.bitwise_and(img_rgb, img_rgb, mask=mask)

        # --- Plotting Row i ---

        # Column 1: Original + Overlay
        ax_orig = axes[i, 0]
        preview = img_rgb.copy()
        cv2.ellipse(preview, (cx, cy), (rx, ry), 0, 0, 360, (0, 255, 0), 4)
        ax_orig.imshow(preview)
        ax_orig.set_title(f"Sample {i + 1}: {row['filename']}", fontsize=10)
        ax_orig.axis('off')

        # Column 2: Masked/Cropped Region
        ax_masked = axes[i, 1]
        ax_masked.imshow(masked_img)
        ax_masked.set_title("Masked Image", fontsize=10)
        ax_masked.axis('off')

        # Column 3: Binary Map
        ax_bin = axes[i, 2]
        ax_bin.imshow(mask, cmap='gray')
        ax_bin.set_title("Binary Mask", fontsize=10)
        ax_bin.axis('off')

    plt.suptitle(f"Batch QC: 9 Random Samples from {csv_path}", fontsize=18, fontweight='bold', y=0.92)

    # Save the check if needed, or just show
    plt.savefig('annotation_check_9.png', bbox_inches='tight', dpi=150)
    print(f"Verification plot for {num_to_sample} images saved as 'annotation_check_9.png'")
    plt.show()


if __name__ == "__main__":
    verify_9_annotations()