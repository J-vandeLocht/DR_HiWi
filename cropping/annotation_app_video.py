import cv2
import numpy as np
import csv
import os
from pathlib import Path

# --- Configuration ---
# This points to the folder created by your extraction script
input_folder = Path('frames_to_annotate_larger')
output_csv = 'annotations_video_frames_larger.csv'

# UI PADDING: Extra space around the image in the window so you can click
# "outside" the frame if the retina is partially off-screen.
PADDING = 100
BG_COLOR = [30, 30, 30]  # Dark grey background for the UI
GUIDE_COLOR = (80, 80, 80)
window_name = "Retina Annotator - Square Frames"

# --- State ---
h_point, v_point = None, None


class AnnotationDB:
    def __init__(self, filepath):
        self.filepath = filepath
        self.data = {}  # {filename: [cx, cy, rx, ry]}
        self.fieldnames = ['filename', 'center_x', 'center_y', 'radius_x', 'radius_y']
        self.load()

    def load(self):
        if not os.path.exists(self.filepath):
            return
        with open(self.filepath, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    self.data[row['filename']] = [
                        float(row['center_x']), float(row['center_y']),
                        float(row['radius_x']), float(row['radius_y'])
                    ]
                except (ValueError, KeyError):
                    continue

    def save_entry(self, filename, cx, cy, rx, ry):
        self.data[filename] = [cx, cy, rx, ry]
        self.write_to_disk()

    def write_to_disk(self):
        with open(self.filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()
            for fname in sorted(self.data.keys()):
                vals = self.data[fname]
                writer.writerow({
                    'filename': fname,
                    'center_x': vals[0], 'center_y': vals[1],
                    'radius_x': vals[2], 'radius_y': vals[3]
                })

    def is_annotated(self, filename):
        return filename in self.data

    def get_annotation(self, filename):
        return self.data.get(filename)


# --- Helper Math ---
def get_ellipse_params(p_horz, p_vert):
    # p_vert defines the x-coordinate of center, p_horz defines the y-coordinate
    center_x, center_y = p_vert[0], p_horz[1]
    axis_x = abs(p_horz[0] - center_x)
    axis_y = abs(p_vert[1] - center_y)
    return (center_x, center_y), (axis_x, axis_y)


# --- UI Drawing ---
def draw_hud(img, current_idx, total, filename, is_done):
    h, w = img.shape[:2]
    color = (40, 60, 40) if is_done else (40, 40, 40)
    cv2.rectangle(img, (0, 0), (w, 80), color, -1)

    status = " [ANNOTATED]" if is_done else " [REQUIRED]"
    info = f"Image: {current_idx}/{total} | {filename}{status}"
    cv2.putText(img, info, (20, 35), cv2.FONT_HERSHEY_DUPLEX, 0.7, (220, 220, 220), 1, cv2.LINE_AA)

    ctrls = "L-Click: Horizontal Edge | Ctrl+Click: Vertical Edge | Space: Save | N: Next | B: Back"
    cv2.putText(img, ctrls, (20, 65), cv2.FONT_HERSHEY_PLAIN, 1.0, (180, 180, 180), 1, cv2.LINE_AA)


def redraw(base_img, filename, idx, total, db):
    display = base_img.copy()
    existing = db.get_annotation(filename)
    is_done = existing is not None

    if h_point or v_point:
        if h_point:
            cv2.line(display, (0, h_point[1]), (display.shape[1], h_point[1]), GUIDE_COLOR, 1)
            cv2.drawMarker(display, h_point, (255, 150, 0), cv2.MARKER_TILTED_CROSS, 15, 2)
        if v_point:
            cv2.line(display, (v_point[0], 0), (v_point[0], display.shape[0]), GUIDE_COLOR, 1)
            cv2.drawMarker(display, v_point, (0, 180, 255), cv2.MARKER_TILTED_CROSS, 15, 2)

        if h_point and v_point:
            center, axes = get_ellipse_params(h_point, v_point)
            cv2.ellipse(display, center, axes, 0, 0, 360, (0, 255, 100), 2, cv2.LINE_AA)
            cv2.circle(display, center, 3, (0, 255, 255), -1)

    elif existing:
        ex_cx, ex_cy, ex_rx, ex_ry = existing
        # Project saved coords back into padded UI space
        center = (int(ex_cx + PADDING), int(ex_cy + PADDING))
        axes = (int(ex_rx), int(ex_ry))
        cv2.ellipse(display, center, axes, 0, 0, 360, (120, 120, 120), 1, cv2.LINE_AA)
        cv2.putText(display, "EXISTING LABEL", (center[0] - 50, center[1] - 10),
                    cv2.FONT_HERSHEY_PLAIN, 0.8, (120, 120, 120), 1)

    draw_hud(display, idx, total, filename, is_done)
    cv2.imshow(window_name, display)


def mouse_callback(event, x, y, flags, param):
    global h_point, v_point
    if event == cv2.EVENT_LBUTTONDOWN:
        if flags & cv2.EVENT_FLAG_CTRLKEY:
            v_point = (x, y)
        else:
            h_point = (x, y)
        redraw(param['img'], param['name'], param['idx'], param['total'], param['db'])


# --- Main App ---
def main():
    global current_index, h_point, v_point, db

    db = AnnotationDB(output_csv)
    all_files = sorted([f for f in input_folder.iterdir() if f.suffix.lower() in {'.png', '.jpg', '.jpeg'}])

    if not all_files:
        print(f"No images found in {input_folder}")
        return

    # Find where we left off
    current_index = 0
    for i, f in enumerate(all_files):
        if not db.is_annotated(f.name):
            current_index = i
            break

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while True:
        if current_index >= len(all_files): current_index = 0
        if current_index < 0: current_index = len(all_files) - 1

        path = all_files[current_index]
        img = cv2.imread(str(path))
        if img is None:
            current_index += 1
            continue

        # Add UI Padding for easy clicking outside the frame
        padded = cv2.copyMakeBorder(img, PADDING, PADDING, PADDING, PADDING,
                                    cv2.BORDER_CONSTANT, value=BG_COLOR)
        h_point, v_point = None, None

        state = {'img': padded, 'name': path.name, 'idx': current_index + 1, 'total': len(all_files), 'db': db}
        cv2.setMouseCallback(window_name, mouse_callback, state)
        redraw(padded, path.name, current_index + 1, len(all_files), db)

        while True:
            key = cv2.waitKey(20) & 0xFF
            if key == ord('q'): return
            if key == ord('r'):
                h_point, v_point = None, None
                redraw(padded, path.name, current_index + 1, len(all_files), db)
            if key == ord('n'):
                current_index += 1
                break
            if key == ord('b'):
                current_index -= 1
                break
            if key in [32, 13]:  # Space or Enter
                if h_point and v_point:
                    center, axes = get_ellipse_params(h_point, v_point)
                    # Subtract PADDING to save true coordinates relative to image file
                    db.save_entry(path.name, center[0] - PADDING, center[1] - PADDING, axes[0], axes[1])
                    print(f"Saved {path.name}")
                    current_index += 1
                    break
                else:
                    print("Place both points before saving!")


if __name__ == "__main__":
    main()