import cv2
import numpy as np
import csv
import os
from pathlib import Path

# --- Configuration ---
input_folder = Path('data/2024_Paxos_Frames/frames')
output_csv = 'annotations.csv'
PADDING = 400
BG_COLOR = [45, 42, 42]
GUIDE_COLOR = (80, 80, 80)

# --- State ---
h_point, v_point = None, None
window_name = "Smart Fundus Annotator"


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
                # Store numeric values
                try:
                    self.data[row['filename']] = [
                        float(row['center_x']), float(row['center_y']),
                        float(row['radius_x']), float(row['radius_y'])
                    ]
                except ValueError:
                    continue

    def save_entry(self, filename, cx, cy, rx, ry):
        # Update internal dict
        self.data[filename] = [cx, cy, rx, ry]
        self.write_to_disk()

    def write_to_disk(self):
        # Rewrite the entire file to ensure no duplicates and sorted order
        with open(self.filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()
            # Sort by filename to keep CSV tidy
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
    center_x, center_y = p_vert[0], p_horz[1]
    axis_x = abs(p_horz[0] - center_x)
    axis_y = abs(p_vert[1] - center_y)
    return (center_x, center_y), (axis_x, axis_y)


# --- Drawing & UI ---
def draw_hud(img, current_idx, total, filename, is_done):
    h, w = img.shape[:2]
    # Header bar
    color = (30, 50, 30) if is_done else (30, 28, 28)  # Greenish tint if done
    cv2.rectangle(img, (0, 0), (w, 80), color, -1)

    status = " [DONE]" if is_done else " [TODO]"
    info = f"IMG: {current_idx}/{total} | {filename}{status}"

    cv2.putText(img, info, (20, 35), cv2.FONT_HERSHEY_DUPLEX, 0.7, (220, 220, 220), 1, cv2.LINE_AA)

    ctrls = "L-Click: Horiz | Ctrl+Click: Vert | Space: Save | N: Next | B: Back | G: Goto..."
    cv2.putText(img, ctrls, (20, 65), cv2.FONT_HERSHEY_PLAIN, 1.0, (150, 150, 150), 1, cv2.LINE_AA)


def redraw(base_img, filename, idx, total, db):
    display = base_img.copy()

    # Check if already annotated in DB to show previous work
    existing = db.get_annotation(filename)
    is_done = existing is not None

    # If we have points currently set by user, draw them
    if h_point or v_point:
        # Draw user guides
        if h_point:
            cv2.line(display, (0, h_point[1]), (display.shape[1], h_point[1]), GUIDE_COLOR, 1)
            cv2.drawMarker(display, h_point, (255, 150, 0), cv2.MARKER_TILTED_CROSS, 20, 2)
        if v_point:
            cv2.line(display, (v_point[0], 0), (v_point[0], display.shape[0]), GUIDE_COLOR, 1)
            cv2.drawMarker(display, v_point, (0, 180, 255), cv2.MARKER_TILTED_CROSS, 20, 2)

        if h_point and v_point:
            center, axes = get_ellipse_params(h_point, v_point)
            cv2.ellipse(display, center, axes, 0, 0, 360, (0, 255, 100), 2, cv2.LINE_AA)
            cv2.circle(display, center, 4, (0, 255, 255), -1)

    # If no user points yet, but DB has data, draw the saved annotation (Ghosted)
    elif existing:
        ex_cx, ex_cy, ex_rx, ex_ry = existing
        # Add padding back to draw on this specific view
        center = (int(ex_cx + PADDING), int(ex_cy + PADDING))
        axes = (int(ex_rx), int(ex_ry))
        cv2.ellipse(display, center, axes, 0, 0, 360, (100, 100, 100), 1, cv2.LINE_AA)
        cv2.putText(display, "SAVED ANNOTATION", (center[0] - 60, center[1]), cv2.FONT_HERSHEY_PLAIN, 1,
                    (100, 100, 100), 1)

    draw_hud(display, idx, total, filename, is_done)
    cv2.imshow(window_name, display)


def mouse_callback(event, x, y, flags, param):
    global h_point, v_point
    # param is a simple object/dict to hold current state references
    state = param

    if event == cv2.EVENT_LBUTTONDOWN:
        if flags & cv2.EVENT_FLAG_CTRLKEY:
            v_point = (x, y)
        else:
            h_point = (x, y)
        redraw(state['img'], state['name'], state['idx'], state['total'], state['db'])


# --- Main App Logic ---
db = AnnotationDB(output_csv)
all_files = sorted([f for f in input_folder.iterdir() if f.suffix.lower() in {'.png', '.jpg', '.jpeg'}])

if not all_files:
    print("No images found.")
    exit()

# Find first unannotated image
start_index = 0
all_done = True
for i, f in enumerate(all_files):
    if not db.is_annotated(f.name):
        start_index = i
        all_done = False
        break

if all_done:
    print("All images have been annotated! Starting review mode from beginning.")
    start_index = 0

current_index = start_index
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

while True:
    # 1. Load Current Image
    # Handle index wrapping
    if current_index >= len(all_files):
        current_index = 0  # Loop back to start
    if current_index < 0:
        current_index = len(all_files) - 1

    path = all_files[current_index]
    original = cv2.imread(str(path))
    if original is None:
        current_index += 1
        continue

    padded = cv2.copyMakeBorder(original, PADDING, PADDING, PADDING, PADDING, cv2.BORDER_CONSTANT, value=BG_COLOR)

    # 2. Reset Points for new image
    h_point, v_point = None, None

    # 3. Setup Callback State
    state = {
        'img': padded,
        'name': path.name,
        'idx': current_index + 1,
        'total': len(all_files),
        'db': db
    }
    cv2.setMouseCallback(window_name, mouse_callback, state)

    # 4. Initial Draw
    redraw(padded, path.name, current_index + 1, len(all_files), db)

    # 5. Wait Loop
    move_to_next = False
    while not move_to_next:
        key = cv2.waitKey(20) & 0xFF

        if key == ord('q'):
            exit()

        # R - Reset current points
        if key == ord('r'):
            h_point, v_point = None, None
            redraw(padded, path.name, current_index + 1, len(all_files), db)

        # N - Skip to Next (without saving)
        if key == ord('n'):
            current_index += 1
            move_to_next = True

        # B - Back to Previous
        if key == ord('b'):
            current_index -= 1
            move_to_next = True

        # G - Goto specific index (simple console input)
        if key == ord('g'):
            try:
                val = int(input("Jump to image index: "))
                current_index = val - 1
                move_to_next = True
            except:
                print("Invalid input")

        # SPACE / ENTER - Save & Auto-Advance
        if key in [32, 13, 10]:
            if h_point and v_point:
                center, axes = get_ellipse_params(h_point, v_point)
                # Save to DB (handles overwrite automatically)
                db.save_entry(path.name, center[0] - PADDING, center[1] - PADDING, axes[0], axes[1])
                print(f"Saved: {path.name}")

                # Auto-Advance Logic
                # Check if there are any remaining unannotated images
                next_unannotated = -1
                for i in range(current_index + 1, len(all_files)):
                    if not db.is_annotated(all_files[i].name):
                        next_unannotated = i
                        break

                if next_unannotated != -1:
                    current_index = next_unannotated
                else:
                    # If all future images are done, just go to immediate next (Review mode)
                    # Or loop to start if at end
                    current_index += 1
                    # Check if we just finished the LAST image
                    if current_index >= len(all_files):
                        # If we are effectively "done" with the set
                        remaining = [f for f in all_files if not db.is_annotated(f.name)]
                        if not remaining:
                            print("🎉 All images annotated! Looping back to start.")
                            current_index = 0

                move_to_next = True
            else:
                # If trying to save but no points, just treat as "Next" if already annotated?
                # Or block? Let's block to prevent accidental empty saves.
                print("⚠️  Please place points to save (or press 'N' to skip)")

cv2.destroyAllWindows()