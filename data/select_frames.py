import sys
import json
from pathlib import Path
import cv2
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QGridLayout,
    QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QMessageBox,
    QFileDialog
)
# Fixed import: QShortcut is in QtGui, not QtWidgets
from PyQt6.QtGui import QImage, QPixmap, QKeySequence, QShortcut
from PyQt6.QtCore import Qt


class FullResWindow(QMainWindow):
    """Secondary window to display full-resolution images."""

    def __init__(self, pixmap: QPixmap, title: str = "Full Resolution Image"):
        super().__init__()
        self.setWindowTitle(title)
        self.setStyleSheet("background-color: #1e1e1e;")

        label = QLabel()
        label.setPixmap(pixmap)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.setCentralWidget(label)
        self.resize(900, 900)


class ImageWidget(QLabel):
    """Custom QLabel handling clicks and selection states."""

    def __init__(self, index: int, low_res_pixmap: QPixmap, full_res_pixmap: QPixmap, frame_idx: int):
        super().__init__()
        self.index = index  # Grid index (0-8)
        self.frame_idx = frame_idx  # Original video frame number
        self.low_res_pixmap = low_res_pixmap
        self.full_res_pixmap = full_res_pixmap
        self.is_selected = False

        self.setPixmap(self.low_res_pixmap)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setScaledContents(True)
        self.setFixedSize(280, 280)
        self.update_style()

    def update_style(self):
        """Toggle styled border when selected."""
        if self.is_selected:
            self.setStyleSheet("""
                border: 5px solid #ff4d4d;
                border-radius: 8px;
                padding: 0px;
                background-color: #2a2a2a;
            """)
        else:
            self.setStyleSheet("""
                border: 2px solid #3c3c3c;
                border-radius: 8px;
                padding: 0px;
                background-color: #2a2a2a;
            """)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            # Left Click: Toggle Selection
            self.is_selected = not self.is_selected
            self.update_style()
        elif event.button() == Qt.MouseButton.RightButton:
            # Right Click: Open Full-Res Window
            self.full_window = FullResWindow(
                self.full_res_pixmap,
                title=f"Frame {self.frame_idx} - Full Resolution"
            )
            self.full_window.show()


class FrameSelectorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Diabetic Retinopathy Frame Annotator")

        # Start Maximized
        self.showMaximized()

        # Dark Theme Styling
        self.setStyleSheet("""
            QMainWindow {
                background-color: #121212;
            }
            QLabel {
                color: #e0e0e0;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            QPushButton {
                background-color: #2563eb;
                color: white;
                font-size: 15px;
                font-weight: bold;
                border: none;
                border-radius: 6px;
                padding: 10px 20px;
            }
            QPushButton:hover {
                background-color: #1d4ed8;
            }
            QPushButton:pressed {
                background-color: #1e40af;
            }
        """)

        # App State
        self.video_files = []
        self.current_video_idx = 0
        self.annotations = {}
        self.image_widgets = []
        self.video_dir = None

        # Quit Shortcut (Ctrl + Q)
        self.quit_shortcut = QShortcut(QKeySequence("Ctrl+Q"), self)
        self.quit_shortcut.activated.connect(self.close)

        self.init_ui()
        self.select_directory()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(30, 20, 30, 20)

        # Header Info Banner
        header_layout = QHBoxLayout()
        self.info_label = QLabel("Please select a directory containing .mp4 videos...")
        self.info_label.setStyleSheet("font-size: 18px; font-weight: 600; color: #f3f4f6;")

        self.progress_label = QLabel("")
        self.progress_label.setStyleSheet("font-size: 16px; color: #9ca3af;")

        header_layout.addWidget(self.info_label)
        header_layout.addStretch()
        header_layout.addWidget(self.progress_label)
        main_layout.addLayout(header_layout)

        main_layout.addSpacing(10)

        # 3x3 Image Grid Container
        self.grid_container = QWidget()
        self.grid_layout = QGridLayout(self.grid_container)
        self.grid_layout.setSpacing(20)
        self.grid_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        main_layout.addWidget(self.grid_container, stretch=1)

        # Bottom Action Bar
        btn_layout = QHBoxLayout()

        self.submit_btn = QPushButton("Save & Next Video →")
        self.submit_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.submit_btn.setFixedHeight(48)
        self.submit_btn.clicked.connect(self.on_submit)

        btn_layout.addStretch()
        btn_layout.addWidget(self.submit_btn)
        btn_layout.addStretch()

        main_layout.addLayout(btn_layout)

    def select_directory(self):
        """Prompt user to choose folder with MP4 files."""
        dir_path = QFileDialog.getExistingDirectory(self, "Select Video Folder")
        if not dir_path:
            sys.exit(0)

        self.video_dir = Path(dir_path)
        self.video_files = sorted(list(self.video_dir.glob("*.mp4")))

        if not self.video_files:
            QMessageBox.warning(self, "No Videos Found", "No .mp4 files found in the selected folder.")
            sys.exit(0)

        self.load_video(0)

    def cv_frame_to_pixmap(self, frame: np.ndarray, width: int = None, height: int = None) -> QPixmap:
        """Convert BGR OpenCV frame to QPixmap safely (WSL-friendly memory management)."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # .copy() prevents memory segmentation faults across C++ Qt bindings
        rgb_frame = np.ascontiguousarray(rgb_frame)

        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qimage = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)

        pixmap = QPixmap.fromImage(qimage)
        if width and height:
            return pixmap.scaled(width, height, Qt.AspectRatioMode.KeepAspectRatio,
                                 Qt.TransformationMode.SmoothTransformation)
        return pixmap

    def load_video(self, idx: int):
        """Extract 9 uniformly sampled frames and build grid."""
        # Clear existing grid widgets
        for widget in self.image_widgets:
            widget.deleteLater()
        self.image_widgets.clear()

        video_path = self.video_files[idx]
        self.info_label.setText(f"Current Video: {video_path.name}")
        self.progress_label.setText(f"Video {idx + 1} of {len(self.video_files)}")

        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames < 9:
            QMessageBox.critical(self, "Error", f"Video {video_path.name} has fewer than 9 frames.")
            return

        # Sample 9 uniformly spaced frame indices
        frame_indices = np.linspace(0, total_frames - 1, 9, dtype=int)

        for grid_idx, f_idx in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret, frame = cap.read()
            if not ret:
                continue

            # Convert to QPixmap for low-res tile and high-res pop-out
            low_res_pixmap = self.cv_frame_to_pixmap(frame, 280, 280)
            full_res_pixmap = self.cv_frame_to_pixmap(frame)

            img_widget = ImageWidget(grid_idx, low_res_pixmap, full_res_pixmap, frame_idx=int(f_idx))
            self.image_widgets.append(img_widget)

            row, col = divmod(grid_idx, 3)
            self.grid_layout.addWidget(img_widget, row, col)

        cap.release()

    def save_annotations_to_json(self):
        """Save selection data to annotations.json in video folder."""
        out_path = self.video_dir / "annotations.json"
        with open(out_path, "w") as f:
            json.dump(self.annotations, f, indent=4)

    def on_submit(self):
        """Record selected frames and advance to next video."""
        current_video_name = self.video_files[self.current_video_idx].name

        # Save selected frame numbers for the current video
        selected_frames = [
            w.frame_idx for w in self.image_widgets if w.is_selected
        ]

        self.annotations[current_video_name] = {
            "selected_frames": selected_frames,
            "total_selected": len(selected_frames)
        }

        self.save_annotations_to_json()

        # Advance to next video or finish
        if self.current_video_idx + 1 < len(self.video_files):
            self.current_video_idx += 1
            self.load_video(self.current_video_idx)
        else:
            QMessageBox.information(
                self,
                "Completed!",
                f"All {len(self.video_files)} videos have been processed.\nAnnotations saved to annotations.json"
            )
            self.close()
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FrameSelectorApp()
    sys.exit(app.exec())