import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
import cv2
import numpy as np
import json
import os
import re


class FrameDataset(Dataset):
    def __init__(self, json_path, img_dir, transform=None):
        with open(json_path, 'r') as f:
            label_data = json.load(f)

        self.image_paths = []
        self.labels = []

        for img_name, grade in label_data.items():
            self.image_paths.append(os.path.join(img_dir, img_name))
            # Convert 0-4 to binary: 0-1 is non-referable (0), 2-4 is referable (1)
            self.labels.append(1 if grade >= 2 else 0)

        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(self.labels[idx], dtype=torch.float32)


class MILVideoDataset(Dataset):
    def __init__(self, json_path, video_dir, num_frames=16, transform=None):
        with open(json_path, 'r') as f:
            self.label_map = json.load(f)

        self.video_ids = list(self.label_map.keys())
        self.video_dir = video_dir
        self.num_frames = num_frames
        self.transform = transform

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, idx):
        vid_name = self.video_ids[idx]
        vid_path = os.path.join(self.video_dir, vid_name)
        grade = self.label_map[vid_name]

        # Convert Grade to Binary Label (Referable >= 2)
        label = 1.0 if grade >= 2 else 0.0

        # Load and sample frames
        bag = self._load_video_frames(vid_path)

        return bag, torch.tensor(label, dtype=torch.float32)

    def _load_video_frames(self, path):
        frames = []
        cap = cv2.VideoCapture(path)

        if not cap.isOpened():
            raise RuntimeError(f"Could not open video file: {path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # FAIL FAST: If the video is shorter than our required sample size,
        # we don't want to pad/interpolate. We want to know about it.
        if total_frames < self.num_frames:
            cap.release()
            raise RuntimeError(
                f"Video {path} only has {total_frames} frames. "
                f"Minimum required is {self.num_frames}."
            )

        # Sample N frames uniformly
        indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)

        for i in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            if i in indices:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                if self.transform:
                    img = self.transform(img)
                frames.append(img)

        cap.release()

        # Final check: Ensure we actually got the count we expected
        if len(frames) != self.num_frames:
            raise RuntimeError(
                f"Frame extraction failed for {path}. "
                f"Expected {self.num_frames}, got {len(frames)}."
            )

        return torch.stack(frames) # Shape: [Bag_Size, 3, 224, 224]


class MILVideoDatasetNew(Dataset):
    def __init__(self, json_path, num_frames=32, transform=None):
        with open(json_path, 'r') as f:
            self.label_map = json.load(f)

        self.num_frames = num_frames
        self.transform = transform

        search_dirs = [
            Path('data/own_clips_hd/train_videos/cleaned_videos'),
            Path('data/own_clips_hd/val_videos/cleaned_videos')
        ]

        all_video_paths = []
        for d in search_dirs:
            if d.exists():
                all_video_paths.extend(list(d.glob('*.mp4')))

        self.data = []
        missing_count = 0
        corrupt_count = 0

        print(f"Mapping dataset from {json_path}...")

        for vid_name, grade in self.label_map.items():
            matched_path = None

            # 1. Clean up the JSON key in case it has an extension (like "R062R.mp4")
            base_name = vid_name.replace('.mp4', '').replace('.avi', '')

            # 2. Build the Regex pattern.
            # re.escape makes sure any weird characters in your names are handled safely.
            # (?![a-zA-Z0-9]) ensures the match isn't just the prefix of a longer ID.
            pattern = re.compile(rf"{re.escape(base_name)}(?![a-zA-Z0-9])")

            for path in all_video_paths:
                if pattern.search(path.name):
                    matched_path = path
                    break

            if matched_path and matched_path.exists():
                # 3. Health check against 0-byte zombie files
                if matched_path.stat().st_size > 1000:
                    self.data.append({
                        'vid_name': vid_name,
                        'path': str(matched_path),
                        'label': 1.0 if grade >= 2 else 0.0
                    })
                else:
                    corrupt_count += 1
            else:
                missing_count += 1

        print(f"Successfully mapped {len(self.data)} videos.")
        if missing_count > 0:
            print(f"Skipped {missing_count} videos (not found).")
        if corrupt_count > 0:
            print(f"Skipped {corrupt_count} videos (found but appear empty/corrupt).")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        bag = self._load_video_frames(item['path'])
        return bag, torch.tensor(item['label'], dtype=torch.float32)

    def _load_video_frames(self, path):
        frames = []
        cap = cv2.VideoCapture(path)

        if not cap.isOpened():
            raise RuntimeError(f"Could not open video file: {path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Prevent math errors if a video is completely broken
        if total_frames <= 0:
            cap.release()
            raise RuntimeError(f"Video {path} has 0 frames or unreadable metadata.")

        # Generate target indices. If total_frames < num_frames, this automatically creates duplicates!
        target_indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int).tolist()

        last_valid_frame = None

        for i in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break  # Video ended earlier than metadata suggested

            last_valid_frame = frame

            # Count how many times this specific frame index is needed
            times_to_add = target_indices.count(i)

            if times_to_add > 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)

                if self.transform:
                    img = self.transform(img)

                # Append it the required number of times (handles the duplication)
                for _ in range(times_to_add):
                    frames.append(img)

            # Optimization: break early if we filled the bag
            if len(frames) == self.num_frames:
                break

        cap.release()

        # Failsafe: If OpenCV's total_frames metadata was wrong and the video ended early,
        # we pad the remaining slots with the last valid frame we successfully read.
        while len(frames) < self.num_frames and last_valid_frame is not None:
            frame_rgb = cv2.cvtColor(last_valid_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            if self.transform:
                img = self.transform(img)
            frames.append(img)

        if len(frames) != self.num_frames:
            raise RuntimeError(
                f"Critical failure: Expected {self.num_frames} frames from {path}, but ended up with {len(frames)}.")

        return torch.stack(frames)  # Shape: [num_frames, C, H, W]


class KaggleDRDataset(Dataset):
    def __init__(self, split_dir, transform=None):
        self.image_paths = []
        self.labels = []
        self.transform = transform

        for grade_folder in sorted(os.listdir(split_dir)):
            grade_path = os.path.join(split_dir, grade_folder)
            if os.path.isdir(grade_path) and grade_folder.isdigit():
                grade = int(grade_folder)
                for img_name in os.listdir(grade_path):
                    if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                        self.image_paths.append(os.path.join(grade_path, img_name))
                        self.labels.append(grade) # Return 0, 1, 2, 3, or 4

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(self.labels[idx], dtype=torch.long)
