import torch
import pandas as pd
import os
import json
import re
from PIL import Image
from tqdm import tqdm
import argparse
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

from data.dataset import MILVideoDataset
from dino.utils import make_val_transform_dino
from dino.train_dino_mil import DinoMIL
from dino.train_dino_transformer import DinoSelfAttention
from dino.train_dino_classifier import DinoClassifier


class FrameDataset(Dataset):
    def __init__(self, json_path, img_dirs, transform=None):
        with open(json_path, 'r') as f:
            label_data = json.load(f)

        self.filename_to_path = {}
        for img_dir in img_dirs:
            for fname in os.listdir(img_dir):
                full_path = os.path.join(img_dir, fname)
                if fname in self.filename_to_path:
                    continue
                self.filename_to_path[fname] = full_path

        self.image_paths = []
        self.image_names = []
        self.grades = []
        self.labels = []
        missing = []

        for img_name, grade in label_data.items():
            if img_name not in self.filename_to_path:
                missing.append(img_name)
                continue

            self.image_paths.append(self.filename_to_path[img_name])
            self.image_names.append(img_name)
            self.grades.append(grade)
            self.labels.append(1 if grade >= 2 else 0)

        if missing:
            print(f"WARNING: {len(missing)} image(s) from {json_path} were not "
                  f"found in any of img_dirs and were skipped.")

        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)

        return (
            img,
            torch.tensor(self.grades[idx], dtype=torch.int64),
            torch.tensor(self.labels[idx], dtype=torch.float32),
            self.image_names[idx]
        )


def match_img_to_video(img_name, vid_names):
    """Matches a frame image name back to its corresponding video key in the dataset."""
    for vid_name in vid_names:
        base_name = vid_name.replace('.mp4', '').replace('.MOV', '')
        if base_name in img_name:
            return vid_name
    return None


def run_inference_transformer(model, loader, device):
    model.eval()
    results = {}
    dataset_samples = loader.dataset.data

    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc="Transformer Inference")):
            inputs = batch[0].squeeze(0).to(device)
            labels = batch[1]
            grade = batch[2]
            video_name = dataset_samples[i]['vid_name']

            logits, _ = model(inputs)
            prob = torch.sigmoid(logits).item()

            results[video_name] = {
                "label": int(labels.item()),
                "prob": prob,
                "grade": int(grade.item())
            }
    return results


def run_inference_transformer_all(model, loader, device, num_classes=5):
    model.eval()
    results = {}
    dataset_samples = loader.dataset.data

    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc="Transformer Inference")):
            inputs = batch[0].squeeze(0).to(device)
            labels = batch[1]
            grade = batch[2]
            video_name = dataset_samples[i]['vid_name']

            logits, _ = model(inputs)
            probs = torch.softmax(logits, dim=1).squeeze(0)
            pred_grade = int(torch.argmax(probs).item())

            result = {
                "label": int(labels.item()),
                "grade": int(grade.item()),
                "pred_grade": pred_grade,
            }
            for c in range(num_classes):
                result[f"prob_{c}"] = probs[c].item()

            results[video_name] = result

    return results


def run_inference_classifier_frames(model, loader, device, vid_names):
    """Predicts on individual frames and averages probabilities per video for binary classification."""
    model.eval()
    video_probs = {v: [] for v in vid_names}
    video_labels = {}
    video_grades = {}

    with torch.no_grad():
        for batch in tqdm(loader, desc="Classifier Frames Inference (Binary)"):
            inputs, grades, labels, img_names = batch
            inputs = inputs.to(device)

            logits = model(inputs)
            probs = torch.sigmoid(logits).squeeze(-1)

            for i, img_name in enumerate(img_names):
                v_name = match_img_to_video(img_name, vid_names)
                if v_name:
                    video_probs[v_name].append(probs[i].item())
                    video_labels[v_name] = int(labels[i].item())
                    video_grades[v_name] = int(grades[i].item())

    results = {}
    for v_name in vid_names:
        if video_probs[v_name]:
            results[v_name] = {
                "label": video_labels[v_name],
                "prob": sum(video_probs[v_name]) / len(video_probs[v_name]),
                "grade": video_grades[v_name]
            }
        else:
            print(f"Warning: No frames matched for video {v_name}")

    return results


def run_inference_classifier_frames_all(model, loader, device, vid_names, num_classes=5):
    """Predicts on individual frames and averages probabilities per video for 5-class grading."""
    model.eval()
    video_probs = {v: [] for v in vid_names}
    video_labels = {}
    video_grades = {}

    with torch.no_grad():
        for batch in tqdm(loader, desc="Classifier Frames Inference (Multi-class)"):
            inputs, grades, labels, img_names = batch
            inputs = inputs.to(device)

            logits = model(inputs)
            probs = torch.softmax(logits, dim=1)

            for i, img_name in enumerate(img_names):
                v_name = match_img_to_video(img_name, vid_names)
                if v_name:
                    video_probs[v_name].append(probs[i].cpu())
                    video_labels[v_name] = int(labels[i].item())
                    video_grades[v_name] = int(grades[i].item())

    results = {}
    for v_name in vid_names:
        if video_probs[v_name]:
            stacked = torch.stack(video_probs[v_name])
            avg_prob = stacked.mean(dim=0)
            pred_grade = int(torch.argmax(avg_prob).item())

            result = {
                "label": video_labels[v_name],
                "grade": video_grades[v_name],
                "pred_grade": pred_grade,
            }
            for c in range(num_classes):
                result[f"prob_{c}"] = avg_prob[c].item()

            results[v_name] = result
        else:
            print(f"Warning: No frames matched for video {v_name}")

    return results


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    val_trans = make_val_transform_dino(args.img_size, args.complex_augs)

    for split in [1, 2, 3, 4, 5]:
        csv_path = os.path.join(args.output_dir, f"predictions_split_{split}_{args.dataset_name}.csv")

        # 1. Load Video Split (Transformer)
        val_ds = MILVideoDataset(
            json_path=os.path.join(args.annotations_path, f"split_{split}", f"mil_val_{args.dataset_name}.json"),
            num_frames=32,
            transform=val_trans,
            search_dir_paths=[args.videos_path],
        )
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
        vid_names = [sample['vid_name'] for sample in val_ds.data]

        # 2. Load Frame Split (Classifier)
        frame_json_path = os.path.join(args.frame_annotations_path, f"split_{split}", f"frame_val_{args.dataset_name}.json")
        frame_ds = FrameDataset(
            json_path=frame_json_path,
            img_dirs=[args.frame_img_dir],
            transform=val_trans
        )
        frame_loader = DataLoader(frame_ds, batch_size=32, shuffle=False, num_workers=4)

        run_name = f"split_{split}"
        print(f"\n>>> Evaluating Run: {run_name}")

        if args.binary_classification:
            # Transformer
            trans_model = DinoSelfAttention(full_checkpoint_path=args.trans_paths[split - 1]).to(device)
            trans_results = run_inference_transformer(trans_model, val_loader, device)

            # Classifier
            classifier = DinoClassifier(freeze_backbone=True, num_classes=1).to(device)
            state_dict = torch.load(args.classifier_paths[split - 1], map_location=device)
            classifier.load_state_dict(state_dict)
            clf_results = run_inference_classifier_frames(classifier, frame_loader, device, vid_names)

            df_clf = pd.DataFrame.from_dict(clf_results, orient='index')
            df_clf = df_clf.rename(columns={'prob': f"Clf_{run_name}"}).drop(columns=['label', 'grade'], errors='ignore')

            df_trans = pd.DataFrame.from_dict(trans_results, orient='index')
            df_trans = df_trans.rename(columns={'prob': f"Trans_{run_name}"})

            master_df = df_trans.join(df_clf)
            master_df.index.name = 'video'
            master_df = master_df[['label', f"Clf_{run_name}", f"Trans_{run_name}", "grade"]]
            master_df.to_csv(csv_path)

        else:
            # Transformer
            trans_model = DinoSelfAttention(full_checkpoint_path=args.trans_paths[split - 1], num_classes=5).to(device)
            trans_results = run_inference_transformer_all(trans_model, val_loader, device)

            # Classifier
            classifier = DinoClassifier(freeze_backbone=True, num_classes=5).to(device)
            state_dict = torch.load(args.classifier_paths[split - 1], map_location=device)
            classifier.load_state_dict(state_dict)
            clf_results = run_inference_classifier_frames_all(classifier, frame_loader, device, vid_names)

            prob_cols = [f"prob_{c}" for c in range(5)]
            df_clf = pd.DataFrame.from_dict(clf_results, orient='index')
            df_trans = pd.DataFrame.from_dict(trans_results, orient='index')

            clf_rename_dict = {col: f"Clf_{col}" for col in prob_cols}
            clf_rename_dict['pred_grade'] = f"Clf_pred_grade_{run_name}"
            df_clf = df_clf.rename(columns=clf_rename_dict).drop(columns=['label', 'grade'], errors='ignore')

            trans_rename_dict = {col: f"Trans_{col}" for col in prob_cols}
            trans_rename_dict['pred_grade'] = f"Trans_pred_grade_{run_name}"
            df_trans = df_trans.rename(columns=trans_rename_dict)

            master_df = df_trans.join(df_clf)
            master_df.index.name = 'video'

            clf_prob_cols = [f"Clf_{c}" for c in prob_cols]
            trans_prob_cols = [f"Trans_{c}" for c in prob_cols]

            ordered_cols = (
                ['label', 'grade', f"Clf_pred_grade_{run_name}", f"Trans_pred_grade_{run_name}"]
                + clf_prob_cols
                + trans_prob_cols
            )

            master_df = master_df[ordered_cols]
            master_df.to_csv(csv_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--trans_paths', nargs='+', required=True, help="List of 5 Transformer checkpoint paths")
    parser.add_argument('--classifier_paths', nargs='+', required=True, help="List of 5 Classifier checkpoint paths")

    # Frame dataset arguments
    parser.add_argument('--frame_annotations_path', type=str, default="data/stratified_splits")
    parser.add_argument('--frame_img_dir', default="data/2024_Paxos_Frames/cropped_matched_frames")

    parser.add_argument('--dataset_name', type=str, required=True, default="paxos2020")
    parser.add_argument('--annotations_path', type=str, default="data/stratified_splits")
    parser.add_argument('--videos_path', type=str, default="data/ensemble_results_paxos2020/cleaned_videos")
    parser.add_argument('--output_dir', type=str, default="multi_eval")
    parser.add_argument('--img_size', type=int, default=512)
    parser.add_argument('--complex_augs', action='store_true')
    parser.add_argument('--binary_classification', action='store_true')

    args = parser.parse_args()
    main(args)