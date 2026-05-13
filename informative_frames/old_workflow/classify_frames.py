import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from torchvision import models, transforms, datasets
from tqdm import tqdm
import pandas as pd
import numpy as np
import os
import cv2
from PIL import Image
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Added medical metrics
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_recall_curve, auc, confusion_matrix, \
    precision_score, recall_score

# --- Configuration ---
DATA_DIR = 'data/labeled_and_cropped_dataset'
MODEL_SAVE_PATH = 'models'
IMG_SIZE = 512
BATCH_SIZE = 16
EPOCHS = 5  # Increased epochs
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Iterate through models seamlessly
MODELS_TO_TEST = ['b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7']


class TransformSubset(torch.utils.data.Dataset):
    """Wrapper to apply transforms to a Subset of a dataset."""

    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform: x = self.transform(x)
        return x, y

    def __len__(self): return len(self.subset)


def apply_clahe_cv2(img_pil):
    img_np = np.array(img_pil)
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return Image.fromarray(cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB))


def save_confusion_matrix(y_true, y_pred, epoch, run_name):
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted (1=Informative, 0=Non-Informative)')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - Epoch {epoch}')

    save_path = os.path.join(MODEL_SAVE_PATH, f"{run_name}_cm_epoch_{epoch:03d}.png")
    plt.savefig(save_path)
    plt.close()


def main():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

    # 1. Group by Video ID
    full_dataset = datasets.ImageFolder(root=DATA_DIR, transform=None)
    video_to_indices = {}

    for i, (path, label) in enumerate(full_dataset.samples):
        fname = os.path.basename(path)
        video_id = fname.rsplit('_f', 1)[0]
        if video_id not in video_to_indices:
            video_to_indices[video_id] = []
        video_to_indices[video_id].append(i)

    # 2. Split Video IDs (80/20)
    unique_videos = list(video_to_indices.keys())
    np.random.shuffle(unique_videos)
    split_idx = int(0.8 * len(unique_videos))
    train_vids, val_vids = unique_videos[:split_idx], unique_videos[split_idx:]

    train_indices = [idx for vid in train_vids for idx in video_to_indices[vid]]
    val_indices = [idx for vid in val_vids for idx in video_to_indices[vid]]

    # 3. Save Methodical Split Info
    split_info = []
    for vid in unique_videos:
        split = 'train' if vid in train_vids else 'val'
        for idx in video_to_indices[vid]:
            path, label = full_dataset.samples[idx]
            split_info.append({'filename': os.path.basename(path), 'video_id': vid, 'label': label, 'split': split})
    pd.DataFrame(split_info).to_csv(f'split_info_{timestamp}.csv', index=False)

    # 4. Data Loaders
    train_trans = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.Lambda(apply_clahe_cv2),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_trans = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.Lambda(apply_clahe_cv2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_loader = DataLoader(TransformSubset(Subset(full_dataset, train_indices), train_trans), batch_size=BATCH_SIZE,
                              shuffle=True)
    val_loader = DataLoader(TransformSubset(Subset(full_dataset, val_indices), val_trans), batch_size=BATCH_SIZE,
                            shuffle=False)

    # Mapping to instantiate the different models dynamically
    efficientnet_weights = {
        'b0': (models.efficientnet_b0, models.EfficientNet_B0_Weights.DEFAULT),
        'b1': (models.efficientnet_b1, models.EfficientNet_B1_Weights.DEFAULT),
        'b2': (models.efficientnet_b2, models.EfficientNet_B2_Weights.DEFAULT),
        'b3': (models.efficientnet_b3, models.EfficientNet_B3_Weights.DEFAULT),
        'b4': (models.efficientnet_b4, models.EfficientNet_B4_Weights.DEFAULT),
        'b5': (models.efficientnet_b5, models.EfficientNet_B5_Weights.DEFAULT),
        'b6': (models.efficientnet_b6, models.EfficientNet_B6_Weights.DEFAULT),
        'b7': (models.efficientnet_b7, models.EfficientNet_B7_Weights.DEFAULT)
    }

    # 5. Model & Training Loop
    for model_name in MODELS_TO_TEST:
        run_name = f"Informative_{model_name.upper()}_{timestamp}"
        print(f"\n--- Starting Training for {model_name.upper()} ---")

        # Initialize a new writer for each model to keep TensorBoard clean
        writer = SummaryWriter(log_dir=os.path.join("runs", run_name))

        # Setup Model
        model_fn, weights = efficientnet_weights[model_name]
        model = model_fn(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
        model.to(DEVICE)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=LR)

        # Scheduler: Drop LR by a factor of 10 every 10 epochs
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

        best_pr_auc = 0.0

        for epoch in range(EPOCHS):
            model.train()
            running_loss = 0.0
            for imgs, lbls in tqdm(train_loader, desc=f"{model_name.upper()} Ep {epoch} Train"):
                imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE).float().view(-1, 1)
                optimizer.zero_grad()
                loss = criterion(model(imgs), lbls)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            t_loss = running_loss / len(train_loader)

            # --- Validation & Metrics ---
            model.eval()
            y_true, y_probs = [], []
            v_loss = 0.0

            with torch.no_grad():
                for imgs, lbls in val_loader:
                    imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE).float().view(-1, 1)
                    outputs = model(imgs)
                    loss = criterion(outputs, lbls)
                    v_loss += loss.item()

                    y_probs.extend(torch.sigmoid(outputs).cpu().numpy())
                    y_true.extend(lbls.cpu().numpy())

            v_loss = v_loss / len(val_loader)
            y_true = np.array(y_true)
            y_probs = np.array(y_probs)
            y_pred = (y_probs > 0.5).astype(float)

            # Calculate all extended metrics
            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            roc_auc = roc_auc_score(y_true, y_probs) if len(np.unique(y_true)) > 1 else 0.5
            prec_curve, rec_curve, _ = precision_recall_curve(y_true, y_probs)
            pr_auc = auc(rec_curve, prec_curve)

            # Step the scheduler at the end of the epoch
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']

            print(
                f"Ep {epoch} | LR: {current_lr:.6f} | Val Loss: {v_loss:.4f} | PR-AUC: {pr_auc:.4f} | F1: {f1:.4f} | Acc: {acc:.4f}")

            # TensorBoard Logging
            writer.add_scalar('Loss/train', t_loss, epoch)
            writer.add_scalar('Loss/val', v_loss, epoch)
            writer.add_scalar('LR', current_lr, epoch)
            writer.add_scalar('Metrics/PR_AUC', pr_auc, epoch)
            writer.add_scalar('Metrics/F1', f1, epoch)
            writer.add_scalar('Metrics/Accuracy', acc, epoch)
            writer.add_scalar('Metrics/ROC_AUC', roc_auc, epoch)
            writer.add_scalar('Metrics/Precision', precision, epoch)
            writer.add_scalar('Metrics/Recall', recall, epoch)

            # Save visuals and best model
            save_confusion_matrix(y_true, y_pred, epoch, run_name)

            if pr_auc > best_pr_auc:
                best_pr_auc = pr_auc
                torch.save(model.state_dict(), os.path.join(MODEL_SAVE_PATH, f"best_{run_name}.pth"))

        writer.close()
        print(f"Finished {model_name.upper()}. Best PR-AUC: {best_pr_auc:.4f}\n")


if __name__ == "__main__":
    main()