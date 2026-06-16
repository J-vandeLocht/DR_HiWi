import torch
import torch.nn as nn
from tqdm import tqdm
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, f1_score, cohen_kappa_score,
                             precision_recall_curve, auc, confusion_matrix)
from sklearn.preprocessing import label_binarize


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    pbar = tqdm(loader, leave=False)
    for i, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device).long()
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pbar.set_postfix({'batch_loss': f"{loss.item():.4f}"})
    return total_loss / len(loader)


def evaluate_and_log(model, loader, device, epoch, run_dir):
    model.eval()
    all_preds, all_probs, all_labels = [], [], []
    running_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device).long()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate Metrics
    metrics = {
        'loss': running_loss / len(loader),
        'acc': accuracy_score(all_labels, all_preds),
        'f1_macro': f1_score(all_labels, all_preds, average='macro'),
        'f1_weighted': f1_score(all_labels, all_preds, average='weighted'),
        'qwk': cohen_kappa_score(all_labels, all_preds, weights='quadratic')
    }

    # PR-AUC
    y_bin = label_binarize(all_labels, classes=[0, 1, 2, 3, 4])
    probs_np = np.array(all_probs)
    pr_aucs = []
    for i in range(5):
        if np.sum(y_bin[:, i]) > 0:
            p, r, _ = precision_recall_curve(y_bin[:, i], probs_np[:, i])
            pr_aucs.append(auc(r, p))
    metrics['pr_auc_mean'] = np.mean(pr_aucs) if pr_aucs else 0.0

    # Save Confusion Matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Epoch {epoch} Confusion Matrix')
    plt.savefig(os.path.join(run_dir, f"cm_epoch_{epoch}.png"))
    plt.close()

    # # Save Score-CAM (using a subset or the full loader)
    # scorecam_path = os.path.join(run_dir, f"scorecam_epoch_{epoch}.png")
    # save_scorecam_grid(model, loader, device, scorecam_path)

    return metrics