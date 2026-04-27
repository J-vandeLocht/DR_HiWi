import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import seaborn as sns
from PIL import Image
from torch.utils.data import DataLoader
from sklearn.metrics import (accuracy_score, f1_score, cohen_kappa_score,
                             precision_recall_curve, auc, confusion_matrix)
from sklearn.preprocessing import label_binarize
from pytorch_grad_cam import ScoreCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


def save_scorecam_grid(model, loader, device, output_path):
    model.eval()
    collected_images = []

    # Collect 9 random images from the shuffled loader
    for inputs, _ in loader:
        for i in range(inputs.size(0)):
            collected_images.append(inputs[i])
            if len(collected_images) == 9: break
        if len(collected_images) == 9: break

    if not collected_images: return

    h, w = collected_images[0].shape[1], collected_images[0].shape[2]
    grid_img = np.zeros((h * 3, w * 3, 3), dtype=np.uint8)
    cam = ScoreCAM(model=model, target_layers=[model.features[-1]])

    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)

    for i in range(len(collected_images)):
        input_tensor = collected_images[i].unsqueeze(0).to(device)
        cam_input = torch.nn.functional.interpolate(input_tensor, size=(256, 256), mode='bilinear')

        with torch.no_grad():
            output = model(cam_input)
            pred = torch.argmax(output, dim=1).item()

        targets = [ClassifierOutputTarget(pred)]
        cam_map = cam(input_tensor=cam_input, targets=targets)[0]
        cam_map = cv2.resize(cam_map, (w, h))

        # Denormalize image
        img = input_tensor[0].detach().cpu().permute(1, 2, 0).numpy()
        img = np.clip((img * std) + mean, 0, 1)

        heatmap = cv2.applyColorMap(np.uint8(255 * cam_map), cv2.COLORMAP_TURBO)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0

        overlay = np.clip(0.6 * img + 0.4 * heatmap, 0, 1)
        row, col = i // 3, i % 3
        grid_img[row * h:(row + 1) * h, col * w:(col + 1) * w] = (overlay * 255).astype(np.uint8)

    Image.fromarray(grid_img).save(output_path)


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

    # Save Score-CAM (using a subset or the full loader)
    scorecam_path = os.path.join(run_dir, f"scorecam_epoch_{epoch}.png")
    save_scorecam_grid(model, loader, device, scorecam_path)

    return metrics