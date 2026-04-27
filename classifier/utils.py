import numpy as np
import torch
from tqdm import tqdm

from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             precision_recall_curve, auc,
                             precision_score, recall_score)


def train_one_epoch_classifier(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0

    for images, labels in tqdm(loader, leave=False, desc="Batch"):
        images, labels = images.to(device), labels.to(device).float().unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def validate_classifier(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    y_true, y_probs = [], []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            y_probs.extend(torch.sigmoid(outputs).cpu().numpy())
            y_true.extend(labels.cpu().numpy())

    y_true = np.array(y_true)
    y_probs = np.array(y_probs)
    y_pred = (y_probs > 0.5).astype(float)

    # Metric Calculations
    prec_curve, rec_curve, _ = precision_recall_curve(y_true, y_probs)

    return {
        'loss': running_loss / len(loader),
        'acc': accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_true, y_probs) if len(np.unique(y_true)) > 1 else 0.5,
        'pr_auc': auc(rec_curve, prec_curve),
        'y_true': y_true,
        'y_pred': y_pred
    }
