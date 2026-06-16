from tqdm import tqdm
import torch
import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             precision_recall_curve, auc,
                             precision_score, recall_score)


def validate_extended_mil(model, loader, criterion, device):
    # Extended validation for MIL to calculate PR-AUC, F1, etc.
    model.eval()
    running_loss = 0.0
    y_true, y_probs = [], []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.squeeze(0).to(device)
            labels = labels.to(device).float().unsqueeze(1)

            outputs = model(inputs)

            # Handle tuple returns if your MIL model returns (logits, attention_weights)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs

            loss = criterion(logits, labels)
            running_loss += loss.item()

            y_probs.extend(torch.sigmoid(logits).cpu().numpy())
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


def train_one_epoch_mil(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0

    for bag, label in tqdm(loader):
        # bag shape from loader: [1, Bag_Size, C, H, W] -> Need to squeeze batch dim
        bag = bag.squeeze(0).to(device)
        label = label.to(device).unsqueeze(1)  # [1, 1]

        optimizer.zero_grad()

        logits, _ = model(bag)
        loss = criterion(logits, label)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)