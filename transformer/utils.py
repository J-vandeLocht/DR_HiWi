import torch
import numpy as np
from tqdm import tqdm
from misc.utils import binary_metrics, multiclass_metrics

def train_one_epoch_trans(model, loader, criterion, optimizer, binary_classification, device):
    """
    Trains one epoch of the Transformer sequence aggregator model.
    """
    model.train()
    total_loss = 0

    for sequence, label, grade in tqdm(loader, desc="Training"):
        # sequence shape from loader: [1, num_frames, C, H, W] -> Squeeze out batch dim
        sequence = sequence.squeeze(0).to(device)
        if binary_classification:
            target = label.to(device).float().unsqueeze(1)  # [1, 1]
        else:
            target = grade.to(device).long()  # [1]

        optimizer.zero_grad()

        # Model returns (logits, frame_attention_weights)
        logits, _ = model(sequence)
        loss = criterion(logits, target)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


# def train_one_epoch_trans_rope(model, loader, criterion, optimizer, device):
#     """
#     Trains one epoch of the Transformer sequence aggregator model.
#     """
#     model.train()
#     total_loss = 0
#
#     for sequence, label, _, frame_positions in tqdm(loader, desc="Training"):
#         # sequence shape from loader: [1, num_frames, C, H, W] -> Squeeze out batch dim
#         sequence = sequence.squeeze(0).to(device)
#         label = label.to(device).unsqueeze(1)  # [1, 1]
#
#         optimizer.zero_grad()
#
#         # Model returns (logits, frame_attention_weights)
#         logits, _ = model(sequence, frame_positions)
#         loss = criterion(logits, label)
#
#         loss.backward()
#         optimizer.step()
#
#         total_loss += loss.item()
#
#     return total_loss / len(loader)


def validate_extended_trans(model, loader, criterion, binary_classification, device):
    """
    Extended validation loop for tracking sequential frame sequence metrics
    such as PR-AUC, ROC-AUC, F1, Precision, and Recall (binary), or
    QWK, macro F1/precision/recall, and per-class breakdowns (multiclass).
    """
    model.eval()
    running_loss = 0.0
    y_true, y_probs = [], []

    with torch.no_grad():
        for inputs, labels, grades in loader:
            inputs = inputs.squeeze(0).to(device)
            if binary_classification:
                target = labels.to(device).float().unsqueeze(1)
            else:
                target = grades.to(device).long()

            outputs = model(inputs)

            # Safely unpack the tuple output (logits, attention_weights)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs

            loss = criterion(logits, target)
            running_loss += loss.item()

            if binary_classification:
                probs = torch.sigmoid(logits)          # (1,1)
            else:
                probs = torch.softmax(logits, dim=1)     # (1,5)

            y_probs.extend(probs.cpu().numpy())
            y_true.extend(target.cpu().numpy())

    y_true = np.array(y_true).reshape(-1) if binary_classification else np.array(y_true)
    y_probs = np.array(y_probs)

    if binary_classification:
        y_pred = (y_probs.reshape(-1) > 0.5).astype(float)
        metrics = binary_metrics(y_true, y_pred, y_probs.reshape(-1))
    else:
        y_pred = y_probs.argmax(axis=1)
        metrics = multiclass_metrics(y_true, y_pred, y_probs, num_classes=5)

    metrics['loss'] = running_loss / len(loader)
    metrics['y_true'] = y_true
    metrics['y_pred'] = y_pred
    return metrics


# def validate_extended_trans_rope(model, loader, criterion, device):
#     """
#     Extended validation loop for tracking sequential frame sequence metrics
#     such as PR-AUC, ROC-AUC, F1, Precision, and Recall.
#     """
#     model.eval()
#     running_loss = 0.0
#     y_true, y_probs = [], []
#
#     with torch.no_grad():
#         for inputs, labels, _, frame_positions in loader:
#             inputs = inputs.squeeze(0).to(device)
#             labels = labels.to(device).float().unsqueeze(1)
#
#             outputs = model(inputs, frame_positions)
#
#             # Safely unpack the tuple output (logits, attention_weights)
#             logits = outputs[0] if isinstance(outputs, tuple) else outputs
#
#             loss = criterion(logits, labels)
#             running_loss += loss.item()
#
#             y_probs.extend(torch.sigmoid(logits).cpu().numpy())
#             y_true.extend(labels.cpu().numpy())
#
#     y_true = np.array(y_true)
#     y_probs = np.array(y_probs)
#     y_pred = (y_probs > 0.5).astype(float)
#
#     # Metric Curves
#     prec_curve, rec_curve, _ = precision_recall_curve(y_true, y_probs)
#
#     return {
#         'loss': running_loss / len(loader),
#         'acc': accuracy_score(y_true, y_pred),
#         'f1': f1_score(y_true, y_pred, zero_division=0),
#         'precision': precision_score(y_true, y_pred, zero_division=0),
#         'recall': recall_score(y_true, y_pred, zero_division=0),
#         'roc_auc': roc_auc_score(y_true, y_probs) if len(np.unique(y_true)) > 1 else 0.5,
#         'pr_auc': auc(rec_curve, prec_curve),
#         'y_true': y_true,
#         'y_pred': y_pred
#     }