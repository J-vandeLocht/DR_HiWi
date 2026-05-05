import os
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, \
    precision_recall_curve, auc
from data.dataset import MILVideoDatasetNew
from .vis_attention import DinoMIL
from dino.utils import make_val_transform_dino


def validate_extended_mil(model, loader, criterion, device, output_file="attention_weights_log.txt"):
    """
    Extended validation for MIL to calculate PR-AUC, F1, etc.
    MODIFIED: Now intercepts Softmax and Raw weights and saves them to a TXT file.
    """
    model.eval()
    running_loss = 0.0
    y_true, y_probs = [], []

    print(f"--- Starting Evaluation. Saving weights to: {output_file} ---")

    # Open the file and write a header line
    with open(output_file, 'w') as f:
        f.write("Video_Index\tTrue_Label\tPred_Prob\tSoftmax_Weights_32\tRaw_Scores_32\n")

        with torch.no_grad():
            for idx, (inputs, labels) in enumerate(loader):
                # inputs shape from loader: [1, 32, 3, H, W] -> squeeze -> [32, 3, H, W]
                inputs = inputs.squeeze(0).to(device)
                labels = labels.to(device).float().unsqueeze(1)

                # --- 1. Forward Pass ---
                # Unpack the 3 outputs from the updated DinoMIL
                outputs = model(inputs)
                if isinstance(outputs, tuple) and len(outputs) == 3:
                    logits, weights, raw_weights = outputs
                else:
                    raise RuntimeError("Model did not return (logits, weights, raw_weights). Check DinoMIL forward().")

                # --- 2. Standard Metric Logging ---
                loss = criterion(logits, labels)
                running_loss += loss.item()

                pred_prob = torch.sigmoid(logits).item()
                y_probs.append(pred_prob)

                # Assuming batch_size=1, extract single label
                true_label = int(labels.item())
                y_true.append(true_label)

                # --- 3. Weight Extraction & Formatting ---
                # Squeeze to 1D arrays of length 32
                w_list = weights.squeeze().cpu().tolist()
                r_list = raw_weights.squeeze().cpu().tolist()

                # Format as comma-separated strings to keep them neatly on one line
                w_str = ",".join([f"{w:.5f}" for w in w_list])
                r_str = ",".join([f"{r:.5f}" for r in r_list])

                # Write the row for this video
                # Format: Index <tab> Label <tab> Pred <tab> w1,w2...w32 <tab> r1,r2...r32
                f.write(f"{idx}\t{true_label}\t{pred_prob:.4f}\t{w_str}\t{r_str}\n")

    y_true = np.array(y_true)
    y_probs = np.array(y_probs)
    y_pred = (y_probs > 0.5).astype(float)

    # Metric Calculations
    prec_curve, rec_curve, _ = precision_recall_curve(y_true, y_probs)

    metrics = {
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

    print(f"--- Evaluation Complete. Weights successfully saved. ---")
    return metrics


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.BCEWithLogitsLoss()
    val_trans = make_val_transform_dino(512, True)
    val_ds = MILVideoDatasetNew(os.path.join("data/stratified_splits/split_1", "mil_val.json"),
                                num_frames=32,
                                transform=val_trans)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=1, shuffle=False)
    path = os.path.join("mil", "models", f"dino_mil_complex_split_{1}", "best_mil_model.pth")
    model = DinoMIL(checkpoint_path=path).to(device)
    # Assuming you have your model, val_loader, criterion, and device ready:

    # 1. Define where you want the file saved
    log_path = os.path.join("mil_viz", "eval_weights_split1.txt")
    os.makedirs("mil_viz", exist_ok=True)

    # 2. Run the evaluation
    m = validate_extended_mil(model, val_loader, criterion, device, output_file=log_path)

    print(f"Validation PR-AUC: {m['pr_auc']:.4f}")