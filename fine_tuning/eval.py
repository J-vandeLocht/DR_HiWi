import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import models, transforms
from sklearn.metrics import accuracy_score, f1_score, precision_recall_curve, auc, confusion_matrix
from sklearn.preprocessing import label_binarize
import seaborn as sns

# Score-CAM imports
from pytorch_grad_cam import ScoreCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from .train import apply_clahe_cv2
from data.dataset import KaggleDRDataset


def save_scorecam_grid(model, loader, device, run_name):
    output_dir = f"eval_results_{run_name}"
    os.makedirs(output_dir, exist_ok=True)
    model.eval()

    collected_images = []
    collected_labels = []

    # Collect 9 images
    for inputs, labels in loader:
        for i in range(inputs.size(0)):
            collected_images.append(inputs[i])
            collected_labels.append(labels[i])
            if len(collected_images) == 9:
                break
        if len(collected_images) == 9:
            break

    if len(collected_images) == 0:
        print("No images found.")
        return

    h, w = collected_images[0].shape[1], collected_images[0].shape[2]
    grid_img = np.zeros((h * 3, w * 3, 3), dtype=np.uint8)

    # Initialize Score-CAM
    cam = ScoreCAM(model=model, target_layers=[model.features[-1]])
    cam.batch_size = 16  # prevent internal explosion

    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)

    for i in range(len(collected_images)):
        input_tensor = collected_images[i].unsqueeze(0).to(device)

        # --- Downscale for CAM (CRUCIAL for memory) ---
        cam_input = torch.nn.functional.interpolate(
            input_tensor, size=(256, 256), mode='bilinear', align_corners=False
        )

        # --- Predicted class target ---
        with torch.no_grad():
            output = model(cam_input)
            pred = torch.argmax(output, dim=1).item()

        targets = [ClassifierOutputTarget(pred)]

        # --- Generate CAM ---
        cam_map = cam(input_tensor=cam_input, targets=targets)[0]

        # Resize CAM back to original size
        cam_map = cv2.resize(cam_map, (w, h))

        # Normalize CAM
        cam_map = cam_map - np.min(cam_map)
        cam_map = cam_map / (np.max(cam_map) + 1e-8)

        # --- Denormalize image ---
        img = input_tensor[0].detach().cpu().permute(1, 2, 0).numpy()
        img = (img * std) + mean
        img = np.clip(img, 0, 1)

        # --- Heatmap ---
        heatmap = cv2.applyColorMap(
            np.uint8(255 * cam_map), cv2.COLORMAP_TURBO
        )
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0

        # --- Overlay ---
        overlay = (0.6 * img + 0.4 * heatmap)
        overlay = np.clip(overlay, 0, 1)
        overlay = (overlay * 255).astype(np.uint8)

        # Place in grid
        row, col = i // 3, i % 3
        grid_img[row*h:(row+1)*h, col*w:(col+1)*w] = overlay

    save_path = os.path.join(output_dir, "scorecam_grid.png")
    Image.fromarray(grid_img).save(save_path)
    print(f"Saved Score-CAM grid to: {save_path}")


def evaluate_model(weight_path, data_dir, img_size=380, batch_size=32):
    """
    img_size=380 is the standard for EfficientNet-B4.
    Change it if you trained with a different resolution.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_name = os.path.basename(weight_path).split('.')[0]
    output_dir = f"eval_results_{run_name}"
    os.makedirs(output_dir, exist_ok=True)

    print(f"--- Starting Evaluation for: {run_name} ---")
    print(f"Device: {device} | Image Size: {img_size}x{img_size}")

    # 1. Load EfficientNet-B4
    model = models.efficientnet_b4(weights=None)

    # Dynamically match the input features of the classifier and set output to 5 classes
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 5)

    # Load weights
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.to(device).eval()

    # 2. Data Preparation
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.Lambda(apply_clahe_cv2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    dataset = KaggleDRDataset(data_dir, transform=transform)
    # Shuffle=False is important here so labels and predictions align perfectly
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # For Score-CAM, we create a separate loader with shuffle=True so we get
    # a random assortment of 9 images for the grid, rather than just the first 9.
    scorecam_loader = DataLoader(dataset, batch_size=9, shuffle=True)

    all_preds = []
    all_probs = []
    all_labels = []

    print(f"Running Inference on {len(dataset)} images...")

    # 3. Inference Loop
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            outputs = model(imgs)

            # Get probabilities and predictions
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())

    # 4. Calculate Metrics
    acc = accuracy_score(all_labels, all_preds)
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')
    f1_macro = f1_score(all_labels, all_preds, average='macro')

    # PR-AUC Calculation (One-vs-Rest for multi-class)
    y_bin = label_binarize(all_labels, classes=[0, 1, 2, 3, 4])
    probs_np = np.array(all_probs)
    pr_auc_list = []

    for i in range(5):
        # Prevent errors if a class is entirely missing from the validation set
        if np.sum(y_bin[:, i]) > 0:
            precision, recall, _ = precision_recall_curve(y_bin[:, i], probs_np[:, i])
            pr_auc_list.append(auc(recall, precision))
        else:
            pr_auc_list.append(0.0)

    mean_pr_auc = np.mean(pr_auc_list)

    print("\n--- Final Metrics ---")
    print(f"Accuracy:      {acc:.4f}")
    print(f"F1 (Weighted): {f1_weighted:.4f}")
    print(f"F1 (Macro):    {f1_macro:.4f}")
    print(f"PR-AUC (Mean): {mean_pr_auc:.4f}")

    # Write metrics to a text file
    with open(os.path.join(output_dir, "metrics.txt"), "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"F1 (Weighted): {f1_weighted:.4f}\n")
        f.write(f"F1 (Macro): {f1_macro:.4f}\n")
        f.write(f"PR-AUC (Mean): {mean_pr_auc:.4f}\n")

    # 5. Confusion Matrix Plot
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[0, 1, 2, 3, 4], yticklabels=[0, 1, 2, 3, 4])
    plt.xlabel('Predicted Grade')
    plt.ylabel('Actual Grade')
    plt.title(f'Confusion Matrix: {run_name}')
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    print(f"Saved Confusion Matrix to: {cm_path}")

    # 6. Run Score-CAM Grid
    print("Generating Score-CAM visualizations...")
    save_scorecam_grid(model, scorecam_loader, device, run_name)


if __name__ == "__main__":
    WEIGHT_PATH = "fine_tuning/ep25_size512/2026-04-15_13-19-56_Classifier_ep25_size512.pth"
    VAL_DIR = "data/fused_dr_dataset/dr_unified_v2/dr_unified_v2/val"

    evaluate_model(WEIGHT_PATH, VAL_DIR, img_size=512, batch_size=16)