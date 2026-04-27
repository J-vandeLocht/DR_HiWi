from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import os
import torch
import cv2
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def save_confusion_matrix(y_true, y_pred, epoch, run_dir):
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - Epoch {epoch}')

    save_path = os.path.join(run_dir, f"cm_epoch_{epoch:03d}.png")
    plt.savefig(save_path)
    plt.close()


def apply_clahe_cv2(img_pil):
    img_np = np.array(img_pil)
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    return Image.fromarray(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))


def visualize_full_video_attention(model, video_path, transform, device, epoch, output_dir="mil_viz"):
    os.makedirs(output_dir, exist_ok=True)
    model.eval()

    # 1. Load EVERY frame from the video
    cap = cv2.VideoCapture(video_path)
    all_frames = []
    original_frames = []  # Keep for saving (denormalized)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert for model
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)

        # We store the transformed tensor and a "display" version
        all_frames.append(transform(img_pil))
        original_frames.append(img_rgb)
    cap.release()

    if not all_frames:
        return

    # 2. Extract features in mini-batches (to avoid GPU OOM)
    # We only need the 'h' (features) from the model's backbone
    all_tensors = torch.stack(all_frames).to(device)  # [Total_Frames, 3, H, W]
    feature_list = []

    batch_size = 16  # Process in chunks
    with torch.no_grad():
        for i in range(0, len(all_tensors), batch_size):
            chunk = all_tensors[i:i + batch_size]
            # Use the feature_extractor part of your GatedAttentionMIL
            features = model.feature_extractor(chunk).view(chunk.size(0), -1)
            feature_list.append(features)

        h = torch.cat(feature_list, dim=0)  # [Total_Frames, 1280]

        # 3. Calculate Attention on the full sequence
        a_v = model.attention_V(h)
        a_u = model.attention_U(h)
        a = model.attention_w(a_v * a_u)
        weights = torch.softmax(a, dim=0).squeeze()  # [Total_Frames]

    # 4. Get Top 4 and Bottom 4 indices
    num_to_save = min(4, len(weights))
    top_vals, top_idx = torch.topk(weights, num_to_save)
    bot_vals, bot_idx = torch.topk(weights, num_to_save, largest=False)

    # 5. Save individual images with weights in filename
    video_id = os.path.basename(video_path).split('.')[0]

    def save_subset(indices, values, prefix):
        for i in range(len(indices)):
            idx = indices[i].item()
            weight = values[i].item()

            # Get the original frame
            img_out = original_frames[idx]
            img_out = cv2.cvtColor(img_out, cv2.COLOR_RGB2BGR)

            filename = f"ep{epoch}_{video_id}_{prefix}_rank{i}_weight_{weight:.4f}.png"
            cv2.imwrite(os.path.join(output_dir, filename), img_out)

    save_subset(top_idx, top_vals, "TOP")
    save_subset(bot_idx, bot_vals, "BOT")

    print(f"--- Saved full video attention for {video_id} (Total frames: {len(all_frames)}) ---")


def save_gradcam_plusplus_grid(model, loader, device, epoch, run_dir):
    model.eval()
    collected_images = []
    with torch.no_grad():
        for inputs, _ in loader:
            for i in range(inputs.size(0)):
                collected_images.append(inputs[i])
                if len(collected_images) == 9: break
            if len(collected_images) == 9: break

    num_to_process = len(collected_images)
    if num_to_process == 0: return

    input_samples = torch.stack(collected_images).to(device).requires_grad_(True)
    target_layers = [model.features[-1]]
    cam = GradCAMPlusPlus(model=model, target_layers=target_layers)
    targets = [ClassifierOutputTarget(0) for _ in range(num_to_process)]

    grayscale_cams = cam(input_tensor=input_samples, targets=targets)
    h, w = input_samples.shape[2], input_samples.shape[3]
    grid_img = np.zeros((h * 3, w * 3, 3), dtype=np.uint8)

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    for i in range(num_to_process):
        img_np = input_samples[i].detach().permute(1, 2, 0).cpu().numpy()
        img_np = (img_np * std) + mean
        img_np = np.clip(img_np, 0, 1)
        vis = show_cam_on_image(img_np, grayscale_cams[i], use_rgb=True)
        row, col = i // 3, i % 3
        grid_img[row * h:(row + 1) * h, col * w:(col + 1) * w, :] = vis

    save_path = os.path.join(run_dir, f"gradcam_epoch_{epoch:03d}.png")
    Image.fromarray(grid_img).save(save_path)


def save_preprocessing_check(loader, output_name, num_images=9):
    """
    Saves a 3x3 grid of images from a loader to verify preprocessing/augmentations.
    Reverses normalization so colors look correct.
    """
    collected = []
    # 1. Collect images until we have enough
    for images, labels in loader:
        for i in range(images.size(0)):
            collected.append(images[i])
            if len(collected) == num_images:
                break
        if len(collected) == num_images:
            break

    # 2. Setup Grid Parameters
    h, w = collected[0].shape[1], collected[0].shape[2]
    grid_img = np.zeros((h * 3, w * 3, 3), dtype=np.uint8)

    # ImageNet stats to reverse normalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    # 3. Denormalize and stitch
    for i in range(len(collected)):
        # Convert (C, H, W) -> (H, W, C)
        img_np = collected[i].permute(1, 2, 0).cpu().numpy()

        # Reverse: (x * std) + mean
        img_np = (img_np * std) + mean
        img_np = np.clip(img_np, 0, 1) * 255
        img_np = img_np.astype(np.uint8)

        row, col = i // 3, i % 3
        grid_img[row * h:(row + 1) * h, col * w:(col + 1) * w, :] = img_np

    save_path = f"{output_name}.png"
    Image.fromarray(grid_img).save(save_path)
