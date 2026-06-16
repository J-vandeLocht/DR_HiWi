import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torchvision.transforms import v2
from dino.utils import apply_clahe_cv2_dino


# --- 1. MODEL DEFINITION ---
# Updated to return (logits, softmax_weights, raw_logits)
class EvalDinoMIL(nn.Module):
    def __init__(self, checkpoint_path=None, freeze_backbone=True, D=512, K=1):
        super(EvalDinoMIL, self).__init__()
        # 1. Load the DINOv3 backbone
        self.backbone = torch.hub.load(
            "dino/dinov3", "dinov3_vitl16", source="local",
            weights="dino/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
        )
        self.L = self.backbone.embed_dim

        # 2. Define the Attention and Classifier modules FIRST
        # (They must exist before we can load weights into them)
        self.attention_V = nn.Sequential(nn.Linear(self.L, D), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(self.L, D), nn.Sigmoid())
        self.attention_w = nn.Linear(D, K)
        self.classifier = nn.Sequential(
            nn.Linear(self.L, 512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 1)
        )

        if checkpoint_path is not None:
            print(f"Loading full checkpoint from {checkpoint_path}")
            state_dict = torch.load(checkpoint_path, map_location="cpu")

            # 1. Load Backbone (using the 'backbone.' prefix)
            backbone_state = {k.replace("backbone.", ""): v for k, v in state_dict.items() if k.startswith("backbone.")}
            self.backbone.load_state_dict(backbone_state, strict=True)  # Set to True to ensure backbone is perfect
            print("Backbone loaded strictly and successfully.")

            # 2. Load MIL Head layers specifically
            # We map the keys in the state_dict directly to the sub-modules
            mil_modules = {
                'attention_V': self.attention_V,
                'attention_U': self.attention_U,
                'attention_w': self.attention_w,
                'classifier': self.classifier
            }

            for name, module in mil_modules.items():
                # Extract only keys belonging to this specific module (e.g., 'classifier.0.weight')
                mod_state = {k.replace(f"{name}.", ""): v for k, v in state_dict.items() if k.startswith(f"{name}.")}
                if mod_state:
                    module.load_state_dict(mod_state, strict=True)
                    print(f"Module '{name}' loaded strictly.")
                else:
                    print(f"Warning: No weights found for {name}")

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x):
        """
        x: [N, 3, H, W] where N is number of frames
        """
        feats = self.backbone.forward_features(x)
        h = feats["x_norm_clstoken"]  # [N, L]

        # Gated Attention Mechanism
        a_v = self.attention_V(h)
        a_u = self.attention_U(h)
        a = self.attention_w(a_v * a_u)  # [N, 1] - Raw Logits

        weights = torch.softmax(a, dim=0)  # [N, 1] - Softmax Weights

        bag_rep = torch.sum(weights * h, dim=0)
        logits = self.classifier(bag_rep.unsqueeze(0))

        # Return Logits, Softmax weights, and Raw logits (flattened)
        return logits, weights, a.flatten()


def get_processed_data(video_path, img_size, norm_transform, indices=None):
    """ Replicates the MILVideoDatasetNew loading and transform order exactly. """
    cap = cv2.VideoCapture(video_path)
    resizer = v2.Resize((img_size, img_size), antialias=True)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # If no indices provided, we take the whole video
    target = indices if indices is not None else list(range(total_frames))

    model_ready, display_ready = [], []
    current_f = 0

    while current_f <= max(target):
        ret, frame = cap.read()
        if not ret: break
        if current_f in target:
            # Match Training: Resize -> CLAHE -> Normalize
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_tensor = v2.functional.to_image(img_rgb)
            img_resized = resizer(img_tensor)
            img_clahe = apply_clahe_cv2_dino(img_resized)

            # Display: HWC (NumPy) | Model: Normalized Tensor
            display_ready.append(img_clahe.permute(1, 2, 0).numpy())
            model_ready.append(norm_transform(img_clahe))
        current_f += 1
    cap.release()
    return torch.stack(model_ready), display_ready


# --- 3. VISUALIZATION FUNCTIONS ---

def visualize_full_report(model, video_path, img_size, device, split, norm_trans, output_dir="mil_viz"):
    """ Analysis of EVERY frame. Shows Raw scores and Global Softmax. """
    os.makedirs(output_dir, exist_ok=True)
    video_id = os.path.basename(video_path).split('.')[0]

    inputs, display_frames = get_processed_data(video_path, img_size, norm_trans)

    model.eval()
    raw_scores = []
    with torch.no_grad():
        for i in range(0, len(inputs), 16):
            _, _, raw_a = model(inputs[i:i + 16].to(device))
            raw_scores.append(raw_a.cpu().numpy())

    raw_scores = np.concatenate(raw_scores)
    # Global Softmax: how these frames compare across the whole video
    full_weights = torch.softmax(torch.from_numpy(raw_scores), dim=0).numpy()

    idx_sort = np.argsort(raw_scores)
    top_idx, bot_idx = idx_sort[-8:][::-1], idx_sort[:8]

    # Layout Setup
    DPI = 100
    fig = plt.figure(figsize=(32, 18), dpi=DPI)
    fig.suptitle(f"FULL VIDEO ANALYSIS | {video_id} | Split: {split}", fontsize=40, fontweight='bold', y=0.98)
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.2], wspace=0.1)
    gs_grid = gridspec.GridSpecFromSubplotSpec(4, 4, subplot_spec=gs[0], wspace=0.05, hspace=0.35)

    def plot_grid(indices, start_row, label, color):
        for i, idx in enumerate(indices):
            ax = fig.add_subplot(gs_grid[start_row + (i // 4), i % 4])
            ax.imshow(np.clip(display_frames[idx], 0, 1), interpolation='none')
            ax.set_title(f"{label} {i + 1}\nW: {full_weights[idx]:.5f}\n(Raw: {raw_scores[idx]:.3f})",
                         fontsize=15, color=color, fontweight='bold')
            ax.axis('off')

    plot_grid(top_idx, 0, "TOP", "darkgreen")
    plot_grid(bot_idx, 2, "BOT", "darkred")

    ax_plot = fig.add_subplot(gs[1])
    ax_plot.plot(raw_scores, label="Raw Logits", alpha=0.7)
    ax_plot.set_title("Temporal Attention Distribution", fontsize=30)
    ax_plot.set_ylabel("Raw Score"), ax_plot.set_xlabel("Frame Index")

    plt.savefig(os.path.join(output_dir, f"full_{video_id}_split{split}.png"), bbox_inches='tight')
    plt.close()


def visualize_eval_simulation(model, video_path, img_size, device, split, norm_trans, output_dir="mil_viz"):
    """ Analysis of 32 uniform frames. Exactly matches Evaluation/Validation. """
    os.makedirs(output_dir, exist_ok=True)
    video_id = os.path.basename(video_path).split('.')[0]

    cap = cv2.VideoCapture(video_path)
    total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    sample_indices = np.linspace(0, total_f - 1, 32, dtype=int)

    inputs, display_frames = get_processed_data(video_path, img_size, norm_trans, indices=sample_indices)

    model.eval()
    with torch.no_grad():
        logits, weights, raw_a = model(inputs.to(device))
        weights = weights.squeeze().cpu().numpy()
        raw_scores = raw_a.cpu().numpy()
        prob = torch.sigmoid(logits).item()

    idx_sort = np.argsort(weights)
    top_idx, bot_idx = idx_sort[-8:][::-1], idx_sort[:8]

    fig = plt.figure(figsize=(32, 18), dpi=200)
    fig.suptitle(f"EVAL SIMULATION (32 Frames) | Prediction: {prob:.4f}\n{video_id} | Split: {split}",
                 fontsize=40, fontweight='bold', y=0.98)
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.2], wspace=0.1)
    gs_grid = gridspec.GridSpecFromSubplotSpec(4, 4, subplot_spec=gs[0], wspace=0.05, hspace=0.35)

    def plot_grid_sim(indices, start_row, label, color):
        for i, idx in enumerate(indices):
            ax = fig.add_subplot(gs_grid[start_row + (i // 4), i % 4])
            ax.imshow(np.clip(display_frames[idx], 0, 1), interpolation='none')
            ax.set_title(f"{label} {i + 1}\nW: {weights[idx]:.4f}\n(Raw: {raw_scores[idx]:.3f})",
                         fontsize=15, color=color, fontweight='bold')
            ax.axis('off')

    plot_grid_sim(top_idx, 0, "TOP", "darkgreen")
    plot_grid_sim(bot_idx, 2, "BOT", "darkred")

    ax_bar = fig.add_subplot(gs[1])
    ax_bar.bar(range(32), weights, color='tab:blue', alpha=0.7)
    ax_bar.set_xticks(range(32)), ax_bar.set_xticklabels(sample_indices, rotation=90)
    ax_bar.set_title("Bag Softmax Weights", fontsize=30)

    plt.savefig(os.path.join(output_dir, f"eval_{video_id}_split{split}.png"), bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    videos = [
        "ensemble_results/cleaned_videos/CLEAN_2024_02_11_12_26_IMG_4608 LE MILD NPDR.mp4",
        "ensemble_results/cleaned_videos/CLEAN_2024_03_30_11_37_IMG_4851 RE MILD NPDR.mp4",
        "ensemble_results/cleaned_videos/CLEAN_2024_04_02_12_48_IMG_4883 RE SEV NPDR WITH DME.mp4",
        "ensemble_results/cleaned_videos/CLEAN_R058R2.mp4"
    ]
    split_paths = [
        ("mil/models/dino_mil_complex_split_1_segmented_0512_1705", 1),
        ("mil/models/dino_mil_complex_split_2_segmented_0512_1824", 2),
        ("mil/models/dino_mil_complex_split_3_segmented_0512_1945", 3),
        ("mil/models/dino_mil_complex_split_4_segmented_0512_2107", 4),
        ("mil/models/dino_mil_complex_split_5_segmented_0514_1205", 5)
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Define normalization (standard ImageNet used in DinoV3)
    norm_transform = v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    for (path, split) in split_paths:
        model_path = os.path.join(path, "best_mil_model.pth")
        if not os.path.exists(path): continue

        print(f"Processing split in {path}...")
        model = EvalDinoMIL(checkpoint_path=model_path).to(device)

        for vid in videos:
            print(f"Analyzing: {os.path.basename(vid)}")
            # Run both reports
            visualize_full_report(model, vid, 512, device, split, norm_transform)
            visualize_eval_simulation(model, vid, 512, device, split, norm_transform)