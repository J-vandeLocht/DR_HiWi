import matplotlib
matplotlib.use("Agg")
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dino.train_dino_classifier import DinoClassifier
from data.dataset import FrameDataset
from dino.utils import make_val_transform_dino

def plot_confidence_histogram(
    model_path,
    repo_dir,
    weight_path,
    data_json,
    data_root,
    img_size=224,
    batch_size=32,
    use_cls=True,
    device="cuda"
):
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # --- Load model ---
    model = DinoClassifier(
        repo_dir=repo_dir,
        weights=weight_path,
        freeze_backbone=True,
        use_cls=use_cls
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # --- Dataset ---
    transform = make_val_transform_dino(img_size, True)

    dataset = FrameDataset(
        data_json,
        data_root,
        transform=transform
    )

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)

            outputs = model(inputs)
            probs = torch.sigmoid(outputs)

            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())

    # --- Flatten ---
    all_probs = np.array(all_probs).ravel()
    all_labels = np.array(all_labels).ravel()

    pos_probs = all_probs[all_labels == 1]
    neg_probs = all_probs[all_labels == 0]

    # --- 1. Combined histogram ---
    plt.figure()
    plt.hist(all_probs, bins=50)
    plt.xlabel("Predicted Probability")
    plt.ylabel("Count")
    plt.title("All Predictions")
    plt.savefig("all_predictions_hist.png", dpi=300)
    plt.close()

    # --- 2. Positive class histogram ---
    plt.figure()
    plt.hist(pos_probs, bins=50)
    plt.xlabel("Predicted Probability")
    plt.ylabel("Count")
    plt.title("Positive Class (y=1)")
    plt.savefig("positive_hist.png", dpi=300)
    plt.close()

    # --- 3. Negative class histogram ---
    plt.figure()
    plt.hist(neg_probs, bins=50)
    plt.xlabel("Predicted Probability")
    plt.ylabel("Count")
    plt.title("Negative Class (y=0)")
    plt.savefig("negative_hist.png", dpi=300)
    plt.close()

    # --- Optional stats ---
    print("All probs percentiles:", np.percentile(all_probs, [0, 25, 50, 75, 90, 99]))
    print("Pos probs percentiles:", np.percentile(pos_probs, [0, 25, 50, 75, 90, 99]))
    print("Neg probs percentiles:", np.percentile(neg_probs, [0, 25, 50, 75, 90, 99]))


if __name__ == "__main__":
    plot_confidence_histogram(
        model_path="classifier/dino_complex_0424_1900/best_model.pth",
        repo_dir="dino/dinov3",
        weight_path="dino/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
        data_json="data/frame_val.json",
        data_root="data/2024_Paxos_Frames/cropped_frames",
        img_size=512,
        batch_size=16,
        use_cls=True,
        device="cuda"
    )
