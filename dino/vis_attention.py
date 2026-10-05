import torch
import torch.nn.functional as F
import os
import re
import math
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader

from data.dataset import MILVideoDataset
from dino.utils import make_val_transform_dino
from dino.train_dino_transformer import DinoSelfAttention


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def infer_model_config(path):
    """Reads num_blocks and num_classes from a DinoSelfAttention checkpoint so they
    never have to be passed by hand (the strict loading needs an exact match)."""
    state_dict = torch.load(path, map_location="cpu")

    block_ids = {int(m.group(1)) for k in state_dict if (m := re.match(r"blocks\.(\d+)\.", k))}
    if not block_ids:
        raise ValueError(f"No 'blocks.N.' keys found in {path}. Is this a DinoSelfAttention checkpoint?")
    num_blocks = max(block_ids) + 1

    # The last Linear of the classifier head holds num_classes in its output dim.
    head_ids = {int(m.group(1)) for k in state_dict if (m := re.match(r"classifier\.(\d+)\.weight$", k))}
    num_classes = state_dict[f"classifier.{max(head_ids)}.weight"].shape[0]

    return num_blocks, num_classes


def make_thumbnails(inputs, thumb_size):
    """Unnormalizes a (T, 3, H, W) batch and resizes it to height `thumb_size`
    (aspect ratio kept). Returns a (T, h, w, 3) float numpy array in [0, 1]."""
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    imgs = (inputs * std + mean).clamp(0, 1)

    h, w = imgs.shape[-2:]
    new_size = (thumb_size, max(1, round(w * thumb_size / h)))
    imgs = F.interpolate(imgs, size=new_size, mode="bilinear", antialias=True, align_corners=False)
    return imgs.clamp(0, 1).permute(0, 2, 3, 1).numpy()


def cls_to_frames(attn):
    """CLS -> frame attention of one (T+1, T+1) matrix, renormalized over the frames."""
    w = attn[0, 1:]
    return w / (w.sum() + 1e-12)


# ----------------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------------
def plot_layer_attention(thumbs, attn, video_name, layer, num_layers, out_path, dpi, cols=8):
    """One figure per (video, layer):
      - top left:  full (T+1)x(T+1) attention matrix (rows = query, cols = key, index 0 = CLS)
      - top right: CLS -> frame attention as a bar chart
      - bottom:    the T frames at thumbnail resolution, titled with frame index and CLS weight
    """
    T = thumbs.shape[0]
    n_rows = math.ceil(T / cols)
    cls_w = cls_to_frames(attn)

    top_h, row_h = 6.5, 1.9
    fig = plt.figure(figsize=(15, top_h + row_h * n_rows))
    outer = fig.add_gridspec(2, 1, height_ratios=[top_h, row_h * n_rows], hspace=0.12,
                             left=0.02, right=0.98, top=0.95, bottom=0.01)
    top = outer[0].subgridspec(1, 2, width_ratios=[1, 1], wspace=0.25)
    grid = outer[1].subgridspec(n_rows, cols, wspace=0.04, hspace=0.35)

    # Full attention matrix
    ax_mat = fig.add_subplot(top[0, 0])
    im = ax_mat.imshow(attn, cmap="viridis", aspect="equal", interpolation="nearest")
    labels = ["CLS"] + [str(i) for i in range(T)]
    ax_mat.set_xticks(range(T + 1))
    ax_mat.set_yticks(range(T + 1))
    ax_mat.set_xticklabels(labels, rotation=90, fontsize=6)
    ax_mat.set_yticklabels(labels, fontsize=6)
    ax_mat.set_xlabel("Key (attended to)")
    ax_mat.set_ylabel("Query (attending)")
    ax_mat.set_title(f"Layer {layer + 1}/{num_layers} attention (head-averaged)", fontsize=11, fontweight="bold")
    fig.colorbar(im, ax=ax_mat, fraction=0.046, pad=0.02)

    # CLS -> frames
    ax_bar = fig.add_subplot(top[0, 1])
    ax_bar.bar(np.arange(T), cls_w, color="royalblue", edgecolor="black")
    ax_bar.axhline(1.0 / T, color="red", linestyle="--", label=f"Uniform (1/{T})")
    ax_bar.set_xlim(-0.5, T - 0.5)
    ax_bar.set_xlabel("Frame Index")
    ax_bar.set_ylabel("Attention Weight (renormalized over frames)")
    ax_bar.set_title(f"CLS -> frames (CLS -> CLS: {attn[0, 0]:.3f})", fontsize=11)
    ax_bar.legend()

    # Frames
    for i in range(T):
        ax = fig.add_subplot(grid[i // cols, i % cols])
        ax.imshow(thumbs[i])
        ax.set_title(f"Fr {i} | CLS: {cls_w[i]:.3f}", fontsize=8)
        ax.axis("off")

    fig.suptitle(video_name, fontsize=14, fontweight="bold", y=0.995)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------
# Summary statistics
# ----------------------------------------------------------------------------
def analyze_attention_distribution(attention_records, num_layers):
    """Entropy / max weight of the CLS -> frame attention, reported per layer."""
    print("\n--- Transformer Attention Overview (CLS -> frames) ---")
    if not attention_records:
        return

    for layer in range(num_layers):
        max_weights, entropies = [], []
        for rec in attention_records:
            w = cls_to_frames(rec["attn"][layer])
            max_weights.append(np.max(w))
            # Entropy: -sum(p * log(p)). Lower entropy = spikier attention.
            entropies.append(-np.sum(w * np.log(w + 1e-12)))

        T = len(cls_to_frames(attention_records[0]["attn"][layer]))
        uniform = 1.0 / T
        avg_max = np.mean(max_weights)
        ratio = avg_max / uniform

        print(f"[Layer {layer + 1}/{num_layers}]")
        print(f"  Mean Max Attention Weight per video: {avg_max:.4f} (Uniform would be ~{uniform:.4f})")
        print(f"  Highest Single-Frame Attention found: {np.max(max_weights):.4f}")
        print(f"  Mean Attention Entropy: {np.mean(entropies):.4f} (Max possible ~{math.log(T):.3f})")

        # Thresholds equal the old 0.15 / 0.06 at 32 frames (4.8x / 1.9x uniform).
        if ratio > 4.8:
            print("  Verdict: highly selective/spiky, focusing on specific frames.")
        elif ratio < 1.9:
            print("  Verdict: relatively uniform global pooling.")
        else:
            print("  Verdict: moderate selectivity with some focal points.")
    print("------------------------------------------------------\n")


# ----------------------------------------------------------------------------
# Inference
# ----------------------------------------------------------------------------
def run_attention_analysis(model, loader, device, run_name, base_out_dir, args):
    model.eval()
    attention_records = []
    dataset_samples = loader.dataset.data
    attn_dir = os.path.join(base_out_dir, f"attention_vis_{run_name}")
    os.makedirs(attn_dir, exist_ok=True)
    num_layers = model.num_blocks

    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc="Transformer Attention")):
            inputs = batch[0].squeeze(0).to(device)
            video_name = dataset_samples[i]['vid_name']
            stem = os.path.splitext(video_name)[0]

            _, attn_layers = model.forward_with_attention(inputs)
            # Each entry: (1, T+1, T+1), head-averaged. Index 0 = CLS, 1: = frames.
            attn_np = [a[0].cpu().numpy() for a in attn_layers]
            attention_records.append({"video": video_name, "attn": attn_np})

            thumbs = make_thumbnails(inputs.cpu(), args.thumb_size)
            for layer, attn in enumerate(attn_np):
                out_path = os.path.join(attn_dir, f"{stem}_layer{layer + 1}.png")
                plot_layer_attention(thumbs, attn, video_name, layer, num_layers, out_path, args.dpi)

            if args.save_raw:
                np.savez_compressed(os.path.join(attn_dir, f"{stem}_attn.npz"),
                                    **{f"layer{l + 1}": a for l, a in enumerate(attn_np)})

    analyze_attention_distribution(attention_records, num_layers)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    val_trans = make_val_transform_dino(args.img_size, args.complex_augs)

    if len(args.trans_paths) != len(args.splits):
        raise ValueError(f"Got {len(args.trans_paths)} checkpoint path(s) but {len(args.splits)} "
                         f"split(s) ({args.splits}).")

    for split, ckpt_path in zip(args.splits, args.trans_paths):
        run_name = f"split_{split}"
        print(f"\n>>> Analyzing Attention for Run: {run_name} ({ckpt_path})")

        val_ds = MILVideoDataset(
            json_path=os.path.join(args.annotations_path, run_name, f"mil_val_{args.dataset_name}.json"),
            num_frames=32,
            transform=val_trans,
            search_dir_paths=[args.videos_path],
        )
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

        num_blocks, num_classes = infer_model_config(ckpt_path)
        print(f"Detected {num_blocks} block(s), {num_classes} output class(es).")

        model = DinoSelfAttention(
            full_checkpoint_path=ckpt_path,
            num_blocks=num_blocks,
            num_classes=num_classes,
        ).to(device)

        run_attention_analysis(model, val_loader, device, run_name, args.output_dir, args)

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--trans_paths', nargs='+', required=True,
                        help="One Transformer checkpoint path per split in --splits")
    parser.add_argument('--splits', nargs='+', type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument('--dataset_name', type=str, default="2025")
    parser.add_argument('--annotations_path', type=str, default="data/stratified_splits")
    parser.add_argument('--videos_path', type=str, default="data/ensemble_results_paxos2025/cleaned_videos")
    parser.add_argument('--output_dir', type=str, default="transformer_attention_analysis")
    parser.add_argument('--img_size', type=int, default=512)
    parser.add_argument('--thumb_size', type=int, default=256, help="Height in pixels of each frame thumbnail.")
    parser.add_argument('--dpi', type=int, default=150, help="Output resolution. At 150 each frame tile is ~256 px wide.")
    parser.add_argument('--save_raw', action='store_true', help="Also save the raw attention matrices as .npz.")
    parser.add_argument('--complex_augs', action='store_true')

    args = parser.parse_args()
    main(args)