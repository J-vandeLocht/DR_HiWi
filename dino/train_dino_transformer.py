import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os
import argparse
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler

from data.dataset import MILVideoDataset
from .utils import make_train_transform_dino, make_val_transform_dino
from misc.utils import save_confusion_matrix
from transformer.utils import train_one_epoch_trans, validate_extended_trans


class TransformerBlock(nn.Module):
    """A single MHA + FFN block (post-norm, matching the original design).
    Returns output and the raw attention weights so the caller can inspect
    CLS->frame attention from the last block."""

    def __init__(self, dim, num_heads=8, ffn_dim=2048, dropout=0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_out, attn_weights = self.mha(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))

        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))

        return x, attn_weights


class DinoSelfAttention(nn.Module):
    def __init__(
            self,
            freeze_backbone=True,
            num_heads=8,
            ffn_dim=2048,
            dropout=0.1,
            num_blocks=1,
            use_pos_embedding=False,
            num_frames=32,
            classifier_checkpoint_path=None,
            full_checkpoint_path=None,
    ):
        """
        Two independent, mutually exclusive, strict checkpoint-loading paths:

        - classifier_checkpoint_path: path to a *DinoClassifier* state dict.
          Only the backbone weights are extracted and loaded (strict=True
          against the backbone's own key set). Use this when starting
          Self-Attention training from a fine-tuned DinoClassifier backbone.

        - full_checkpoint_path: path to a *DinoSelfAttention* state dict
          (i.e. one saved from this exact class, with matching
          num_blocks/use_pos_embedding/num_frames config). Loaded strictly
          into the whole model. Use this for inference / resuming after
          Self-Attention training has finished.

        Passing both is an error - decide which stage you're in.
        """
        super().__init__()

        if classifier_checkpoint_path is not None and full_checkpoint_path is not None:
            raise ValueError(
                "Specify either classifier_checkpoint_path or full_checkpoint_path, not both."
            )

        self.num_blocks = num_blocks
        self.use_pos_embedding = use_pos_embedding
        self.num_frames = num_frames

        self.backbone = torch.hub.load(
            "dino/dinov3",
            "dinov3_vitl16",
            source="local",
            weights="dino/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
        )

        self.L = self.backbone.embed_dim  # feature dim (e.g., 1024 for ViT-L)

        # Learnable CLS token that will aggregate information across all frames
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.L))

        # Optional learned positional embedding, one per frame slot.
        # NOT applied to the CLS token, only to the frame tokens, since the
        # frames form the "sequence" with potential ordering/position info
        # and the CLS token is just an aggregation query.
        if self.use_pos_embedding:
            self.pos_embedding = nn.Parameter(torch.zeros(1, num_frames, self.L))
        else:
            self.pos_embedding = None

        # Stack of Self-Attention Blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(self.L, num_heads=num_heads, ffn_dim=ffn_dim, dropout=dropout)
            for _ in range(num_blocks)
        ])

        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(self.L, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        self._init_weights()

        # --- Checkpoint loading (mutually exclusive, both strict) ---
        if classifier_checkpoint_path is not None:
            self._load_classifier_backbone(classifier_checkpoint_path)
        elif full_checkpoint_path is not None:
            self._load_full_checkpoint(full_checkpoint_path)

        # Freeze backbone (applied after loading, so it applies regardless
        # of which checkpoint path was used, or none at all)
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def _load_classifier_backbone(self, path):
        """Load only the backbone weights from a DinoClassifier checkpoint.
        Strict: the checkpoint's backbone.* keys must exactly match this
        model's backbone's own key set, or this raises."""
        print(f"Loading backbone from DinoClassifier checkpoint: {path}")
        state_dict = torch.load(path, map_location="cpu")

        backbone_state = {
            k[len("backbone."):]: v for k, v in state_dict.items() if k.startswith("backbone.")
        }
        if not backbone_state:
            raise ValueError(
                f"No keys starting with 'backbone.' found in {path}. "
                f"Is this really a DinoClassifier checkpoint?"
            )

        self.backbone.load_state_dict(backbone_state, strict=True)
        print(f"Backbone weights loaded strictly ({len(backbone_state)} tensors).")

    def _load_full_checkpoint(self, path):
        """Load a full DinoSelfAttention checkpoint (backbone + cls_token +
        blocks + classifier). Strict: requires an exact key/shape match,
        i.e. the model must have been constructed with the same
        num_blocks/use_pos_embedding/num_frames as when the checkpoint was
        saved."""
        print(f"Loading full DinoSelfAttention checkpoint: {path}")
        state_dict = torch.load(path, map_location="cpu")

        self.load_state_dict(state_dict, strict=True)
        print(f"Full model weights loaded strictly ({len(state_dict)} tensors).")

    def _init_weights(self):
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

        # Initialize each block's FFN + MHA
        for block in self.blocks:
            for layer in block.ffn:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)

            nn.init.xavier_uniform_(block.mha.in_proj_weight)
            if block.mha.in_proj_bias is not None:
                nn.init.constant_(block.mha.in_proj_bias, 0)
            nn.init.xavier_uniform_(block.mha.out_proj.weight)
            if block.mha.out_proj.bias is not None:
                nn.init.constant_(block.mha.out_proj.bias, 0)

        # Initialize CLS Token
        nn.init.normal_(self.cls_token, std=1e-6)

        # Initialize positional embedding, if used
        if self.pos_embedding is not None:
            nn.init.trunc_normal_(self.pos_embedding, std=0.02)

        print(
            f"--- DINO Self-Attention [{self.L} features, {self.num_blocks} block(s), "
            f"pos_embedding={self.use_pos_embedding}] weights initialized ---"
        )

    # ---------------------------------------------------------
    # MODULAR EVALUATION METHODS
    # ---------------------------------------------------------
    def extract_features(self, x):
        # Runs ONLY the DINO backbone to cache features.
        feats = self.backbone.forward_features(x)
        return feats["x_norm_clstoken"]

    def forward_head(self, h):
        # h shape: [num_frames, L] -> e.g., [32, 1024]
        # Expand to pseudo-batch format [batch_size=1, num_frames, L]
        h = h.unsqueeze(0)

        if self.pos_embedding is not None:
            n = h.size(1)
            if n != self.num_frames:
                raise ValueError(
                    f"Got {n} frames but positional embedding was initialized "
                    f"for num_frames={self.num_frames}."
                )
            h = h + self.pos_embedding

        # Expand CLS token to match batch size
        cls_tokens = self.cls_token.expand(h.size(0), -1, -1)  # [1, 1, L]

        # Prepend CLS token to the frame features sequence
        x = torch.cat((cls_tokens, h), dim=1)  # [1, num_frames + 1, L]

        # Run through the stack of self-attention blocks.
        # Keep the attention weights from the LAST block only, since that's
        # the one whose CLS-token attention best reflects what actually
        # drove the final classification.
        attn_weights = None
        for block in self.blocks:
            x, attn_weights = block(x)

        # Extract the processed CLS token embedding for classification
        bag_representation = x[:, 0, :]  # [1, L]
        logits = self.classifier(bag_representation)  # [1, 1]

        # Extract attention weights originating from the CLS token to all target video frames
        # attn_weights shape: [batch_size, queries, keys] -> [1, num_frames+1, num_frames+1]
        # Query 0 is the CLS token. Keys 1: are the video frames.
        frame_weights = attn_weights[0, 0, 1:]  # [num_frames]

        # Normalize weights over the frames to sum to 1 (preserves expected MIL downstream properties)
        frame_weights = frame_weights / (frame_weights.sum() + 1e-12)

        # Returns logits, frame weights, and raw attention vector for max-attention strategy compatibility
        return logits, frame_weights, frame_weights

    def forward(self, x):
        """
        x: [num_frames, 3, H, W]
        """
        h = self.extract_features(x)
        logits, weights, _ = self.forward_head(h)

        return logits, weights


def train_self_attention(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    timestamp = datetime.now().strftime('%m%d_%H%M')
    run_name = (f"dino_sa"
                f"{args.split_path.split('/')[-1]}_"
                f"{timestamp}_"
                f"{"full" if args.train_json == "mil_train.json" else args.train_json.split(".")[0].split("_")[-1]}")

    run_dir = os.path.join("transformer", "models", run_name)
    os.makedirs(run_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=os.path.join("runs", run_name))
    writer.add_text("args", str(args))

    # --- Model ---
    model = DinoSelfAttention(classifier_checkpoint_path=args.classifier_checkpoint,
                              freeze_backbone=args.freeze_backbone,
                              num_blocks=args.num_blocks,
                              use_pos_embedding=args.use_pos_embedding).to(device)

    # --- Transforms ---
    train_trans = make_train_transform_dino(args.img_size, args.complex_augs)
    val_trans = make_val_transform_dino(args.img_size, args.complex_augs)

    # --- Datasets ---
    if args.fused_dataset:
        search_dir_paths = ["data/ensemble_results_paxos2025/cleaned_videos", "data/ensemble_results_paxos2020/cleaned_videos"]
    else:
        search_dir_paths = ["data/ensemble_results_paxos2025/cleaned_videos"]

    # --- Datasets ---
    train_ds = MILVideoDataset(os.path.join(args.split_path, args.train_json),
                               num_frames=32,
                               transform=train_trans,
                               random_segment_sample=args.random_segment_sample,
                               search_dir_paths=search_dir_paths)
    val_ds = MILVideoDataset(os.path.join(args.split_path, args.val_json),
                             num_frames=32,
                             transform=val_trans,
                             search_dir_paths=search_dir_paths)

    # --- Setup Oversampling ---
    train_labels = np.array(train_ds.labels)
    class_counts = np.bincount(train_labels)

    # Compute inverse class frequencies
    class_weights = 1.0 / class_counts

    # Map each video in the dataset to its class weight
    sample_weights = class_weights[train_labels]
    sample_weights = torch.from_numpy(sample_weights).double()

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    # Use sampler in DataLoader (shuffle MUST be set to False)
    train_loader = DataLoader(
        train_ds,
        batch_size=1,
        sampler=sampler,
        shuffle=False
    )
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=0.1)

    best_pr_auc = 0.0

    print(f"Starting Self-Attention Training: {run_name}")

    for epoch in range(args.epochs):
        t_loss = train_one_epoch_trans(model, train_loader, criterion, optimizer, device)
        m = validate_extended_trans(model, val_loader, criterion, device)

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # Save confusion matrix
        save_confusion_matrix(m['y_true'], m['y_pred'], epoch, run_dir)

        # Save best model
        if m['pr_auc'] > best_pr_auc:
            best_pr_auc = m['pr_auc']
            torch.save(model.state_dict(), os.path.join(run_dir, "best_sa_model.pth"))

        writer.add_scalar('Meta/Learning_Rate', current_lr, epoch)

        writer.add_scalar('Loss/train', t_loss, epoch)
        writer.add_scalar('Loss/val', m['loss'], epoch)

        writer.add_scalar('Metric/Accuracy', m['acc'], epoch)
        writer.add_scalar('Metric/F1', m['f1'], epoch)
        writer.add_scalar('Metric/Precision', m['precision'], epoch)
        writer.add_scalar('Metric/Recall', m['recall'], epoch)
        writer.add_scalar('Metric/PR_AUC', m['pr_auc'], epoch)
        writer.add_scalar('Metric/ROC_AUC', m['roc_auc'], epoch)

        print(
            f"Epoch {epoch} | "
            f"LR: {current_lr:.6f} | "
            f"Loss: {t_loss:.3f} | "
            f"Acc: {m['acc']:.3f} | "
            f"F1: {m['f1']:.3f} | "
            f"PR-AUC: {m['pr_auc']:.3f}"
        )

    writer.close()
    print(f"Done. Run directory: {run_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--classifier_checkpoint', type=str, default=None,
                        help='Path to trained classifier model (.pth)')
    parser.add_argument('--split_path', type=str, required=True)
    parser.add_argument('--train_json', type=str, default="mil_train.json")
    parser.add_argument('--val_json', type=str, default="mil_val.json")
    parser.add_argument('--weight_path', type=str, default=None,
                        help='Path to pretrained DINO classifier weights')
    parser.add_argument('--freeze_backbone', action='store_true')
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--lr_step', type=int, default=10)
    parser.add_argument('--img_size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--complex_augs', action='store_true')
    parser.add_argument('--use_pos_embedding', action='store_true')
    parser.add_argument('--num_blocks', type=int, default=1)
    parser.add_argument('--random_segment_sample', action='store_true')
    parser.add_argument('--fused_dataset', action='store_true')

    args = parser.parse_args()

    train_self_attention(args)