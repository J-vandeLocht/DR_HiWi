import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os
import argparse

from data.dataset import MILVideoDataset
from .utils import make_train_transform_dino, make_val_transform_dino
from misc.utils import save_confusion_matrix
from transformer.utils import train_one_epoch_trans, validate_extended_trans


class DinoSelfAttention(nn.Module):
    def __init__(
            self,
            checkpoint_path=None,
            freeze_backbone=True,
            num_heads=8,
            ffn_dim=2048,
            dropout=0.1
    ):
        super().__init__()

        self.backbone = torch.hub.load(
            "dino/dinov3",
            "dinov3_vitl16",
            source="local",
            weights="dino/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
        )

        self.L = self.backbone.embed_dim  # feature dim (e.g., 1024 for ViT-L)

        # Learnable CLS token that will aggregate information across all frames
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.L))

        # Self-Attention Layer Components
        self.mha = nn.MultiheadAttention(embed_dim=self.L, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(self.L)
        self.norm2 = nn.LayerNorm(self.L)

        self.ffn = nn.Sequential(
            nn.Linear(self.L, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, self.L)
        )
        self.dropout = nn.Dropout(dropout)

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

        # Override with checkpoint weights if provided
        if checkpoint_path is not None:
            print(f"Loading checkpoint from: {checkpoint_path}")
            state_dict = torch.load(checkpoint_path, map_location="cpu")

            # Load backbone weights
            backbone_state = {k.replace("backbone.", ""): v for k, v in state_dict.items() if k.startswith("backbone.")}
            if backbone_state:
                self.backbone.load_state_dict(backbone_state, strict=False)
                print("Backbone weights loaded.")

            # Load Attention Head weights
            head_state = {k: v for k, v in state_dict.items() if not k.startswith("backbone.")}
            if head_state:
                self.load_state_dict(head_state, strict=False)
                print("Self-Attention Head weights loaded.")

        # Freeze backbone
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def _init_weights(self):
        # Initialize FFN and Classifier
        for m in [self.ffn, self.classifier]:
            for layer in m:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)

        # Initialize Multihead Attention
        nn.init.xavier_uniform_(self.mha.in_proj_weight)
        if self.mha.in_proj_bias is not None:
            nn.init.constant_(self.mha.in_proj_bias, 0)
        nn.init.xavier_uniform_(self.mha.out_proj.weight)
        if self.mha.out_proj.bias is not None:
            nn.init.constant_(self.mha.out_proj.bias, 0)

        # Initialize CLS Token
        nn.init.normal_(self.cls_token, std=1e-6)

        print(f"--- DINO Self-Attention [{self.L} features] weights initialized ---")

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

        # Expand CLS token to match batch size
        cls_tokens = self.cls_token.expand(h.size(0), -1, -1)  # [1, 1, L]

        # Prepend CLS token to the frame features sequence
        x = torch.cat((cls_tokens, h), dim=1)  # [1, num_frames + 1, L]

        # 1. Multi-Head Self-Attention Block
        attn_out, attn_weights = self.mha(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))

        # 2. Feed-Forward Network Block
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))

        # Extract the processed CLS token embedding for classification
        bag_representation = x[:, 0, :]  # [1, L]
        logits = self.classifier(bag_representation)  # [1, 1]

        # Extract attention weights originating from the CLS token to all target video frames
        # attn_weights shape: [batch_size, queries, keys] -> [1, 33, 33]
        # Query 0 is the CLS token. Keys 1 to 33 are the video frames.
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
    run_name = f"dino_sa_{'complex' if args.complex_augs else 'simple'}_{args.split_path.split('/')[-1]}_{'segmented' if args.random_segment_sample else 'uniform'}_{timestamp}"

    run_dir = os.path.join("transformer", "models", run_name)
    os.makedirs(run_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=os.path.join("runs", run_name))
    writer.add_text("args", str(args))

    # --- Model ---
    model = DinoSelfAttention(checkpoint_path=args.classifier_checkpoint, freeze_backbone=args.freeze_backbone).to(device)

    # --- Transforms ---
    train_trans = make_train_transform_dino(args.img_size, args.complex_augs)
    val_trans = make_val_transform_dino(args.img_size, args.complex_augs)

    # --- Datasets ---
    train_ds = MILVideoDataset(os.path.join(args.split_path, "mil_train.json"),
                               num_frames=32,
                               transform=train_trans,
                               random_segment_sample=args.random_segment_sample)
    val_ds = MILVideoDataset(os.path.join(args.split_path, "mil_val.json"),
                             num_frames=32,
                             transform=val_trans)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=1, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=1, shuffle=False)

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
            torch.save(model.state_dict(), os.path.join(run_dir, "best_mil_model.pth"))

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
    parser.add_argument('--weight_path', type=str, default=None,
                        help='Path to pretrained DINO classifier weights')
    parser.add_argument('--freeze_backbone', action='store_true')
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--lr_step', type=int, default=10)
    parser.add_argument('--img_size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--complex_augs', action='store_true')
    parser.add_argument('--random_segment_sample', action='store_true')

    args = parser.parse_args()

    train_self_attention(args)