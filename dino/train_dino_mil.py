import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os
import argparse

from data.dataset import MILVideoDataset, MILVideoDatasetNew
from .utils import make_train_transform_dino, make_val_transform_dino
from misc.utils import visualize_full_video_attention, save_confusion_matrix
from mil.utils import train_one_epoch_mil, validate_extended_mil


class DinoBackbone(nn.Module):
    def __init__(self, checkpoint_path=None, freeze=True):
        super().__init__()

        self.model = torch.hub.load(
            "dino/dinov3",
            "dinov3_vitl16",
            source="local",
            weights="dino/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
        )

        self.embed_dim = self.model.embed_dim

        # Load classifier checkpoint
        if checkpoint_path is not None:
            print(f"Loading classifier backbone from: {checkpoint_path}")
            state_dict = torch.load(checkpoint_path, map_location="cpu")

            backbone_state = {}
            for k, v in state_dict.items():
                if k.startswith("backbone."):
                    new_k = k.replace("backbone.", "")
                    backbone_state[new_k] = v

            self.model.load_state_dict(backbone_state, strict=False)
            print("Backbone weights loaded from classifier.")

        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False

    def forward(self, x):
        feats = self.model.forward_features(x)
        return feats["x_norm_clstoken"]


class DinoMIL(nn.Module):
    def __init__(
            self,
            checkpoint_path=None,
            freeze_backbone=True,
            D=512,
            K=1
    ):
        super().__init__()

        self.backbone = torch.hub.load(
            "dino/dinov3",
            "dinov3_vitl16",
            source="local",
            weights="dino/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
        )

        self.L = self.backbone.embed_dim  # feature dim

        self.attention_V = nn.Sequential(
            nn.Linear(self.L, D),
            nn.Tanh()
        )
        self.attention_U = nn.Sequential(
            nn.Linear(self.L, D),
            nn.Sigmoid()
        )
        self.attention_w = nn.Linear(D, K)

        self.classifier = nn.Sequential(
            nn.Linear(self.L, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        # 1. ALWAYS initialize the MIL head randomly first (Preserves your training setup)
        self._init_weights()

        # 2. Override with checkpoint weights if provided
        if checkpoint_path is not None:
            print(f"Loading checkpoint from: {checkpoint_path}")
            state_dict = torch.load(checkpoint_path, map_location="cpu")

            # Load backbone weights
            backbone_state = {k.replace("backbone.", ""): v for k, v in state_dict.items() if k.startswith("backbone.")}
            if backbone_state:
                self.backbone.load_state_dict(backbone_state, strict=False)
                print("Backbone weights loaded.")

            # Load MIL Head weights (Only does something during evaluation)
            head_state = {k: v for k, v in state_dict.items() if not k.startswith("backbone.")}
            if head_state:
                self.load_state_dict(head_state, strict=False)
                print("MIL Head weights loaded.")

        # Freeze backbone
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def _init_weights(self):
        for m in [self.attention_V, self.attention_U, self.attention_w, self.classifier]:
            if isinstance(m, nn.Sequential):
                for layer in m:
                    if isinstance(layer, nn.Linear):
                        nn.init.xavier_uniform_(layer.weight)
                        if layer.bias is not None:
                            nn.init.constant_(layer.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        print(f"--- DINO MIL [{self.L} features] weights initialized ---")

    # ---------------------------------------------------------
    # MODULAR EVALUATION METHODS (For 4-way script)
    # ---------------------------------------------------------
    def extract_features(self, x):
        """ Runs ONLY the DINO backbone to cache features. """
        feats = self.backbone.forward_features(x)
        return feats["x_norm_clstoken"]

    def forward_head(self, h):
        """ Runs ONLY the MIL Head on pre-computed features. """
        a_v = self.attention_V(h)
        a_u = self.attention_U(h)
        a = self.attention_w(a_v * a_u)  # [N, K]

        weights = torch.softmax(a, dim=0)  # over frames
        bag_representation = torch.sum(weights * h, dim=0)  # [L]
        logits = self.classifier(bag_representation.unsqueeze(0))  # [1,1]

        # Returns raw_a for the 'Max Attention' evaluation strategy
        return logits, weights, a.flatten()

    # ---------------------------------------------------------
    # STANDARD TRAINING FORWARD PASS (Unchanged)
    # ---------------------------------------------------------
    def forward(self, x):
        """
        x: [num_frames, 3, H, W]
        """
        h = self.extract_features(x)
        logits, weights, _ = self.forward_head(h)

        # We explicitly drop the raw_a here so your training script
        # continues to receive exactly 2 items, preventing crashes.
        return logits, weights


def train_mil(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    timestamp = datetime.now().strftime('%m%d_%H%M')
    run_name = f"dino_mil_{'complex' if args.complex_augs else 'simple'}_{args.split_path.split('/')[-1]}_{'segmented' if args.random_segment_sample else 'uniform'}_{timestamp}"

    run_dir = os.path.join("mil", "models", run_name)
    os.makedirs(run_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=os.path.join("runs", run_name))
    writer.add_text("args", str(args))

    # --- Model ---
    model = DinoMIL(checkpoint_path=args.classifier_checkpoint, freeze_backbone=args.freeze_backbone).to(device)

    # --- Transforms ---
    train_trans = make_train_transform_dino(args.img_size, args.complex_augs)
    val_trans = make_val_transform_dino(args.img_size, args.complex_augs)

    # --- Datasets ---
    train_ds = MILVideoDatasetNew(os.path.join(args.split_path, "mil_train.json"),
                                  num_frames=32,
                                  transform=train_trans,
                                  random_segment_sample=args.random_segment_sample)
    val_ds = MILVideoDatasetNew(os.path.join(args.split_path, "mil_val.json"),
                                num_frames=32,
                                transform=val_trans)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=1, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=1, shuffle=False)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=0.1)

    best_pr_auc = 0.0

    print(f"Starting MIL Training: {run_name}")

    for epoch in range(args.epochs):
        t_loss = train_one_epoch_mil(model, train_loader, criterion, optimizer, device)
        m = validate_extended_mil(model, val_loader, criterion, device)

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # Save confusion matrix
        save_confusion_matrix(m['y_true'], m['y_pred'], epoch, run_dir)

        # # --- Visualization ---
        # try:
        #     # We pass run_dir so you can modify this function to save inside the run folder
        #     visualize_full_video_attention(
        #         model,
        #         "data/own_clips_hd/val_videos/cleaned_videos/CLEAN_2024_02_11_12_26_IMG_4608 LE MILD NPDR.mp4",
        #         val_trans,
        #         device,
        #         epoch,
        #         output_dir=run_dir
        #     )
        # except Exception as e:
        #     print(f"Visualization skipped: {e}")

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

    train_mil(args)