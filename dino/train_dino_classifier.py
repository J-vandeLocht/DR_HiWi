import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os
import argparse

from data.dataset import FrameDataset
from misc.utils import save_confusion_matrix
from classifier.utils import validate_classifier, train_one_epoch_classifier
from dino.utils import make_train_transform_dino, make_val_transform_dino


class DinoClassifier(nn.Module):
    def __init__(self, freeze_backbone=True, num_classes=5):
        super().__init__()

        self.backbone = torch.hub.load(
            "dino/dinov3",
            "dinov3_vitl16",
            source="local",
            weights="dino/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
        )

        self.embed_dim = self.backbone.embed_dim

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.head = nn.Sequential(
            nn.Linear(self.embed_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

        self._init_weights()

    def _init_weights(self):
        for layer in self.head:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        feats = self.backbone.forward_features(x)
        # Perform classification based on the class token
        x = feats["x_norm_clstoken"]
        return self.head(x)


def train_classifier(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    timestamp = datetime.now().strftime('%m%d_%H%M')
    run_name = f"dino_{'complex' if args.complex_augs else 'simple'}_{args.split_path.split('/')[-1]}_{'kaggle' if args.weight_path else 'imagenet'}_{timestamp}"

    run_dir = os.path.join("classifier", "models", run_name)
    os.makedirs(run_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=os.path.join("runs", run_name))
    writer.add_text("args", str(args))

    # Step 1: Initialize model with ORIGINAL training setup (5 classes)
    model = DinoClassifier(
        freeze_backbone=args.freeze_backbone,
        num_classes=5
    ).to(device)

    # Step 2: Load pretrained weights if provided
    if args.weight_path:
        print(f"Loading pretrained weights from: {args.weight_path}")
        state_dict = torch.load(args.weight_path, map_location=device)

        model.load_state_dict(state_dict)
        print("Successfully loaded pretrained DINO weights.")

    # Step 3: Replace head for NEW task (binary = 1 output)
    model.head = nn.Sequential(
            nn.Linear(model.embed_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        ).to(device)
    model._init_weights()

    train_trans = make_train_transform_dino(args.img_size, args.complex_augs)
    val_trans = make_val_transform_dino(args.img_size, args.complex_augs)

    train_dataset = FrameDataset(os.path.join(args.split_path, "frame_train.json"),
                                 "data/2024_Paxos_Frames/cropped_frames",
                                 transform=train_trans)
    val_dataset = FrameDataset(os.path.join(args.split_path, "frame_val.json"),
                               "data/2024_Paxos_Frames/cropped_frames",
                               transform=val_trans)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=0.1)

    best_pr_auc = 0.0

    print(f"Start Training: {run_name}")

    for epoch in range(args.epochs):
        t_loss = train_one_epoch_classifier(model, train_loader, criterion, optimizer, device)
        m = validate_classifier(model, val_loader, criterion, device)

        scheduler.step()

        save_confusion_matrix(m['y_true'], m['y_pred'], epoch, run_dir)

        if m['pr_auc'] > best_pr_auc:
            best_pr_auc = m['pr_auc']
            torch.save(model.state_dict(), os.path.join(run_dir, "best_model.pth"))

        writer.add_scalar('Meta/Learning_Rate', optimizer.param_groups[0]['lr'], epoch)

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
            f"Loss: {t_loss:.3f} | "
            f"Acc: {m['acc']:.3f} | "
            f"F1: {m['f1']:.3f} | "
            f"PR-AUC: {m['pr_auc']:.3f}"
        )

    writer.close()
    print(f"Done. Run directory: {run_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--split_path', type=str, required=True)
    parser.add_argument('--weight_path', type=str, default=None,
                        help='Path to pretrained DINO classifier weights')
    parser.add_argument('--freeze_backbone', action='store_true')
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--lr_step', type=int, default=10)
    parser.add_argument('--img_size', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--complex_augs', action='store_true')

    args = parser.parse_args()

    train_classifier(args)
