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
from dino.utils import apply_clahe_cv2_dino, make_train_transform_dino, make_val_transform_dino


class DinoClassifier(nn.Module):
    def __init__(self, repo_dir, weights, freeze_backbone=True, use_cls=True):
        super().__init__()

        self.backbone = torch.hub.load(
            repo_dir,
            "dinov3_vitl16",
            source="local",
            weights=weights
        )

        self.use_cls = use_cls
        self.embed_dim = self.backbone.embed_dim

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.head = nn.Linear(self.embed_dim, 1)

    def forward(self, x):
        feats = self.backbone.forward_features(x)

        if self.use_cls:
            x = feats["x_norm_clstoken"]
        else:
            x = feats["x_norm_patchtokens"].mean(dim=1)

        return self.head(x)


def train_classifier(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    timestamp = datetime.now().strftime('%m%d_%H%M')
    run_name = f"dino_{'complex' if args.complex_augs else 'simple'}_{timestamp}"

    run_dir = os.path.join("classifier", run_name)
    os.makedirs(run_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=os.path.join("runs", run_name))

    model = DinoClassifier(
        repo_dir=args.repo_dir,
        weights=args.weight_path,
        freeze_backbone=args.freeze_backbone,
        use_cls=args.use_cls
    ).to(device)

    train_trans = make_train_transform_dino(args.img_size, args.complex_augs)
    val_trans = make_val_transform_dino(args.img_size, args.complex_augs)

    train_dataset = FrameDataset("data/frame_train.json", "data/2024_Paxos_Frames/cropped_frames", transform=train_trans)
    val_dataset = FrameDataset("data/frame_val.json", "data/2024_Paxos_Frames/cropped_frames", transform=val_trans)

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

    parser.add_argument('--repo_dir', type=str, required=True)
    parser.add_argument('--weight_path', type=str, required=True)

    parser.add_argument('--freeze_backbone', action='store_true')
    parser.add_argument('--use_cls', action='store_true')

    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--lr_step', type=int, default=10)
    parser.add_argument('--img_size', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)

    parser.add_argument('--complex_augs', action='store_true')

    args = parser.parse_args()

    train_classifier(args)