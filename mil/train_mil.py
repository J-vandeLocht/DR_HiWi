import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from datetime import datetime
import os
import argparse

from .model import GatedAttentionMIL
from .utils import train_one_epoch_mil, load_backbone_weights, validate_extended_mil
from misc.utils import save_confusion_matrix, apply_clahe_cv2, visualize_full_video_attention
from data.dataset import MILVideoDataset, MILVideoDatasetNew


def train_mil(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    timestamp = datetime.now().strftime('%m%d_%H%M')
    run_name = f"MIL_{args.model}_lr{args.lr}_bs{args.batch_size}_sz{args.img_size}_{timestamp}"

    # Folder setup
    run_dir = os.path.join("mil_runs", run_name)
    os.makedirs(run_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join("runs", run_name))

    # --- Transforms ---
    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train_trans = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.Lambda(apply_clahe_cv2),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(180),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(), norm
    ])
    val_trans = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.Lambda(apply_clahe_cv2),
        transforms.ToTensor(), norm
    ])

    # --- Datasets ---
    if not args.use_new_clips:
        train_ds = MILVideoDataset("data/mil_train.json", "data/clips_hd", num_frames=32, transform=train_trans)
        val_ds = MILVideoDataset("data/mil_val.json", "data/clips_hd", num_frames=32, transform=val_trans)
    else:
        train_ds = MILVideoDatasetNew("data/mil_train.json", num_frames=32, transform=train_trans)
        val_ds = MILVideoDatasetNew("data/mil_val.json", num_frames=32, transform=val_trans)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=1, shuffle=False)

    # --- Model Setup ---
    # NOTE: Ensure GatedAttentionMIL accepts `backbone` argument to swap EfficientNets dynamically!
    model = GatedAttentionMIL(backbone=args.model).to(device)

    if args.frame_model_path:
        load_backbone_weights(model, args.frame_model_path)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # StepLR Scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=0.1)

    print(f"🚀 Starting MIL Training: {run_name}")
    print(f"Videos: {len(train_ds)} train, {len(val_ds)} val")

    best_pr_auc = 0.0

    for epoch in range(args.epochs):
        # Train & Validate
        t_loss = train_one_epoch_mil(model, train_loader, criterion, optimizer, device)
        m = validate_extended_mil(model, val_loader, criterion, device)

        # Scheduler step and log
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Meta/Learning_Rate', current_lr, epoch)

        # Saves
        save_confusion_matrix(m['y_true'], m['y_pred'], epoch, run_dir)

        # --- Visualization ---
        try:
            # We pass run_dir so you can modify this function to save inside the run folder
            visualize_full_video_attention(
                model,
                "data/own_clips_hd/val_videos/cleaned_videos/CLEAN_2024_02_11_12_26_IMG_4608 LE MILD NPDR.mp4",
                val_trans,
                device,
                epoch,
                output_dir=run_dir
            )
        except Exception as e:
            print(f"Visualization skipped: {e}")

        # Save Best Model based on PR-AUC
        if m['pr_auc'] > best_pr_auc:
            best_pr_auc = m['pr_auc']
            torch.save(model.state_dict(), os.path.join(run_dir, "best_mil_model.pth"))

        # TensorBoard Logging
        writer.add_scalar('Loss/train', t_loss, epoch)
        writer.add_scalar('Loss/val', m['loss'], epoch)
        writer.add_scalar('Metric/Accuracy', m['acc'], epoch)
        writer.add_scalar('Metric/F1', m['f1'], epoch)
        writer.add_scalar('Metric/Precision', m['precision'], epoch)
        writer.add_scalar('Metric/Recall', m['recall'], epoch)
        writer.add_scalar('Metric/PR_AUC', m['pr_auc'], epoch)
        writer.add_scalar('Metric/ROC_AUC', m['roc_auc'], epoch)

        print(
            f"Epoch {epoch} | LR: {current_lr:.6f} | Loss: {t_loss:.3f} | PR-AUC: {m['pr_auc']:.3f} | F1: {m['f1']:.3f}")

    writer.close()
    print(f"Done. Run directory: {run_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grid Search for Paxos MIL Model")
    parser.add_argument('--model', type=str, default='b0', help='EfficientNet backbone (b0-b7)')
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--img_size', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=1, help='Usually 1 for MIL depending on memory')
    parser.add_argument('--lr', type=float, default=1e-6)
    parser.add_argument('--lr_step', type=int, default=10, help='Epoch to drop LR by 10x')
    parser.add_argument('--frame_model_path', type=str, default=None, help='Path to pre-trained frame weights')
    parser.add_argument('--use_old_clips', dest='use_new_clips', action='store_false', help='Flag to use old dataset')
    parser.set_defaults(use_new_clips=True)

    args = parser.parse_args()
    train_mil(args)