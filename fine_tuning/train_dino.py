import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm
from datetime import datetime
import cv2
import os
import argparse
import numpy as np
from PIL import Image

from data.dataset import KaggleDRDataset
from .eval_eff import evaluate_and_log
from misc.utils import apply_clahe_cv2
from .train_eff import train_one_epoch
from dino.train_dino_classifier import DinoClassifier


def train_classifier(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Setup Directory Structure
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    run_name = f"{timestamp}_DINOv3_ep{args.epochs}_size{args.img_size}_{'frozen' if args.freeze_backbone else 'unfrozen'}"
    run_dir = os.path.join("runs", run_name)
    os.makedirs(run_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=run_dir)
    writer.add_text("args", str(args))

    # 2. Model Initialization
    model = DinoClassifier(
        num_classes=args.num_classes,
        freeze_backbone=args.freeze_backbone
    ).to(device)

    # 3. Transforms
    # Uses the same structure from your EfficientNet script
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(size=args.img_size, scale=(0.7, 1.0)),
        transforms.Lambda(apply_clahe_cv2),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(180),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.Lambda(apply_clahe_cv2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 4. Data & Sampler Logic
    train_dataset = KaggleDRDataset(args.train_path, transform=train_transforms)
    val_dataset = KaggleDRDataset(args.val_path, transform=val_transforms)

    # Calculate class weights for WeightedRandomSampler (handling DR imbalance)
    labels = np.array(train_dataset.labels)
    class_sample_count = np.array([len(np.where(labels == t)[0]) for t in range(args.num_classes)])
    class_weights = 1. / class_sample_count
    sample_weights = torch.from_numpy(np.array([class_weights[t] for t in labels])).double()
    sampler = torch.utils.data.WeightedRandomSampler(weights=sample_weights, num_samples=args.num_samples,
                                                     replacement=True)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # 5. Optimizer, Scheduler, and Loss
    # Filter explicitly so we don't pass frozen weights to the optimizer
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    best_qwk = -1.0
    print(f"Starting training: {run_name}")

    for epoch in range(args.epochs):
        t_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Comprehensive Evaluation via your external function
        metrics = evaluate_and_log(model, val_loader, device, epoch, run_dir)

        # Step the learning rate scheduler
        scheduler.step()

        # Logging to TensorBoard
        writer.add_scalar('Meta/Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalar('Loss/train', t_loss, epoch)
        writer.add_scalar('Loss/val', metrics['loss'], epoch)
        writer.add_scalar('Accuracy/val', metrics['acc'], epoch)
        writer.add_scalar('Kappa/val', metrics['qwk'], epoch)
        writer.add_scalar('F1/Macro', metrics['f1_macro'], epoch)
        writer.add_scalar('PR_AUC/Mean', metrics['pr_auc_mean'], epoch)

        # Save weights based on QWK
        if metrics['qwk'] > best_qwk:
            best_qwk = metrics['qwk']
            save_path = os.path.join(run_dir, f"best_weights.pth")
            torch.save(model.state_dict(), save_path)
            print(f"--> [Epoch {epoch}] New Best QWK: {best_qwk:.4f}")

        print(f"Epoch {epoch}: T_Loss: {t_loss:.3f} | V_Loss: {metrics['loss']:.3f} | QWK: {metrics['qwk']:.3f}")

    writer.close()
    print(f"Done. Run directory: {run_dir}")
    return run_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DINOv3 on Kaggle DR Dataset")

    # Dataset paths (defaulting to the ones used in your EfficientNet script)
    parser.add_argument('--train_path', type=str, default='data/fused_dr_dataset/dr_unified_v2/dr_unified_v2/train')
    parser.add_argument('--val_path', type=str, default='data/fused_dr_dataset/dr_unified_v2/dr_unified_v2/val')
    parser.add_argument('--num_classes', type=int, default=5, help="Number of classes in the dataset")
    parser.add_argument('--num_samples', type=int, default=10000, help="Number of samples per epoch for the sampler")

    # Training parameters
    parser.add_argument('--freeze_backbone', action='store_true',
                        help="Freeze the DINOv3 backbone and only train the linear head")
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--lr_step', type=int, default=10, help="Epochs before learning rate decay")
    parser.add_argument('--img_size', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=16)  # Lowered default from 32 to 16, as ViT-L is heavy
    parser.add_argument('--lr', type=float, default=1e-4)

    args = parser.parse_args()

    train_classifier(args)