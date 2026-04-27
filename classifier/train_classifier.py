import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, models
from datetime import datetime
import os
import argparse

from data.dataset import FrameDataset
from .utils import validate_classifier, train_one_epoch_classifier
from misc.utils import save_gradcam_plusplus_grid, apply_clahe_cv2, save_confusion_matrix


def train_classifier(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    timestamp = datetime.now().strftime('%m%d_%H%M')
    run_name = f"{args.model}_lr{args.lr}_bs{args.batch_size}_sz{args.img_size}_{timestamp}"

    # Folder setup
    run_dir = os.path.join("classifier", run_name)
    os.makedirs(run_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join("runs", run_name))

    # Corrected Model Map
    model_map = {
        'b0': (models.efficientnet_b0, models.EfficientNet_B0_Weights.DEFAULT),
        'b1': (models.efficientnet_b1, models.EfficientNet_B1_Weights.DEFAULT),
        'b2': (models.efficientnet_b2, models.EfficientNet_B2_Weights.DEFAULT),
        'b3': (models.efficientnet_b3, models.EfficientNet_B3_Weights.DEFAULT),
        'b4': (models.efficientnet_b4, models.EfficientNet_B4_Weights.DEFAULT),
        'b5': (models.efficientnet_b5, models.EfficientNet_B5_Weights.DEFAULT),
        'b6': (models.efficientnet_b6, models.EfficientNet_B6_Weights.DEFAULT),
        'b7': (models.efficientnet_b7, models.EfficientNet_B7_Weights.DEFAULT),
    }

    model_fn, weights = model_map[args.model]
    model = model_fn(weights=weights)

    if hasattr(args, 'weight_path') and args.weight_path:
        print(f"Loading custom weights from: {args.weight_path}")

        # Load the state dict
        state_dict = torch.load(args.weight_path, map_location=device)

        # Check if the saved model was a 5-class model (from your first script)
        # We temporarily change the head to 5 to match the file's shape
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, 5)

        # Load the weights
        model.load_state_dict(state_dict)
        print("Successfully loaded pre-trained EfficientNet-B4 weights.")

    # 3. Finalize the Architecture for the NEW task (Binary/1 node)
    # This overwrites the 5-node layer (or the default layer) with a 1-node layer
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
    model.to(device)

    # Transforms
    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train_trans = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.Lambda(apply_clahe_cv2),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(180),
        transforms.ToTensor(), norm
    ])
    val_trans = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.Lambda(apply_clahe_cv2),
        transforms.ToTensor(), norm
    ])

    train_dataset = FrameDataset("data/frame_train.json", "data/2024_Paxos_Frames/cropped_frames",
                                 transform=train_trans)
    val_dataset = FrameDataset("data/frame_val.json", "data/2024_Paxos_Frames/cropped_frames", transform=val_trans)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=0.1)

    print(f"Start Training: {run_name}")
    best_pr_auc = 0.0

    for epoch in range(args.epochs):
        t_loss = train_one_epoch_classifier(model, train_loader, criterion, optimizer, device)
        m = validate_classifier(model, val_loader, criterion, device)
        scheduler.step()

        # Saves
        save_confusion_matrix(m['y_true'], m['y_pred'], epoch, run_dir)
        save_gradcam_plusplus_grid(model, val_loader, device, epoch, run_dir)

        if m['pr_auc'] > best_pr_auc:
            best_pr_auc = m['pr_auc']
            torch.save(model.state_dict(), os.path.join(run_dir, "best_model.pth"))

        # Tensorboard Logging
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Meta/Learning_Rate', current_lr, epoch)
        writer.add_scalar('Loss/train', t_loss, epoch)
        writer.add_scalar('Loss/val', m['loss'], epoch)
        writer.add_scalar('Metric/Accuracy', m['acc'], epoch)
        writer.add_scalar('Metric/F1', m['f1'], epoch)
        writer.add_scalar('Metric/Precision', m['precision'], epoch)
        writer.add_scalar('Metric/Recall', m['recall'], epoch)
        writer.add_scalar('Metric/PR_AUC', m['pr_auc'], epoch)
        writer.add_scalar('Metric/ROC_AUC', m['roc_auc'], epoch)

        print(f"Epoch {epoch} | Loss: {t_loss:.3f} | PR-AUC: {m['pr_auc']:.3f} | F1: {m['f1']:.3f}")

    writer.close()
    print(f"Done. Run directory: {run_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='b4')
    parser.add_argument('--weight_path', type=str, default=None, help='Path to pre-trained .pth file')
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--lr_step', type=int, default=10, help='Epoch at which to reduce LR by 10x')
    parser.add_argument('--img_size', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()

    train_classifier(args)