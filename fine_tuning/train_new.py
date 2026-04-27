import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, models
from tqdm import tqdm
from datetime import datetime
import cv2
import os
import numpy as np
from PIL import Image
from data.dataset import KaggleDRDataset
from .eval_new import evaluate_and_log


def apply_clahe_cv2(img_pil):
    img_np = np.array(img_pil)
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    return Image.fromarray(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    pbar = tqdm(loader, leave=False)
    for i, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device).long()
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pbar.set_postfix({'batch_loss': f"{loss.item():.4f}"})
    return total_loss / len(loader)


def train_classifier(epochs, img_size, pre_trained=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Setup Directory Structure
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    run_name = f"{timestamp}_Classifier_ep{epochs}_size{img_size}"
    run_dir = os.path.join("runs", run_name)
    os.makedirs(run_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=run_dir)

    # 2. Model & Data
    model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.DEFAULT if pre_trained else None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 5)
    model.to(device)

    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(size=img_size, scale=(0.7, 1.0)),
        transforms.Lambda(apply_clahe_cv2),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(180),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.Lambda(apply_clahe_cv2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = KaggleDRDataset('data/fused_dr_dataset/dr_unified_v2/dr_unified_v2/train',
                                    transform=train_transforms)
    val_dataset = KaggleDRDataset('data/fused_dr_dataset/dr_unified_v2/dr_unified_v2/val', transform=val_transforms)

    # Sampler Logic
    labels = np.array(train_dataset.labels)
    class_sample_count = np.array([len(np.where(labels == t)[0]) for t in range(5)])
    class_weights = 1. / class_sample_count
    sample_weights = torch.from_numpy(np.array([class_weights[t] for t in labels])).double()
    sampler = torch.utils.data.WeightedRandomSampler(weights=sample_weights, num_samples=10000, replacement=True)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, sampler=sampler)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_qwk = -1.0
    print(f"Starting training: {run_name}")

    for epoch in range(epochs):
        t_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Comprehensive Evaluation
        metrics = evaluate_and_log(model, val_loader, device, epoch, run_dir)

        # Logging to TensorBoard
        writer.add_scalar('Loss/train', t_loss, epoch)
        writer.add_scalar('Loss/val', metrics['loss'], epoch)
        writer.add_scalar('Accuracy/val', metrics['acc'], epoch)
        writer.add_scalar('Kappa/val', metrics['qwk'], epoch)
        writer.add_scalar('F1/Macro', metrics['f1_macro'], epoch)
        writer.add_scalar('PR_AUC/Mean', metrics['pr_auc_mean'], epoch)

        # Save weights based on QWK
        if metrics['qwk'] > best_qwk:
            best_qwk = metrics['qwk']
            save_path = os.path.join(run_dir, f"best_weights_epoch{epoch}.pth")
            # Remove previous best to keep folder clean (optional)
            torch.save(model.state_dict(), save_path)
            print(f"--> [Epoch {epoch}] New Best QWK: {best_qwk:.4f}")

        print(f"Epoch {epoch}: T_Loss: {t_loss:.3f} | V_Loss: {metrics['loss']:.3f} | QWK: {metrics['qwk']:.3f}")

    writer.close()
    return run_dir


if __name__ == "__main__":
    train_classifier(epochs=2, img_size=512)