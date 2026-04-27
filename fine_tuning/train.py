import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm
from datetime import datetime
import cv2
import os
import numpy as np
from PIL import Image
from torchvision import models
from sklearn.metrics import cohen_kappa_score, f1_score, confusion_matrix
from data.dataset import KaggleDRDataset
from utils import apply_clahe_cv2


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    pbar = tqdm(loader)
    for i, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device).unsqueeze(1)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pbar.set_postfix({'Avg_Loss': f"{total_loss / (i + 1):.4f}"})
    return total_loss / len(loader)


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device).long()
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            # Move to CPU for sklearn
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate Metrics
    epoch_loss = running_loss / len(loader)

    # Quadratic Weighted Kappa (The gold standard for DR)
    # This penalizes a [Class 0 vs Class 4] mistake MUCH more than [Class 0 vs Class 1]
    qwk = cohen_kappa_score(all_labels, all_preds, weights='quadratic')

    # Macro F1 treats all classes equally regardless of frequency
    f1 = f1_score(all_labels, all_preds, average='macro')

    # Accuracy for comparison
    acc = (np.array(all_preds) == np.array(all_labels)).mean()

    # Get confusion matrix to see where the "Bleeding" occurs
    cm = confusion_matrix(all_labels, all_preds)

    return {
        'loss': epoch_loss,
        'acc': acc,
        'qwk': qwk,
        'f1': f1,
        'cm': cm
    }


def get_run_name(model_type, epochs, img_size, pre_trained):
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    return f"{timestamp}_{model_type}_ep{epochs}_size{img_size}"


def train_classifier(epochs, img_size, pre_trained=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # TensorBoard setup
    run_name = get_run_name("Classifier", epochs, img_size, pre_trained)
    writer = SummaryWriter(log_dir=os.path.join("runs", run_name))

    model = models.efficientnet_b4()
    if pre_trained:
        model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 5)
    model.to(device)

    train_transforms = transforms.Compose([
        # Crop the image first, then resize to 512.
        # scale=(0.7, 1.0) means the crop will be between 70% and 100% of the original image area.
        # This simulates a restricted FOV without cropping out too much pathology.
        transforms.RandomResizedCrop(size=img_size, scale=(0.7, 1.0), ratio=(0.9, 1.1)),
        transforms.Lambda(apply_clahe_cv2),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(180),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.Lambda(apply_clahe_cv2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = KaggleDRDataset(
        'data/fused_dr_dataset/dr_unified_v2/dr_unified_v2/train',
        transform=train_transforms)
    val_dataset = KaggleDRDataset(
        'data/fused_dr_dataset/dr_unified_v2/dr_unified_v2/val',
        transform=val_transforms)

    # 1. Calculate weights for the entire dataset
    labels = np.array(train_dataset.labels)
    class_sample_count = np.array([len(np.where(labels == t)[0]) for t in range(5)])

    # Weight for each class is 1 / count
    class_weights = 1. / class_sample_count

    # Create an array that assigns the corresponding class weight to EVERY single image in the dataset
    sample_weights = np.array([class_weights[t] for t in labels])
    sample_weights = torch.from_numpy(sample_weights).double()

    # 2. Create the WeightedRandomSampler
    # num_samples=10000 means it will draw 10,000 images per epoch,
    # but it will draw them based on the weights, effectively balancing the classes.
    train_sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights,
        num_samples=10000,  # Or len(train_dataset) if you want a full epoch
        replacement=True  # MUST be True for oversampling to work
    )

    # 3. Use standard CrossEntropyLoss (no class weights needed here anymore)
    criterion = nn.CrossEntropyLoss()

    # 4. Pass sampler to DataLoader
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=32,
        sampler=train_sampler
    )

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    print(f"Starting Classifier training: {run_name}")
    pbar = tqdm(range(epochs), desc="Training")

    model_name = f"{run_name}.pth"
    best_qwk = -1.0  # Kappa ranges from -1 to 1
    for epoch in range(epochs):
        t_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        metrics = validate(model, val_loader, criterion, device)

        # Pull values out of dictionary
        v_loss = metrics['loss']
        v_qwk = metrics['qwk']
        v_f1 = metrics['f1']
        v_acc = metrics['acc']

        # Log to TensorBoard
        writer.add_scalar('Loss/train', t_loss, epoch)
        writer.add_scalar('Loss/val', v_loss, epoch)
        writer.add_scalar('Accuracy/val', v_acc, epoch)
        writer.add_scalar('Kappa/val', v_qwk, epoch)
        writer.add_scalar('F1/val', v_f1, epoch)

        # Save based on QWK
        if v_qwk > best_qwk:
            best_qwk = v_qwk
            torch.save(model.state_dict(), model_name)
            print(f"--> [Epoch {epoch}] Saved Best QWK: {v_qwk:.4f} | F1: {v_f1:.4f}")

        pbar.set_postfix({'T_Loss': f"{t_loss:.3f}", 'QWK': f"{v_qwk:.3f}"})

    writer.close()
    print(f"Finished. Logs saved to runs/{run_name}")
    return model_name


if __name__ == "__main__":
    train_classifier(25, 512)
