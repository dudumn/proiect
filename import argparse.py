import argparse
import os
import time
from pathlib import Path
import torch
from torchvision import datasets, models, transforms
from PIL import Image

#!/usr/bin/env python3
"""
recognition_template.py

Minimal image recognition template using PyTorch and torchvision.
- Transfer learning with ResNet18
- Training / validation loops
- Single-image prediction helper

Usage examples:
    Train:
        python recognition_template.py --mode train --data_dir ./data --epochs 5 --batch 32 --out model.pth
    Predict:
        python recognition_template.py --mode predict --model model.pth --image test.jpg

Customize:
- Replace transforms, model, optimizer, scheduler as needed.
- For custom datasets, adapt get_dataloaders().
"""


import torch.nn as nn
import torch.optim as optim

# ---------------------------
# Configuration / utilities
# ---------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
torch.manual_seed(SEED)


def get_dataloaders(data_dir, input_size=224, batch_size=32, num_workers=4, val_split=0.2):
        """
        Assumes data_dir has subfolders per class (like torchvision.datasets.ImageFolder).
        Returns train_loader, val_loader, class_names
        """
        data_dir = Path(data_dir)
        # Basic augmentation / normalization for transfer learning
        train_transforms = transforms.Compose([
                transforms.RandomResizedCrop(input_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                                         [0.229, 0.224, 0.225]),
        ])
        val_transforms = transforms.Compose([
                transforms.Resize(int(input_size * 1.14)),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                                         [0.229, 0.224, 0.225]),
        ])

        # Use ImageFolder; expects subfolders per class
        full_dataset = datasets.ImageFolder(str(data_dir), transform=train_transforms)
        class_names = full_dataset.classes
        n_total = len(full_dataset)
        n_val = int(n_total * val_split)
        n_train = n_total - n_val
        train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [n_train, n_val],
                                                                                                                             generator=torch.Generator().manual_seed(SEED))
        # Replace val transform
        val_dataset.dataset = datasets.ImageFolder(str(data_dir), transform=val_transforms)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                                                                             num_workers=num_workers, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                                                                         num_workers=num_workers, pin_memory=True)
        return train_loader, val_loader, class_names


def build_model(num_classes, feature_extract=True, pretrained=True):
        """
        Returns a model (ResNet18) with the final layer adapted to num_classes.
        If feature_extract is True, freezes feature extractor weights.
        """
        model = models.resnet18(pretrained=pretrained)
        if feature_extract:
                for param in model.parameters():
                        param.requires_grad = False
        # Replace final fc
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model.to(DEVICE)


# ---------------------------
# Training / Evaluation
# ---------------------------
def train_model(model, dataloaders, criterion, optimizer, scheduler=None, num_epochs=5, save_path=None):
        best_acc = 0.0
        for epoch in range(num_epochs):
                print(f"Epoch {epoch+1}/{num_epochs}")
                start = time.time()
                # Each epoch has train and val
                for phase in ("train", "val"):
                        if phase == "train":
                                model.train()
                                loader = dataloaders["train"]
                        else:
                                model.eval()
                                loader = dataloaders["val"]

                        running_loss = 0.0
                        running_corrects = 0
                        total = 0

                        for inputs, labels in loader:
                                inputs = inputs.to(DEVICE)
                                labels = labels.to(DEVICE)
                                optimizer.zero_grad()
                                with torch.set_grad_enabled(phase == "train"):
                                        outputs = model(inputs)
                                        loss = criterion(outputs, labels)
                                        _, preds = torch.max(outputs, 1)
                                        if phase == "train":
                                                loss.backward()
                                                optimizer.step()
                                running_loss += loss.item() * inputs.size(0)
                                running_corrects += torch.sum(preds == labels.data).item()
                                total += inputs.size(0)

                        epoch_loss = running_loss / total
                        epoch_acc = running_corrects / total
                        print(f"  {phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

                        if phase == "val" and scheduler is not None:
                                scheduler.step(epoch_loss)

                        if phase == "val" and epoch_acc > best_acc:
                                best_acc = epoch_acc
                                if save_path:
                                        torch.save(model.state_dict(), save_path)
                                        print(f"  Saved best model to {save_path}")

                print(f"  Time: {time.time() - start:.1f}s\n")
        print(f"Best val Acc: {best_acc:.4f}")
        return model


def evaluate_model(model, dataloader, criterion=None):
        """
        Evaluate model on dataloader. Returns (loss, accuracy).
        If criterion is None, loss is None.
        """
        model.eval()
        running_loss = 0.0
        running_corrects = 0
        total = 0
        with torch.no_grad():
                for inputs, labels in dataloader:
                        inputs = inputs.to(DEVICE)
                        labels = labels.to(DEVICE)
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        if criterion is not None:
                                loss = criterion(outputs, labels)
                                running_loss += loss.item() * inputs.size(0)
                        running_corrects += torch.sum(preds == labels.data).item()
                        total += inputs.size(0)
        loss = (running_loss / total) if criterion is not None else None
        acc = running_corrects / total
        return loss, acc


# ---------------------------
# Single-image prediction
# ---------------------------
def predict_image(model, image_path, class_names, input_size=224):
        model.eval()
        preprocess = transforms.Compose([
                transforms.Resize(int(input_size * 1.14)),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                                         [0.229, 0.224, 0.225]),
        ])
        img = Image.open(image_path).convert("RGB")
        tensor = preprocess(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
                outputs = model(tensor)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                top_prob, top_idx = torch.max(probs, 1)
        return class_names[top_idx.item()], top_prob.item()


# ---------------------------
# CLI
# ---------------------------
def parse_args():
        p = argparse.ArgumentParser(description="Image recognition template")
        p.add_argument("--mode", choices=["train", "predict"], required=True)
        p.add_argument("--data_dir", type=str, help="Path to dataset (ImageFolder structure)")
        p.add_argument("--image", type=str, help="Image path for prediction")
        p.add_argument("--model", type=str, help="Path to model weights (.pth)")
        p.add_argument("--out", type=str, default="model.pth", help="Where to save best model")
        p.add_argument("--epochs", type=int, default=5)
        p.add_argument("--batch", type=int, default=32)
        p.add_argument("--lr", type=float, default=1e-3)
        return p.parse_args()


def main():
        args = parse_args()
        if args.mode == "train":
                if not args.data_dir:
                        raise SystemExit("Training requires --data_dir")
                train_loader, val_loader, class_names = get_dataloaders(args.data_dir, batch_size=args.batch)
                model = build_model(num_classes=len(class_names), feature_extract=True, pretrained=True)
                # Only params that require grad will be optimized (final fc)
                params_to_update = [p for p in model.parameters() if p.requires_grad]
                optimizer = optim.Adam(params_to_update, lr=args.lr)
                criterion = nn.CrossEntropyLoss()
                scheduler = None
                dataloaders = {"train": train_loader, "val": val_loader}
                train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=args.epochs, save_path=args.out)
                print("Training finished.")
        elif args.mode == "predict":
                if not args.model or not args.image:
                        raise SystemExit("Prediction requires --model and --image")
                # For prediction we need class names. If you saved them separately, load them here.
                # This template assumes the same dataset structure is available under args.data_dir to load class names.
                if not args.data_dir:
                        raise SystemExit("Provide --data_dir pointing to dataset with same class folders to recover class names")
                _, _, class_names = get_dataloaders(args.data_dir, batch_size=1)
                model = build_model(num_classes=len(class_names), feature_extract=False, pretrained=False)
                model.load_state_dict(torch.load(args.model, map_location=DEVICE))
                label, prob = predict_image(model, args.image, class_names)
                print(f"Prediction: {label} (prob={prob:.4f})")


if __name__ == "__main__":
        main()