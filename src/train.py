"""
Training script: CNN with ResNet transfer learning for defect / no_defect.
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.dataset import (
    get_dataloaders,
    get_mvtec_dataloaders,
    get_mvtec_categories,
    check_data_structure,
    check_mvtec_category,
)
from src.model import DefectClassifier


def load_config(config_path="config.yaml"):
    with open(Path(ROOT) / config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in tqdm(loader, desc="Train", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
    return total_loss / len(loader), correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += loss.item()
        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
    return total_loss / len(loader), correct / total


def run_training(cfg, train_loader, val_loader, class_to_idx, save_path, epochs, device):
    """Train one model and save best checkpoint to save_path."""
    model = DefectClassifier(
        backbone_name=cfg["model"]["name"],
        num_classes=cfg["model"]["num_classes"],
        pretrained=cfg["model"]["pretrained"],
    ).to(device)

    freeze_epochs = cfg["model"].get("freeze_backbone_epochs", 0)
    if freeze_epochs > 0:
        model.freeze_backbone()
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg["training"]["learning_rate"],
            weight_decay=cfg["training"]["weight_decay"],
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg["training"]["learning_rate"],
            weight_decay=cfg["training"]["weight_decay"],
        )

    criterion = nn.CrossEntropyLoss()
    best_val_acc = 0.0

    for epoch in range(epochs):
        if epoch == freeze_epochs:
            model.unfreeze_all()
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=cfg["training"]["learning_rate"],
                weight_decay=cfg["training"]["weight_decay"],
            )

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1}/{epochs} | Train loss: {train_loss:.4f} acc: {train_acc:.4f} | Val loss: {val_loss:.4f} acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "class_to_idx": class_to_idx,
            }, save_path)
            print(f"  -> New best model saved (acc={val_acc:.4f})")

    return best_val_acc


def main():
    parser = argparse.ArgumentParser(description="Train defect / no_defect classifier")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    parser.add_argument("--train-dir", default=None, help="Override train_dir from config")
    parser.add_argument("--val-dir", default=None, help="Override val_dir from config")
    parser.add_argument("--mvtec-dir", default=None, help="MVTec AD root (e.g. data/mvtec_ad)")
    parser.add_argument("--category", default=None, help="MVTec category (e.g. metal_nut, bottle)")
    parser.add_argument("--all-categories", action="store_true", help="Train one model per MVTec category; saves best_model_<category>.pt")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch_size")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    batch_size = args.batch_size or cfg["training"]["batch_size"]
    epochs = args.epochs or cfg["training"]["epochs"]
    image_size = cfg["image"]["size"]
    num_workers = cfg["training"]["num_workers"]

    # MVTec AD: use dataset as-is (train/ + test/ inside one category folder)
    mvtec_root = args.mvtec_dir or data_cfg.get("mvtec_root")
    mvtec_category = args.category or data_cfg.get("mvtec_category")

    if args.all_categories and mvtec_root:
        # Train one model per category; save best_model_<category>.pt
        root_path = Path(mvtec_root)
        if not root_path.is_absolute():
            root_path = Path(ROOT) / root_path
        categories = get_mvtec_categories(root_path)
        if not categories:
            print(f"No MVTec categories found in {root_path}")
            sys.exit(1)
        print(f"Training all categories: {categories}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        out_dir = Path(ROOT) / cfg["output"]["checkpoint_dir"]
        out_dir.mkdir(parents=True, exist_ok=True)
        for cat in categories:
            category_dir = root_path / cat
            print(f"\n--- Category: {cat} ---")
            train_loader, val_loader, class_to_idx = get_mvtec_dataloaders(
                str(category_dir),
                batch_size=batch_size,
                image_size=image_size,
                num_workers=num_workers,
                train_ratio=data_cfg.get("train_ratio", 0.8),
                split_mode=data_cfg.get("split_mode", "pooled_random"),
            )
            save_path = out_dir / f"best_model_{cat}.pt"
            run_training(cfg, train_loader, val_loader, class_to_idx, str(save_path), epochs, device)
        print("\nAll categories trained.")
        return

    save_path_suffix = None  # None = use config default (best_model.pt)
    if mvtec_root and mvtec_category:
        category_dir = Path(mvtec_root) / mvtec_category
        if not category_dir.is_absolute():
            category_dir = Path(ROOT) / category_dir
        ok, msg = check_mvtec_category(category_dir)
        if not ok:
            print(f"MVTec data error: {msg}")
            sys.exit(1)
        print(f"Using MVTec category: {category_dir} ({msg})")
        train_loader, val_loader, class_to_idx = get_mvtec_dataloaders(
            str(category_dir),
            batch_size=batch_size,
            image_size=image_size,
            num_workers=num_workers,
            train_ratio=data_cfg.get("train_ratio", 0.8),
            split_mode=data_cfg.get("split_mode", "pooled_random"),
        )
        save_path_suffix = mvtec_category  # save as best_model_bottle.pt etc.
    else:
        train_dir = args.train_dir or Path(ROOT) / data_cfg["train_dir"]
        val_dir = args.val_dir or Path(ROOT) / data_cfg["val_dir"]
        ok, msg = check_data_structure(train_dir)
        if not ok:
            print(f"Train data error: {msg}")
            sys.exit(1)
        ok, msg = check_data_structure(val_dir)
        if not ok:
            print(f"Val data error: {msg}")
            sys.exit(1)
        train_loader, val_loader, class_to_idx = get_dataloaders(
            str(train_dir), str(val_dir),
            batch_size=batch_size,
            image_size=image_size,
            num_workers=num_workers,
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Classes: {class_to_idx}")

    out_dir = Path(ROOT) / cfg["output"]["checkpoint_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    if save_path_suffix:
        best_path = out_dir / f"best_model_{save_path_suffix}.pt"
    else:
        best_path = Path(ROOT) / cfg["output"]["best_model"]
    run_training(cfg, train_loader, val_loader, class_to_idx, str(best_path), epochs, device)
    print("Training done.")


if __name__ == "__main__":
    main()
