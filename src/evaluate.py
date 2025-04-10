"""
Evaluate saved checkpoints on the validation split.
Reports validation accuracy and majority-class baseline per category.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.dataset import get_mvtec_categories, get_mvtec_dataloaders
from src.model import DefectClassifier
from src.train import evaluate


def load_config(config_path="config.yaml"):
    with open(Path(ROOT) / config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def majority_baseline(val_dataset):
    """Accuracy if we always predict the most common class in the val split."""
    labels = [label for _, label in val_dataset.samples]
    if not labels:
        return 0.0, 0
    counts = Counter(labels)
    return max(counts.values()) / len(labels), len(labels)


def evaluate_category(cfg, category, mvtec_root, device):
    category_dir = mvtec_root / category
    checkpoint_path = Path(ROOT) / cfg["output"]["checkpoint_dir"] / f"best_model_{category}.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    batch_size = cfg["training"]["batch_size"]
    image_size = cfg["image"]["size"]
    num_workers = cfg["training"]["num_workers"]
    train_ratio = cfg["data"].get("train_ratio", 0.8)

    _, val_loader, class_to_idx = get_mvtec_dataloaders(
        str(category_dir),
        batch_size=batch_size,
        image_size=image_size,
        num_workers=num_workers,
        train_ratio=train_ratio,
    )

    model = DefectClassifier(
        backbone_name=cfg["model"]["name"],
        num_classes=cfg["model"]["num_classes"],
        pretrained=False,
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    criterion = nn.CrossEntropyLoss()
    _, val_acc = evaluate(model, val_loader, criterion, device)
    baseline, n_val = majority_baseline(val_loader.dataset)

    return {
        "category": category,
        "val_accuracy": round(val_acc, 4),
        "majority_baseline": round(baseline, 4),
        "improvement_vs_baseline": round(val_acc - baseline, 4),
        "n_val_samples": n_val,
        "class_to_idx": ckpt.get("class_to_idx", class_to_idx),
        "checkpoint": str(checkpoint_path.relative_to(ROOT)).replace("\\", "/"),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate defect classifiers on validation split")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    parser.add_argument("--category", default=None, help="MVTec category (e.g. bottle)")
    parser.add_argument("--all-categories", action="store_true", help="Evaluate all categories with checkpoints")
    parser.add_argument("--output", default="results/metrics.json", help="Output JSON path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    mvtec_root = Path(cfg["data"].get("mvtec_root", "data"))
    if not mvtec_root.is_absolute():
        mvtec_root = Path(ROOT) / mvtec_root

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.all_categories:
        categories = get_mvtec_categories(mvtec_root)
        ckpt_dir = Path(ROOT) / cfg["output"]["checkpoint_dir"]
        categories = [
            c for c in categories
            if (ckpt_dir / f"best_model_{c}.pt").exists()
        ]
        if not categories:
            print("No checkpoints found for MVTec categories.")
            sys.exit(1)
    elif args.category:
        categories = [args.category]
    else:
        category = cfg["data"].get("mvtec_category")
        if not category:
            print("Specify --category or --all-categories")
            sys.exit(1)
        categories = [category]

    results = []
    for cat in categories:
        print(f"\n--- {cat} ---")
        try:
            metrics = evaluate_category(cfg, cat, mvtec_root, device)
            results.append(metrics)
            print(
                f"Val accuracy: {metrics['val_accuracy']:.2%} | "
                f"Baseline: {metrics['majority_baseline']:.2%} | "
                f"Improvement: {metrics['improvement_vs_baseline']:+.2%} | "
                f"n={metrics['n_val_samples']}"
            )
        except (FileNotFoundError, ValueError) as exc:
            print(f"Skipped {cat}: {exc}")

    if not results:
        print("No categories evaluated.")
        sys.exit(1)

    output_path = Path(ROOT) / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "split": "validation (random 80/20 from MVTec train+test pool)",
        "model": cfg["model"]["name"],
        "categories": results,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nMetrics saved to {output_path}")


if __name__ == "__main__":
    main()
