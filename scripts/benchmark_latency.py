"""
Benchmark inference latency and peak RSS for a ResNet defect classifier.
Uses a synthetic 224x224 tensor — no dataset required.
Requires at least one checkpoint under checkpoints/best_model_*.pt.

Usage:
  python scripts/benchmark_latency.py
  python scripts/benchmark_latency.py --checkpoint checkpoints/best_model_bottle.pt
"""

import argparse
import json
import resource
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import yaml

from src.model import DefectClassifier


def load_config(config_path="config.yaml"):
    with open(Path(ROOT) / config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def find_checkpoint(explicit=None):
    if explicit:
        path = Path(explicit)
        if not path.is_absolute():
            path = Path(ROOT) / path
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path
    ckpt_dir = Path(ROOT) / "checkpoints"
    checkpoints = sorted(ckpt_dir.glob("best_model_*.pt"))
    if not checkpoints:
        return None
    return checkpoints[0]


def percentile(sorted_values, p):
    if not sorted_values:
        return 0.0
    k = (len(sorted_values) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_values) - 1)
    if f == c:
        return sorted_values[f]
    return sorted_values[f] + (sorted_values[c] - sorted_values[f]) * (k - f)


def peak_rss_mb():
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes; Linux reports kilobytes
    if sys.platform == "darwin":
        return usage / (1024 * 1024)
    return usage / 1024


def benchmark(model, device, image_size, warmup, runs):
    x = torch.randn(1, 3, image_size, image_size, device=device)
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()

        times_ms = []
        for _ in range(runs):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times_ms.append((time.perf_counter() - t0) * 1000.0)

    times_ms.sort()
    mean_ms = sum(times_ms) / len(times_ms)
    return {
        "mean_ms": round(mean_ms, 3),
        "p50_ms": round(percentile(times_ms, 50), 3),
        "p95_ms": round(percentile(times_ms, 95), 3),
        "min_ms": round(times_ms[0], 3),
        "max_ms": round(times_ms[-1], 3),
        "n_runs": runs,
        "n_warmup": warmup,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark inference latency")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--checkpoint", default=None, help="Path to .pt checkpoint")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--output", default="results/latency.json")
    args = parser.parse_args()

    try:
        checkpoint = find_checkpoint(args.checkpoint)
    except FileNotFoundError as exc:
        print(exc)
        sys.exit(1)

    if checkpoint is None:
        print(
            "No checkpoints found under checkpoints/best_model_*.pt.\n"
            "Download a .pt from Google Drive (see README), then re-run:\n"
            "  python scripts/benchmark_latency.py"
        )
        sys.exit(0)

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_size = cfg["image"]["size"]

    model = DefectClassifier(
        backbone_name=cfg["model"]["name"],
        num_classes=cfg["model"]["num_classes"],
        pretrained=False,
    )
    ckpt = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)

    stats = benchmark(model, device, image_size, args.warmup, args.runs)
    payload = {
        "checkpoint": str(checkpoint.relative_to(ROOT)).replace("\\", "/"),
        "model": cfg["model"]["name"],
        "image_size": image_size,
        "device": str(device),
        "batch_size": 1,
        "latency": stats,
        "peak_rss_mb": round(peak_rss_mb(), 2),
    }

    output_path = Path(ROOT) / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Checkpoint: {payload['checkpoint']}")
    print(f"Device: {payload['device']}")
    print(
        f"Latency ms — mean: {stats['mean_ms']}, "
        f"p50: {stats['p50_ms']}, p95: {stats['p95_ms']}"
    )
    print(f"Peak RSS: {payload['peak_rss_mb']} MB")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
