"""Smoke tests for the Image Defect Detection project."""

import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def test_imports():
    import src.dataset  # noqa: F401
    import src.model  # noqa: F401
    import src.train  # noqa: F401
    import src.predict  # noqa: F401
    import src.evaluate  # noqa: F401
    import src.metrics_utils  # noqa: F401


def test_config_loads():
    config_path = ROOT / "config.yaml"
    assert config_path.exists()
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    assert cfg["model"]["name"] in ("resnet18", "resnet34", "resnet50")
    assert cfg["model"]["num_classes"] == 2
    assert cfg["data"].get("split_mode", "pooled_random") in (
        "pooled_random",
        "official_holdout",
    )


def test_predict_transform_shape():
    from src.predict import get_inference_transform

    transform = get_inference_transform(224)
    from PIL import Image

    img = Image.new("RGB", (100, 100))
    tensor = transform(img)
    assert tensor.shape == (3, 224, 224)


def test_checkpoint_load():
    ckpt_dir = ROOT / "checkpoints"
    checkpoints = sorted(ckpt_dir.glob("best_model_*.pt"))
    if not checkpoints:
        pytest.skip("No checkpoints available locally")

    from src.predict import load_model

    path = str(checkpoints[0])
    model, idx_to_class, image_size = load_model(path)
    assert model is not None
    assert image_size == 224
    assert set(idx_to_class.values()) <= {"defect", "no_defect"}
