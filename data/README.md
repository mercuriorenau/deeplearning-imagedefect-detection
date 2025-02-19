# Dataset for Image Defect Detection

## Option 1: MVTec AD (no copying)

Put MVTec AD category folders directly under `data/`. Example:

```
data/
  bottle/
    train/          ← good images (or train/good/)
    test/
      good/         ← good test images
      broken_large/ ← defect type
      ...
  metal_nut/
    train/
    test/
      good/
      bent/
      ...
```

Do **not** copy or rename anything. In `config.yaml` set:

```yaml
data:
  mvtec_root: "data"
  mvtec_category: "metal_nut"   # or bottle, cable, capsule, etc.
```

The code uses `train/` + `test/good/` as **no_defect** and all other `test/*` folders as **defect**, then splits train/val (80/20) in code.

**One category:**

```bash
python -m src.train --category metal_nut
```

**All categories (one model per folder):**

```bash
python -m src.train --all-categories
```

Saves `checkpoints/best_model_bottle.pt`, `checkpoints/best_model_metal_nut.pt`, and so on.

## Option 2: Custom folders (defect / no_defect)

```
data/
  train/
    defect/
    no_defect/
  val/
    defect/
    no_defect/
```

- Supported formats: `.jpg`, `.png`, `.jpeg`, `.bmp`, `.webp`
- Folder names must be exactly `defect` and `no_defect`
