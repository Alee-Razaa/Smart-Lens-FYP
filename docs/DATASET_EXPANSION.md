# Smart Lens v2 — Dataset Expansion Guide

## Current Dataset
- **Total**: 1,793 images (Train: 1,569 | Valid: 149 | Test: 75)
- **Classes**: Fighting (0), Fire (1), Gun (2), Knife (3)
- **Source**: Roboflow (workspace: fpy, project: smart-survellaince-lens-2)

## Recommended Open-Source Datasets (Roboflow Universe)

### 🔫 Gun / Weapon Datasets (PRIORITY — weakest class)
| Dataset | URL | Size | Notes |
|---------|-----|------|-------|
| Handgun Detection | roboflow.com/object-detections-jgfbs/handgun-detection-yjebf | ~3K+ | CCTV-style handgun images |
| Pistol Detection | roboflow.com/dss-cctv/pistol-detection-bvaxq | ~1K+ | Surveillance camera angles |
| Gun Object Detection | roboflow.com/weapondetection/gun-object-detection | ~2K+ | Various firearms |
| Weapon Detection | roboflow.com/university-bswku/weapon-detection-cctv | ~1K+ | CCTV weapon detection |

### 🔪 Knife Datasets
| Dataset | URL | Size | Notes |
|---------|-----|------|-------|
| Knife Detection | roboflow.com/knife-5wkrr/knife-detection-l8s6h | ~800+ | Various knife images |
| Sharp Objects | roboflow.com/sharp-objects | ~500+ | Knives and blades |

### 🔥 Fire / Smoke Datasets
| Dataset | URL | Size | Notes |
|---------|-----|------|-------|
| Fire and Smoke | roboflow.com/dsti/fire-and-smoke-dsti | ~2K+ | Indoor/outdoor fire |
| Fire Detection YOLO | roboflow.com/fire-detection-yolo | ~1K+ | Fire-focused |

### 👊 Fighting / Violence Datasets
| Dataset | URL | Size | Notes |
|---------|-----|------|-------|
| Violence Detection | roboflow.com/violence-detection | ~1K+ | Surveillance fighting |

## How to Add a New Dataset to the Notebook

1. **Find the dataset** on [Roboflow Universe](https://universe.roboflow.com/)
2. **Get the API details** (workspace, project, version)
3. **Add it to `ADDITIONAL_DATASETS`** in Cell 2B of the notebook:

```python
{
    "name": "Your Dataset Name",
    "workspace": "workspace-slug",
    "project": "project-slug",
    "version": 1,
    "api_key": "7QsEv54uizzlrvPZ972Z",
    "class_map": {"source_class_name": target_id},  # Map source → your IDs
    "target_classes": [2],  # Which of YOUR classes this dataset covers
},
```

### Class Mapping Examples
If a dataset has class `"handgun"` and you want it as **Gun** (ID 2):
```python
"class_map": {"handgun": 2, "gun": 2, "pistol": 2}
```

If a dataset has class `"fire"` and `"smoke"`, map both to **Fire** (ID 1):
```python
"class_map": {"fire": 1, "smoke": 1}
```

## Fine-Tuning Tips

| Tip | Details |
|-----|---------|
| **Don't over-train** | 50-100 epochs is enough for fine-tuning |
| **Low learning rate** | Use 0.0003-0.0005 (not 0.001 like initial training) |
| **Balance classes** | Try to add equal amounts per class |
| **Iterative** | Fine-tune v1→v2, then v2→v3 with even more data |
| **Quality > Quantity** | 500 well-labeled gun images > 2000 noisy ones |
| **Watch validation loss** | If val loss rises while train loss falls = overfitting |

## Target Dataset Size (Recommended)
For production-quality detection:
- **Per class**: 2,000-5,000 images
- **Total**: 8,000-20,000 images
- **Current gap**: Need ~6,000-18,000 more images
