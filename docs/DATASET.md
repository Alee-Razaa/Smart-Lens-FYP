# 📊 Dataset & AI Model Guide

> **Smart Lens — YOLOv8 Threat Detection Model**

---

## 1. Dataset Source

The labeled dataset for training the Smart Lens threat detection model is hosted on **Roboflow**.

| Field | Details |
|-------|---------|
| **Platform** | Roboflow |
| **Workspace** | `fpy` |
| **Project** | `smart-survellaince-lens-2` |
| **Version** | 1 |
| **Format** | YOLOv8 |
| **API Key** | `7QsEv54uizzlrvPZ972Z` |

---

## 2. How to Download the Dataset

### Option A: Using Python (Recommended)

```python
!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="7QsEv54uizzlrvPZ972Z")
project = rf.workspace("fpy").project("smart-survellaince-lens-2")
version = project.version(1)
dataset = version.download("yolov8")
```

### Option B: From Google Colab

Run the above code in your Colab notebook after cloning the repo. The dataset will be downloaded to the Colab runtime.

---

## 3. Dataset Structure (YOLOv8 Format)

After downloading, the dataset follows the standard YOLOv8 directory structure:

```
smart-survellaince-lens-2-1/
├── data.yaml              ← Class names + paths config
├── train/
│   ├── images/            ← Training images
│   └── labels/            ← YOLO format annotations (.txt)
├── valid/
│   ├── images/            ← Validation images
│   └── labels/            ← Validation annotations
└── test/
    ├── images/            ← Test images
    └── labels/            ← Test annotations
```

### YOLO Label Format

Each `.txt` label file contains one line per object:
```
<class_id> <x_center> <y_center> <width> <height>
```
All values are normalized (0-1) relative to image dimensions.

---

## 4. Detection Classes

Based on the SRS, the Smart Lens AI model detects:

| Class ID | Event Type | SRS Reference |
|----------|-----------|---------------|
| 0 | **Theft** — Suspicious movement near sensitive zones | FR-3.3.1 |
| 1 | **Violence** — Fighting, crowd disturbances | FR-3.3.2 |
| 2 | **Fire/Smoke** — Fire and smoke hazards | FR-3.3.3 |
| 3 | **Gun** — Weapon detection | FR-3.3 |
| 4 | **Knife** — Weapon detection | FR-3.3 |

> ⚠️ **Note**: Exact class names and IDs are defined in the `data.yaml` file downloaded with the dataset. Check it after download to confirm.

---

## 5. AI Model Architecture

| Component | Details |
|-----------|---------|
| **Model** | YOLOv8 (Ultralytics) |
| **Framework** | PyTorch |
| **Task** | Object Detection |
| **Input** | Live CCTV frames via RTSP |
| **Output** | Bounding boxes + class labels + confidence scores |
| **Threshold** | Configurable confidence threshold to minimize false positives (FR-6.5) |

### Why YOLOv8?

- **Real-time performance** — designed for speed + accuracy on video streams
- **Pre-trained weights** — transfer learning from COCO dataset
- **Easy fine-tuning** — train on custom Roboflow dataset
- **Export formats** — ONNX, TensorRT, CoreML for deployment

---

## 6. Training Guide

### 6.1 Prerequisites

```bash
pip install ultralytics roboflow opencv-python torch
```

### 6.2 Training Script (Colab with GPU)

```python
from ultralytics import YOLO

# Load pre-trained YOLOv8 model
model = YOLO('yolov8n.pt')  # nano model (fast) — or yolov8s.pt, yolov8m.pt for better accuracy

# Train on Smart Lens dataset
results = model.train(
    data='smart-survellaince-lens-2-1/data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    name='smart_lens_v1',
    device=0,           # Use GPU
    patience=20,         # Early stopping
    save=True,
    plots=True
)
```

### 6.3 Validation

```python
# Validate the trained model
metrics = model.val()
print(f"mAP50: {metrics.box.map50:.4f}")
print(f"mAP50-95: {metrics.box.map:.4f}")
```

### 6.4 Inference (Test on Image/Video)

```python
# Run inference
results = model.predict(
    source='test_video.mp4',  # or image path, or RTSP URL
    conf=0.5,                  # confidence threshold
    save=True,
    show=True
)
```

### 6.5 Export Model for Deployment

```python
# Export to ONNX for production deployment
model.export(format='onnx')
```

---

## 7. Performance Targets


## 7. Model Performance

### v2 Metrics (2026-02-14)
| Metric | v1 (Original) | v2 (Fine-tuned) |
|--------|--------------|----------------|
| mAP50  | 0.7255       | 0.7536         |
| mAP50-95 | 0.3322     | 0.3513         |
| Precision | 0.8410    | 0.8495         |
| Recall    | 0.6213    | 0.6053         |

**Per-class AP50:** Fighting 0.80 | Fire 0.82 | Gun 0.60 | Knife 0.79

Test video: Only 1 true alert (Gun), no false positives.

### Performance Targets
| Metric | Target | SRS Reference |
|--------|--------|---------------|
| Alert latency | < 30 seconds from detection to notification | NFR-1.1 |
| False positive rate | Minimized via confidence thresholds | NFR-4.1 |
| Continuous cameras | Minimum 4 cameras simultaneously | NFR-1.3 |
| Video quality | Minimum 720p | NFR-1.4 |
| Uptime | 99% under normal conditions | NFR-1.5 |
| Operation | 24/7 continuous | NFR-4.5 |

---

## 8. Related Documents

## 9. Expanding the Dataset for v3

To further improve gun/weapon detection:
1. Go to [Roboflow Universe](https://universe.roboflow.com/) and search for "gun detection" or "weapon detection" datasets.
2. For each dataset, click **Download Dataset → YOLOv8 → show download code** and copy the `workspace`, `project`, and `version`.
3. Add these to the `ADDITIONAL_DATASETS` list in **finetune_smart_lens_v2.ipynb** (see template in Cell 2B).
4. Run the notebook to merge, fine-tune, and export a new model.

- [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md) — High-level project summary
- [SRS.md](SRS.md) — Full Software Requirements Specification
- [SDS.md](SDS.md) — Full Software Design Specification
