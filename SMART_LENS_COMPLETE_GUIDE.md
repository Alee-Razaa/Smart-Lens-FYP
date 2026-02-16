# 🎓 Smart Lens — Complete Learning Guide

> **A Comprehensive Guide to Understanding, Running, and Developing the Smart Lens AI Surveillance System**

**Last Updated:** February 16, 2026 | **Version:** 3.0

---

## 📚 Table of Contents

1. [Introduction — What You'll Learn](#introduction)
2. [Project Overview — The Problem & Solution](#project-overview)
3. [Understanding AI Object Detection](#understanding-ai)
4. [System Architecture Deep Dive](#architecture)
5. [Dataset & Training from Scratch](#dataset-training)
6. [Setting Up Your Local Environment](#local-setup)
7. [Running the Detection System](#running-detection)
8. [Code Deep Dive — How It Works](#code-deep-dive)
9. [Performance Analysis & Results](#performance)
10. [Advanced Topics — Fine-Tuning & Optimization](#advanced-topics)
11. [Next Steps — Backend & Mobile App](#next-steps)
12. [Troubleshooting & FAQ](#troubleshooting)

---

<a name="introduction"></a>
## 1. 📖 Introduction — What You'll Learn

Welcome to the **Smart Lens Complete Learning Guide**! This document will help you:

✅ **Understand** the entire project from concept to implementation  
✅ **Learn** how AI-powered object detection works (YOLOv8)  
✅ **Set up** your own development environment  
✅ **Train** custom AI models on Google Colab (free GPU)  
✅ **Run** threat detection on videos, webcams, and images  
✅ **Analyze** model performance and improve accuracy  
✅ **Extend** the system with new features  

### Who is this guide for?

- **Final Year Students** working on similar AI/ML projects
- **Developers** who want to build surveillance systems
- **Researchers** interested in real-time object detection
- **Anyone** curious about how AI can solve real-world security problems

### Prerequisites

- Basic Python programming (functions, loops, conditionals)
- Familiarity with command line / terminal
- Basic understanding of machine learning concepts (optional but helpful)

---

<a name="project-overview"></a>
## 2. 🎯 Project Overview — The Problem & Solution

### The Problem

Small shops in Pakistan face serious security challenges:

- **Theft & Robbery:** Frequent incidents, especially in local markets
- **Fire Hazards:** Electrical fires, cooking equipment risks
- **Violence & Disputes:** Customer/vendor altercations
- **Passive CCTV:** Traditional cameras only record — no alerts, no prevention

**Current pain points:**
- 📼 Hours of footage to review manually *after* incidents
- 💰 High costs for live security guards
- ⏰ Delayed response times (police arrive too late)
- 📊 99% of recorded footage is never watched

### Our Solution: Smart Lens

**Smart Lens** transforms passive CCTV into an **intelligent, proactive AI security system**.

| Feature | Benefit |
|---------|---------|
| 🤖 **AI Threat Detection** | Automatically detects: Guns, Knives, Fighting, Fire |
| ⚡ **Real-Time Alerts** | Instant mobile push notifications (3-5 second latency) |
| 💾 **Smart Storage** | Only saves suspicious clips (saves 95% storage) |
| 📱 **Mobile Dashboard** | Monitor multiple cameras from your phone |
| 🚨 **Alert Forwarding** | Share alerts with police, neighbors, family |

---

<a name="understanding-ai"></a>
## 3. 🧠 Understanding AI Object Detection

### What is Object Detection?

**Object Detection** = Finding objects in images/videos + drawing boxes around them + labeling what they are.

**Example:**
```
Input: Video frame of a shop
Output: "Gun detected at (x=320, y=180, w=40, h=60)" → Trigger alert!
```

### Why YOLOv8?

**YOLO** = "You Only Look Once" — A fast, accurate AI model for real-time detection.

**Why we chose YOLOv8:**
- ✅ **Fast:** 30-60 FPS on GPU, 10-15 FPS on CPU
- ✅ **Accurate:** 70-85% mAP (mean Average Precision)
- ✅ **Easy to Train:** Works with small datasets (1000-2000 images)
- ✅ **Pre-trained:** Can fine-tune existing models quickly

**YOLOv8 Model Sizes:**
| Model | Parameters | Speed | Accuracy | Use Case |
|-------|-----------|-------|----------|----------|
| YOLOv8n | 3M | Fastest | Lowest | Mobile, IoT |
| YOLOv8s | **11M** | Fast | Good | **← We use this** |
| YOLOv8m | 26M | Medium | Better | Balanced |
| YOLOv8l | 44M | Slower | Best | High accuracy needs |
| YOLOv8x | 68M | Slowest | Highest | Research |

### How YOLOv8 Works (Simplified)

```
1. INPUT: Video frame (640x640 pixels)
   ↓
2. BACKBONE: Extract features (edges, shapes, textures)
   ↓
3. NECK: Combine features at different scales (detect small & large objects)
   ↓
4. HEAD: Predict bounding boxes + confidence scores
   ↓
5. OUTPUT: [{"class": "Gun", "confidence": 0.85, "bbox": [x, y, w, h]}]
```

**Our 4 Classes:**
- **Class 0:** Fighting (physical altercations, violence)
- **Class 1:** Fire (flames, smoke)
- **Class 2:** Gun (pistols, rifles, weapons)
- **Class 3:** Knife (blades, sharp objects)

### Key Concepts

**Confidence Score:** How sure the AI is (0.0-1.0)
- `0.45` = 45% confident = "Maybe a gun?"
- `0.85` = 85% confident = "Definitely a gun!" ← We use this

**IoU (Intersection over Union):** How well the predicted box matches the real object
- `IoU = 0.5` = 50% overlap
- `IoU = 0.8` = 80% overlap = Good detection!

**mAP (mean Average Precision):** Overall model accuracy across all classes
- `mAP50 = 0.75` = 75% accurate at 50% IoU threshold
- Higher is better!

---

<a name="architecture"></a>
## 4. 🏗️ System Architecture Deep Dive

### High-Level Overview

```
┌─────────────────────────────────────────────────┐
│  📱 MOBILE APP (Flutter)                        │
│  • User dashboard                               │
│  • Alert notifications                          │
│  • Camera management                            │
└──────────────────┬──────────────────────────────┘
                   │ HTTPS/WebSocket
┌──────────────────▼──────────────────────────────┐
│  🖥️ BACKEND SERVER (FastAPI)                    │
│  • User authentication (JWT + 2FA)              │
│  • Camera feed management                       │
│  • Alert routing & storage                      │
└──────────────────┬──────────────────────────────┘
                   │ RTSP Stream
┌──────────────────▼──────────────────────────────┐
│  🤖 AI DETECTION PIPELINE (Python)              │
│  ┌──────────────────────────────────────────┐   │
│  │ 1. MOTION DETECTION (OpenCV MOG2)        │   │
│  │    Skip frames with no motion (saves GPU)│   │
│  └──────────────┬───────────────────────────┘   │
│                 │                                │
│  ┌──────────────▼───────────────────────────┐   │
│  │ 2. YOLO INFERENCE (YOLOv8s)              │   │
│  │    Detect: Fighting, Fire, Gun, Knife    │   │
│  └──────────────┬───────────────────────────┘   │
│                 │                                │
│  ┌──────────────▼───────────────────────────┐   │
│  │ 3. TEMPORAL FILTERING                    │   │
│  │    Require 3/8 frames to confirm threat  │   │
│  └──────────────┬───────────────────────────┘   │
│                 │                                │
│  ┌──────────────▼───────────────────────────┐   │
│  │ 4. ALERT GENERATION                      │   │
│  │    • Save video clip                     │   │
│  │    • Upload to Backblaze B2              │   │
│  │    • Send FCM notification               │   │
│  └──────────────────────────────────────────┘   │
└─────────────────────────────────────────────────┘
```

### Data Flow Example

**Scenario:** A person pulls out a gun in a shop.

```
Step 1: Camera captures frame → Motion detected
Step 2: YOLO processes frame → "Gun detected (conf=0.87)"
Step 3: Store in buffer [frame_42: Gun]
Step 4: Check buffer: Gun seen in 4 out of last 8 frames → THRESHOLD PASSED
Step 5: Generate alert:
   • Class: Gun
   • Confidence: 0.87
   • Timestamp: 2026-02-16 14:23:17
   • Video clip: Save 5 seconds before + 5 seconds after
Step 6: Upload clip to Backblaze B2
Step 7: Send push notification to shopkeeper's phone
Step 8: Optional: Forward to police WhatsApp/Email
```

**Total time from detection to alert:** ~3-5 seconds

---

<a name="dataset-training"></a>
## 5. 📊 Dataset & Training from Scratch

### Understanding Our Dataset

**Location:** `Smart-Lens-FYP/dataset/`

```
dataset/
├── train/          # 1569 images (87.5%)
│   ├── images/     # .jpg files
│   └── labels/     # .txt files (YOLO format)
├── valid/          # 149 images (8.3%)
│   ├── images/
│   └── labels/
└── test/           # 75 images (4.2%)
    ├── images/
    └── labels/
```

**Total:** 1793 images

**Class Distribution:**
- Fighting: ~450 annotations
- Fire: ~380 annotations
- Gun: ~340 annotations (weakest class)
- Knife: ~310 annotations

### YOLO Label Format

Each `.txt` file contains bounding box annotations:

```
<class_id> <x_center> <y_center> <width> <height>
```

**Example:** `20221010_222352_jpg.rf.01f15750648dffbb24538a3945b62a9e.txt`
```
2 0.512 0.387 0.124 0.218
```

**Translation:**
- Class 2 = Gun
- Center at (51.2%, 38.7%) of image
- Width = 12.4% of image width
- Height = 21.8% of image height

### Training v1 Model (From Scratch)

**File:** `colab_setup.ipynb`

**Steps:**
1. **Upload dataset to Google Drive** (or download from Roboflow)
2. **Open Colab:** [colab.research.google.com](https://colab.research.google.com)
3. **Enable GPU:** Runtime → Change runtime type → T4 GPU
4. **Run all cells** in `colab_setup.ipynb`

**Training Configuration (v1):**
```python
model = YOLO('yolov8s.pt')  # Start from pre-trained weights
results = model.train(
    data='dataset/data.yaml',
    epochs=200,           # How many times to see all images
    batch=16,             # Images per training step
    imgsz=640,            # Image size (640x640)
    device=0,             # GPU device
    optimizer='AdamW',    # Optimization algorithm
    lr0=0.001,            # Initial learning rate
    patience=30,          # Stop if no improvement for 30 epochs
)
```

**Training Time:** ~2.5 hours on T4 GPU

**v1 Results:**
- mAP50: **0.7255** (72.55% accuracy)
- Precision: 0.8410 (84% of detections are correct)
- Recall: 0.6213 (62% of actual threats are detected)

### Fine-Tuning v2 Model (With More Data)

**File:** `finetune_smart_lens_v2.ipynb`

**What is fine-tuning?**
- Start from existing `best.pt` (already trained)
- Train on **additional datasets** from Roboflow Universe
- Use **lower learning rate** (0.0005 vs 0.001) to preserve learned features
- Shorter training time (~40-60 min vs 2.5 hours)

**v2 Configuration:**
```python
model = YOLO('best.pt')  # Start from v1 model
results = model.train(
    data='merged_dataset/data.yaml',
    epochs=100,           # Fewer epochs (v1 used 200)
    batch=16,
    imgsz=640,
    lr0=0.0005,           # Lower learning rate
    patience=30,
)
```

**v2 Results:**
- mAP50: **0.7536** (+2.8% improvement!)
- Precision: 0.8495
- Recall: 0.6053

**Per-class AP50:**
- Fire: **0.8249** (strongest)
- Fighting: 0.7998
- Knife: 0.7882
- Gun: **0.6012** (weakest — needs more data!)

### Preparing v3 Training (Current Goal)

**Objective:** Improve Gun detection from 0.60 → 0.70+ mAP50

**Strategy:**
1. Add 7 gun/weapon datasets from Roboflow Universe
2. Increase epochs: 100 → 150
3. Increase patience: 30 → 50
4. Merge ~3000-5000 additional gun images

**Expected v3 Results:**
- mAP50: 0.75 → **0.78** (+3%)
- Gun AP50: 0.60 → **0.72** (+12%)

---

<a name="local-setup"></a>
## 6. 💻 Setting Up Your Local Environment

### Prerequisites

- **OS:** Windows 10/11, macOS, or Linux
- **Python:** 3.9 - 3.14 (we use 3.14.2)
- **RAM:** 8GB minimum, 16GB recommended
- **Storage:** 5GB free space

### Step-by-Step Setup

#### 1. Clone the Repository

```powershell
cd C:\Users\YourName\Desktop
git clone https://github.com/Alee-Razaa/Smart-Lens-FYP.git
cd Smart-Lens-FYP
```

#### 2. Create Virtual Environment

**Windows:**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**macOS/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### 3. Install Dependencies

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install ultralytics opencv-python numpy pillow pyyaml
```

**Package Versions (tested):**
- PyTorch: 2.10.0+cpu
- Ultralytics: 8.4.14
- OpenCV: 4.13.0

#### 4. Fix DLL Errors (Windows Only)

If you see `torch.dll not found` error:

**Install Visual C++ Redistributable:**
1. Go to: [https://aka.ms/vs/17/release/vc_redist.x64.exe](https://aka.ms/vs/17/release/vc_redist.x64.exe)
2. Download and run installer
3. Restart terminal
4. Test: `python -c "import torch; print(torch.__version__)"`

#### 5. Download Trained Model

**Option A:** From Google Drive (if you trained on Colab)
```powershell
# Download best.pt from Colab to your local machine
# Place in: Smart-Lens-FYP/trained_models/smart_lens_v2_20260214_1101/
```

**Option B:** From GitHub Releases (not yet available)

#### 6. Verify Installation

```powershell
cd Smart-Lens-FYP
python test_model.py --help
```

**Expected output:**
```
usage: test_model.py [-h] --model MODEL --source SOURCE [--conf CONF]

Testing Smart Lens v1 model
```

✅ **Setup Complete!** You're ready to run detections.

---

<a name="running-detection"></a>
## 7. 🚀 Running the Detection System

### Test on a Single Image

```powershell
python test_model.py --model trained_models/smart_lens_v2_20260214_1101/best.pt --source path/to/image.jpg
```

**Output:**
- Saved image with bounding boxes: `runs/detect/predict/image.jpg`
- Console log: Detected objects with confidence scores

### Test on a Video File

```powershell
python test_model.py --model trained_models/smart_lens_v2_20260214_1101/best.pt --source path/to/video.mp4
```

**Output:**
- Processed video: `runs/detect/predict/video.mp4`
- FPS: ~10-15 on CPU, ~30-60 on GPU

### Test on Webcam (Real-Time)

```powershell
python test_model.py --model trained_models/smart_lens_v2_20260214_1101/best.pt --source 0
```

**Controls:**
- `q` = Quit
- Frame-by-frame processing

### Using v2 Enhanced Pipeline (Recommended!)

**smart_lens_v2.py** adds:
- ✅ Motion gating (skip frames with no movement)
- ✅ Temporal filtering (require 3/8 frames to confirm)
- ✅ Alert cooldown (5 seconds between alerts)
- ✅ Per-class confidence thresholds

```powershell
python smart_lens_v2.py --model trained_models/smart_lens_v2_20260214_1101/best.pt --source path/to/video.mp4
```

**Keyboard Shortcuts:**
- `q` = Quit
- `p` = Pause/Resume
- `+` = Increase confidence threshold
- `-` = Decrease confidence threshold
- `m` = Toggle motion detection
- `d` = Toggle detection visualization
- `s` = Save current frame

**Example Output:**
```
Frame 46/73 | FPS: 12.3 | Motion: YES | Raw: 1 Gun | Filtered: 1 Gun
✅ ALERT [Gun] Confidence: 0.87 (Confirmed in 4/8 frames)
```

### Batch Testing on Test Set

```powershell
python test_model.py --model trained_models/smart_lens_v2_20260214_1101/best.pt --source dataset/test/images
```

**Output:**
```
Processing 75 test images...
✅ 57/75 images with detections (76%)
Total detections: 64
  • Gun: 25
  • Fighting: 19
  • Fire: 12
  • Knife: 8
```

---

<a name="code-deep-dive"></a>
## 8. 🔍 Code Deep Dive — How It Works

### test_model.py (Simple Inference)

**Purpose:** Basic testing script for quick inference.

**Key Functions:**

```python
def load_model(model_path):
    """Load YOLOv8 model from .pt file"""
    model = YOLO(model_path)
    return model

def process_frame(frame, model, conf_threshold=0.45):
    """Run inference on a single frame"""
    results = model(frame, conf=conf_threshold)
    return results[0]  # Detections for this frame

def draw_boxes(frame, detections):
    """Draw bounding boxes and labels"""
    for box in detections.boxes:
        x1, y1, x2, y2 = box.xyxy[0]  # Coordinates
        conf = box.conf[0]             # Confidence
        cls = int(box.cls[0])          # Class ID
        label = f"{model.names[cls]} {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1-10), font, 0.5, color, 2)
```

**Main Loop (Video):**
```python
cap = cv2.VideoCapture(source)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    results = process_frame(frame, model)
    annotated_frame = draw_boxes(frame, results)
    
    cv2.imshow("Smart Lens", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
```

### smart_lens_v2.py (Production Pipeline)

**File:** `Smart-Lens-FYP/smart_lens_v2.py` (911 lines)

**Architecture:**

```python
class MotionDetector:
    """Detects motion using background subtraction"""
    def __init__(self):
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500,       # Frames to remember
            varThreshold=16,   # Sensitivity
            detectShadows=True
        )
    
    def has_motion(self, frame, min_area=1500):
        """Returns True if significant motion detected"""
        fg_mask = self.bg_subtractor.apply(frame)
        contours, _ = cv2.findContours(fg_mask, ...)
        for contour in contours:
            if cv2.contourArea(contour) > min_area:
                return True
        return False


class TemporalFilter:
    """Filters false positives by requiring consistent detections"""
    def __init__(self, buffer_size=8, min_confirmations=3):
        self.buffer = deque(maxlen=buffer_size)  # Last 8 frames
        self.min_confirmations = 3               # Need 3 frames
    
    def add_detection(self, class_name, confidence):
        """Add detection to buffer"""
        self.buffer.append({
            'class': class_name,
            'conf': confidence,
            'timestamp': time.time()
        })
    
    def is_confirmed(self, class_name):
        """Check if threat is confirmed (3/8 frames)"""
        recent_detections = [d for d in self.buffer if d['class'] == class_name]
        return len(recent_detections) >= self.min_confirmations


class AlertManager:
    """Manages alert cooldown to prevent spam"""
    def __init__(self, cooldown_seconds=5):
        self.last_alerts = {}  # {class_name: timestamp}
        self.cooldown = cooldown_seconds
    
    def can_alert(self, class_name):
        """Check if enough time has passed since last alert"""
        if class_name not in self.last_alerts:
            return True
        elapsed = time.time() - self.last_alerts[class_name]
        return elapsed > self.cooldown
    
    def record_alert(self, class_name):
        """Record that alert was sent"""
        self.last_alerts[class_name] = time.time()


# Main Detection Loop
motion_detector = MotionDetector()
temporal_filter = TemporalFilter()
alert_manager = AlertManager()

while True:
    frame = get_frame()
    
    # Step 1: Motion gating (skip if no motion)
    if not motion_detector.has_motion(frame):
        continue
    
    # Step 2: YOLO inference
    results = model(frame, conf=0.45)
    
    # Step 3: Add to temporal buffer
    for detection in results:
        temporal_filter.add_detection(detection.class, detection.conf)
    
    # Step 4: Check for confirmed threats
    for class_name in ['Fighting', 'Fire', 'Gun', 'Knife']:
        if temporal_filter.is_confirmed(class_name):
            if alert_manager.can_alert(class_name):
                send_alert(class_name, frame)
                alert_manager.record_alert(class_name)
```

**Key Features:**

1. **Motion Gating:** Skip ~70% of frames with no motion (saves GPU)
2. **Temporal Filtering:** Require 3/8 frames to confirm (reduces false positives)
3. **Per-Class Confidence:**
   - Fire, Gun: 0.45 (more sensitive, life-threatening)
   - Knife: 0.50
   - Fighting: 0.55 (less urgent)
4. **Alert Cooldown:** 5 seconds between alerts (prevents spam)

**Test Results:**
- Test video: 73 frames total
- Motion frames: 26 (35%)
- Raw detections: 26 Gun detections
- After class filtering: 16 Gun detections
- After temporal filtering: **1 confirmed Gun alert** ✅
- False positives: **0** ✅

---

<a name="performance"></a>
## 9. 📊 Performance Analysis & Results

### Model Comparison: v1 vs v2

| Metric | v1 (Original) | v2 (Fine-tuned) | Change |
|--------|--------------|----------------|--------|
| **mAP50** | 0.7255 | 0.7536 | +2.8% ✅ |
| **mAP50-95** | 0.3322 | 0.3513 | +1.9% ✅ |
| **Precision** | 0.8410 | 0.8495 | +0.85% ✅ |
| **Recall** | 0.6213 | 0.6053 | -1.6% 🔻 |
| **Training Time** | 150 min | 43 min | -72% ✅ |
| **Best Epoch** | ~180 | 47 | — |

### Per-Class Performance (v2)

| Class | AP50 | Strength | Action Needed |
|-------|------|----------|---------------|
| **Fire** | 0.8249 | Strongest | Maintain |
| **Fighting** | 0.7998 | Good | Maintain |
| **Knife** | 0.7882 | Good | Maintain |
| **Gun** | 0.6012 | **Weakest** | **Add more data** ⚠️ |

**Why is Gun weakest?**
- Fewer training images (~340 vs ~450 for Fighting)
- Guns are small objects (harder to detect)
- More variation (pistols, rifles, colors, angles)

**Solution for v3:**
- Add 7 gun/weapon datasets (~3000-5000 images)
- Increase training epochs (100 → 150)
- Expected improvement: 0.60 → 0.72 mAP50

### Real-World Test: Video Analysis

**Test Video:** 73 frames, shoplifting scenario with gun

**Pipeline Performance:**

```
Input:     73 frames
           ↓
Motion:    26 frames with motion (35% reduction)
           ↓
YOLO:      26 raw Gun detections (conf > 0.45)
           ↓
Class:     16 Gun detections (conf > 0.45 for Gun)
           ↓
Temporal:  1 confirmed Gun alert (4/8 frames)
           ↓
Output:    1 alert at frame 46 ✅
```

**Metrics:**
- **True Positives:** 1 Gun alert (correct!)
- **False Positives:** 0 (excellent!)
- **False Negatives:** 0 (detected the threat)
- **Latency:** ~3 seconds from detection to confirmation

### Batch Test: 75 Test Images

```
Total images: 75
Images with detections: 57 (76%)
Total detections: 64

Class distribution:
  • Gun: 25 (39%)
  • Fighting: 19 (30%)
  • Fire: 12 (19%)
  • Knife: 8 (12%)
```

### Inference Speed

| Hardware | FPS | Latency |
|----------|-----|---------|
| CPU (Intel i5) | 10-15 | 66-100ms |
| CPU (Intel i7) | 15-20 | 50-66ms |
| GPU (NVIDIA T4) | 40-60 | 16-25ms |
| GPU (RTX 3060) | 60-90 | 11-16ms |

**For real-time surveillance:**
- Need: 15+ FPS (acceptable)
- Recommended: 30+ FPS (smooth)

---

<a name="advanced-topics"></a>
## 10. 🚀 Advanced Topics — Fine-Tuning & Optimization

### Adding More Data from Roboflow Universe

**Steps:**

1. **Search for datasets:** [universe.roboflow.com](https://universe.roboflow.com)
   - Search: "gun detection", "weapon detection", "fire detection"
   
2. **Check dataset quality:**
   - ✅ 500+ images
   - ✅ High-quality annotations
   - ✅ Similar to your use case (surveillance cameras, not drones)

3. **Get download code:**
   - Click "Download Dataset" → YOLOv8 → "show download code"
   - Copy: `workspace`, `project`, `version`

4. **Add to notebook:**
   
   Edit `finetune_smart_lens_v2.ipynb` Cell 2B:
   
   ```python
   ADDITIONAL_DATASETS = [
       {
           "name": "Gun Detection Dataset",
           "workspace": "weapon-tzz7v",
           "project": "gun-ekh5e",
           "version": 1,
           "class_map": {"gun": 2, "pistol": 2, "weapon": 2},
       },
       # Add more datasets here
   ]
   ```

5. **Run training on Colab**

### Hyperparameter Tuning

**Key parameters to experiment with:**

| Parameter | v2 Value | Suggested Range | Effect |
|-----------|----------|-----------------|--------|
| `epochs` | 100 | 100-200 | More training time |
| `patience` | 30 | 30-50 | Early stopping threshold |
| `batch` | 16 | 8-32 | Memory vs speed |
| `imgsz` | 640 | 640-1280 | Accuracy vs speed |
| `lr0` | 0.0005 | 0.0001-0.001 | Learning speed |
| `dropout` | 0.05 | 0.0-0.3 | Prevent overfitting |

**For v3 training:**
- Increase `epochs`: 100 → 150
- Increase `patience`: 30 → 50
- Keep `imgsz=640` (1280 is 4× slower)

### Model Export Formats

**PyTorch (.pt)** — Default, best for Python
```python
model.export(format='torchscript')
```

**ONNX (.onnx)** — Cross-platform, C++/JavaScript
```python
model.export(format='onnx')
```

**TensorRT (.engine)** — NVIDIA GPU optimization (fastest)
```python
model.export(format='engine', device=0)
```

**TFLite (.tflite)** — Mobile (Android/iOS)
```python
model.export(format='tflite')
```

### Reducing False Positives

**Current methods in smart_lens_v2.py:**

1. **Motion Gating:** Skip frames with no motion
2. **Temporal Filtering:** Require 3/8 frames
3. **Confidence Thresholds:** Different per class
4. **Alert Cooldown:** 5 seconds between alerts

**Additional techniques:**

5. **ROI Filtering:** Only detect in specific areas
   ```python
   def is_in_roi(bbox, roi_polygon):
       center_x = (bbox[0] + bbox[2]) / 2
       center_y = (bbox[1] + bbox[3]) / 2
       return cv2.pointPolygonTest(roi_polygon, (center_x, center_y), False) >= 0
   ```

6. **Size Filtering:** Ignore very small/large detections
   ```python
   def is_valid_size(bbox, min_area=500, max_area=50000):
       w = bbox[2] - bbox[0]
       h = bbox[3] - bbox[1]
       area = w * h
       return min_area < area < max_area
   ```

7. **Tracking:** Follow objects across frames (coming soon)

### Class Weights for Imbalanced Data

If one class has few samples, increase its weight:

```python
# Calculate class weights
class_counts = [450, 380, 340, 310]  # Fighting, Fire, Gun, Knife
total = sum(class_counts)
weights = [total / (4 * count) for count in class_counts]

# Apply in training
model.train(
    data='data.yaml',
    class_weights=weights,  # [1.0, 1.18, 1.32, 1.45]
    ...
)
```

### Ensemble Methods (Advanced)

Use multiple models and vote:

```python
models = [
    YOLO('smart_lens_v1.pt'),
    YOLO('smart_lens_v2.pt'),
    YOLO('smart_lens_v3.pt'),
]

def ensemble_predict(frame):
    votes = {'Fighting': 0, 'Fire': 0, 'Gun': 0, 'Knife': 0}
    for model in models:
        results = model(frame)
        for detection in results:
            votes[detection.class] += detection.conf
    
    # Return class with highest total confidence
    return max(votes, key=votes.get)
```

---

<a name="next-steps"></a>
## 11. 🔮 Next Steps — Backend & Mobile App

### Current Status (Phase 1: AI Model ✅)

✅ v1 model trained (mAP50: 0.7255)  
✅ v2 model fine-tuned (mAP50: 0.7536)  
✅ Local testing environment  
✅ Enhanced detection pipeline (smart_lens_v2.py)  
✅ Documentation  
🔄 v3 training prepared (awaiting execution)  

### Phase 2: Backend API (In Progress)

**Tech Stack:**
- **Framework:** FastAPI (Python)
- **Database:** Supabase (PostgreSQL)
- **Storage:** Backblaze B2 (video clips)
- **Auth:** JWT + 2FA (OTP)

**API Endpoints (Planned):**

```
POST   /auth/register        — Create account
POST   /auth/login           — Login (returns JWT)
POST   /auth/verify-otp      — Verify 2FA code
GET    /cameras              — List user's cameras
POST   /cameras              — Add new camera
PUT    /cameras/{id}         — Update camera
DELETE /cameras/{id}         — Delete camera
GET    /alerts               — Get alert history
GET    /alerts/{id}/video    — Download alert video
POST   /alerts/{id}/share    — Share alert (email/WhatsApp)
WS     /stream/{camera_id}   — Live video feed
```

**Database Schema (Planned):**

```sql
-- Users table
CREATE TABLE users (
    id UUID PRIMARY KEY,
    email VARCHAR(255) UNIQUE,
    password_hash VARCHAR(255),
    phone VARCHAR(20),
    created_at TIMESTAMP,
    otp_secret VARCHAR(32)
);

-- Cameras table
CREATE TABLE cameras (
    id UUID PRIMARY KEY,
    user_id UUID REFERENCES users(id),
    name VARCHAR(100),
    rtsp_url VARCHAR(500),
    location VARCHAR(200),
    status VARCHAR(20),  -- 'online', 'offline'
    created_at TIMESTAMP
);

-- Alerts table
CREATE TABLE alerts (
    id UUID PRIMARY KEY,
    camera_id UUID REFERENCES cameras(id),
    threat_type VARCHAR(20),  -- 'Gun', 'Knife', 'Fire', 'Fighting'
    confidence FLOAT,
    video_url VARCHAR(500),   -- Backblaze B2 URL
    thumbnail_url VARCHAR(500),
    created_at TIMESTAMP,
    acknowledged BOOLEAN
);
```

### Phase 3: Mobile App (Not Started)

**Tech Stack:**
- **Framework:** Flutter (iOS + Android)
- **State Management:** Riverpod/Provider
- **Video Player:** video_player package
- **Notifications:** Firebase Cloud Messaging (FCM)

**Screens:**

1. **Login / Register** — Email + password + 2FA
2. **Dashboard** — List of cameras with live thumbnails
3. **Camera View** — Live stream + controls
4. **Alerts** — Timeline of threats with video clips
5. **Alert Detail** — Watch video, share, acknowledge
6. **Settings** — Manage cameras, notification preferences
7. **Share Alert** — Send to contacts/police

**Key Features:**

- 📹 Live video streaming (RTSP → HLS/WebRTC)
- 🔔 Push notifications (FCM)
- 📱 Picture-in-Picture mode
- 🌙 Dark mode support
- 🔒 Biometric lock (fingerprint/Face ID)

### Phase 4: Deployment (Future)

**Options:**

1. **Cloud Deployment (AWS/GCP/Azure)**
   - FastAPI on EC2/Cloud Run
   - Supabase hosted DB
   - Backblaze B2 for storage
   - **Cost:** ~$50-100/month

2. **On-Premise Deployment (Local Server)**
   - Run on shop owner's PC/NVR
   - Local PostgreSQL
   - Local storage (HDD/NAS)
   - **Cost:** One-time hardware ($300-500)

3. **Hybrid (Recommended)**
   - AI processing on local PC (fast, private)
   - Cloud storage for backups
   - Cloud notifications (FCM)
   - **Cost:** ~$20-30/month

### Timeline Estimate

| Phase | Task | Duration |
|-------|------|----------|
| ✅ 1 | AI Model v1-v2 | 1 week (done) |
| 🔄 1 | AI Model v3 | 2-3 days (in progress) |
| 📝 2 | Backend API | 2-3 weeks |
| 📝 2 | Database Setup | 1 week |
| 📝 2 | Cloud Storage Integration | 1 week |
| 📝 3 | Mobile App UI | 2-3 weeks |
| 📝 3 | Video Streaming | 1-2 weeks |
| 📝 3 | Notifications | 1 week |
| 📝 4 | Testing & Deployment | 2 weeks |
| — | **Total** | **12-14 weeks** |

---

<a name="troubleshooting"></a>
## 12. 🔧 Troubleshooting & FAQ

### Common Errors

#### 1. "torch.dll not found" (Windows)

**Solution:** Install Visual C++ Redistributable

```powershell
# Download and run installer
# https://aka.ms/vs/17/release/vc_redist.x64.exe
```

#### 2. "CUDA out of memory"

**Solution:** Reduce batch size

```python
model.train(
    data='data.yaml',
    batch=8,  # Reduce from 16
    imgsz=640,  # Or reduce from 1280
)
```

#### 3. "No module named 'cv2'"

**Solution:** Install OpenCV

```powershell
pip install opencv-python
```

#### 4. "Can't open video file"

**Solution:** Check codec support

```powershell
# Install codec support
pip install opencv-python-headless
```

#### 5. Model not detecting anything

**Checklist:**
- ✅ Confidence threshold too high? Try `--conf 0.25`
- ✅ Using correct model path?
- ✅ Video resolution too low? (need 640×480 minimum)
- ✅ Object too small in frame? (need 20×20 pixels minimum)

### FAQ

**Q: Can I train on CPU instead of GPU?**  
A: Yes, but it's 10-20× slower. 200 epochs on CPU = ~24 hours. Use Colab's free T4 GPU.

**Q: How much data do I need for good accuracy?**  
A: Minimum 500 images per class. We have 1793 total, but Gun class needs more.

**Q: Can I add more classes (e.g., "Shoplifting")?**  
A: Yes! Collect 500+ images, annotate in Roboflow, retrain model. Shoplifting is hard to define visually though.

**Q: Why is recall lower than precision?**  
A: Model is conservative (fewer false positives, but misses some threats). Acceptable for surveillance.

**Q: Can I run this on Raspberry Pi?**  
A: Yes, but FPS will be ~2-5. Use YOLOv8n (nano) model, reduce resolution to 320×320.

**Q: How to reduce false positives further?**  
A: Increase temporal filter from 3/8 to 4/8 or 5/8. Tradeoff: may miss real threats.

**Q: Can I use pre-trained COCO weights?**  
A: COCO has "person" and "knife" but not "fighting", "fire", "gun". Fine-tuning from COCO helps for knife, but not other classes.

**Q: How to handle multiple cameras?**  
A: Run separate Python processes for each camera, or use multi-threading. Backend API will handle this.

**Q: What's the best confidence threshold?**  
A: Start with 0.45. Increase to 0.55 for fewer false positives, decrease to 0.35 for more detections.

---

## 🎓 Learning Resources

### YOLOv8 Documentation
- [Ultralytics Docs](https://docs.ultralytics.com)
- [YOLOv8 GitHub](https://github.com/ultralytics/ultralytics)

### Computer Vision
- [PyImageSearch Tutorials](https://pyimagesearch.com)
- [OpenCV Documentation](https://docs.opencv.org)

### Deep Learning
- [Fast.ai Course](https://course.fast.ai) — Practical deep learning
- [Andrew Ng's ML Course](https://coursera.org/learn/machine-learning) — Theory

### Python Development
- [Real Python](https://realpython.com)
- [Python Docs](https://docs.python.org/3/)

---

## 📞 Contact & Support

**Team Smart Lens:**
- Ali Raza Memon: 023-22-0200
- Aadil Shah: 023-22-0106
- Waseem Mazari: 023-22-0102

**Supervisor:** Madam Faryal Shamsi

**GitHub:** [github.com/Alee-Razaa/Smart-Lens-FYP](https://github.com/Alee-Razaa/Smart-Lens-FYP)

---

## 📄 License

This project is developed as a Final Year Project for IBA Sukkur. For commercial use, contact the team.

---

**Last Updated:** February 16, 2026 | **Version:** 3.0

Happy Learning! 🚀🎓
