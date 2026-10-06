# 🔍 Smart Lens — AI-Powered CCTV Surveillance System

> **Final Year Project (FYP)** — IBA Sukkur | Project Code: F22-49

[![GitHub](https://img.shields.io/badge/GitHub-Smart--Lens--FYP-blue)](https://github.com/Alee-Razaa/Smart-Lens-FYP)

---

## 🎯 What is Smart Lens?

Smart Lens is an **AI-powered surveillance solution** that transforms traditional passive CCTV systems into intelligent, proactive security tools for small businesses. It uses **YOLOv8 deep learning** to detect fighting, fire, guns and knives in real time — instantly alerting shopkeepers via a mobile app.

### Key Features
- 🤖 **AI Threat Detection** — Fighting, fire, guns, knives (theft detection was in the original proposal and is not implemented yet)
- 📱 **Instant Mobile Alerts** — Push notifications with video evidence
- 📹 **Smart Recording** — The detector runs only on frames with motion, and only confirmed threats are saved (saves storage)
- 🎥 **Multi-Camera Support** — Monitor multiple cameras from one dashboard
- 🚨 **Alert Forwarding** — Share alerts with law enforcement or contacts
- 🔐 **2FA Security** — JWT + OTP authentication

---

## 👥 Team

| Name | Role | ID |
|------|------|----|
| **Ali Raza Memon** | Developer | 023-22-0200 |
| **Aadil Shah** | Developer | 023-22-0106 |
| **Waseem Mazari** | Developer | 023-22-0102 |
| **Madam Faryal Shamsi** | Supervisor | — |

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| AI Model | YOLOv8 (PyTorch) |
| Video Processing | OpenCV |
| Backend | FastAPI (Python) |
| Mobile App | Flutter |
| Database | Supabase (PostgreSQL) |
| Cloud Storage | Backblaze B2 |
| Notifications | Firebase (FCM) |
| Dataset | Roboflow |

---

## 📁 Project Structure

```
Smart-Lens-FYP/
├── docs/                        # 📄 Project Documentation
│   ├── PROJECT_OVERVIEW.md      #    Project summary & architecture
│   ├── SRS.md                   #    Software Requirements Specification
│   ├── SDS.md                   #    Software Design Specification
│   └── DATASET.md               #    Dataset & AI model training guide
├── colab_setup.ipynb            # 🚀 Google Colab GPU training setup
├── Labeled data api.txt         # 🏷️ Roboflow dataset API
├── Smart Lens SRS Finalized.docx   # Original SRS document
├── Smart Lens SDS Finalized.pdf    # Original SDS document
├── .gitignore
└── README.md                    # ← You are here
```

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/Alee-Razaa/Smart-Lens-FYP.git
cd Smart-Lens-FYP
```

### 2. Train or Fine-Tune the AI Model (Google Colab + GPU)
1. Open [Google Colab](https://colab.research.google.com)
2. **File → Open Notebook → GitHub** → search `Alee-Razaa/Smart-Lens-FYP`
3. Open **colab_setup.ipynb** (for training from scratch) or **finetune_smart_lens_v2.ipynb** (to fine-tune with more data)
4. Enable **T4 GPU**: Runtime → Change runtime type → T4 GPU
5. Run all cells in order

#### v2 Model Results (2026-02-14)
| Metric     | v1 (Original) | v2 (Fine-tuned) |
|------------|--------------|----------------|
| mAP50      | 0.7255       | 0.7536         |
| mAP50-95   | 0.3322       | 0.3513         |
| Precision  | 0.8410       | 0.8495         |
| Recall     | 0.6213       | 0.6053         |

**Per-class AP50:** Fighting 0.80 | Fire 0.82 | Gun 0.60 | Knife 0.79

**Test video:** Only 1 true alert (Gun), no false positives.

#### To train v3 with more gun/weapon data:
1. Go to [Roboflow Universe](https://universe.roboflow.com/) and search for "gun detection" or "weapon detection" datasets.
2. For each dataset, click **Download Dataset → YOLOv8 → show download code** and copy the `workspace`, `project`, and `version`.
3. Add these to the `ADDITIONAL_DATASETS` list in **finetune_smart_lens_v2.ipynb** (see template in Cell 2B).
4. Run the notebook to merge, fine-tune, and export a new model.

### 3. Download Dataset
```python
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_ROBOFLOW_API_KEY")
project = rf.workspace("fpy").project("smart-survellaince-lens-2")
version = project.version(1)
dataset = version.download("yolov8")
```

---

## 📄 Documentation

| Document | Description |
|----------|-------------|
| [Project Overview](docs/PROJECT_OVERVIEW.md) | High-level summary, architecture, objectives |
| [SRS](docs/SRS.md) | Full Software Requirements Specification |
| [SDS](docs/SDS.md) | Full Software Design Specification |
| [Dataset Guide](docs/DATASET.md) | Dataset info, training guide, model architecture |

---

## 📜 License

This project is developed as part of the Final Year Project at IBA Sukkur.
