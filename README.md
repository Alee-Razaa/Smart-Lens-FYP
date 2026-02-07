# 🔍 Smart Lens — AI-Powered CCTV Surveillance System

> **Final Year Project (FYP)** — IBA Sukkur | Project Code: F22-49

[![GitHub](https://img.shields.io/badge/GitHub-Smart--Lens--FYP-blue)](https://github.com/Alee-Razaa/Smart-Lens-FYP)

---

## 🎯 What is Smart Lens?

Smart Lens is an **AI-powered surveillance solution** that transforms traditional passive CCTV systems into intelligent, proactive security tools for small businesses. It uses **YOLOv8 deep learning** to detect threats like theft, violence, fire, and weapons in real-time — instantly alerting shopkeepers via a mobile app.

### Key Features
- 🤖 **AI Threat Detection** — Theft, violence, fire, guns, knives
- 📱 **Instant Mobile Alerts** — Push notifications with video evidence
- 📹 **Smart Recording** — Only saves suspicious events (saves storage)
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

### 2. Train the AI Model (Google Colab + GPU)
1. Open [Google Colab](https://colab.research.google.com)
2. **File → Open Notebook → GitHub** → search `Alee-Razaa/Smart-Lens-FYP`
3. Open **colab_setup.ipynb**
4. Enable **T4 GPU**: Runtime → Change runtime type → T4 GPU
5. Run all cells in order

### 3. Download Dataset
```python
from roboflow import Roboflow
rf = Roboflow(api_key="7QsEv54uizzlrvPZ972Z")
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
