# 🔍 Smart Lens — Project Overview

> **AI-Powered CCTV Surveillance System for Small-Scale Businesses**

---

## 📋 Project Information

| Field | Details |
|-------|---------|
| **Project Name** | Smart Lens CCTV Surveillance System |
| **Project Code** | F22-49 |
| **Supervisor** | Madam Faryal Shamsi |
| **Team Members** | Ali Raza Memon (023-22-0200), Aadil Shah (023-22-0106), Waseem Mazari (023-22-0102) |
| **Institution** | IBA Sukkur |
| **Submission Date** | 25-01-2025 |

---

## 🎯 Problem Statement

Small and medium-scale shops are a vital part of developing economies like Pakistan but are highly vulnerable to security threats such as **theft, looting, fire, and violence**. Their current security relies on traditional, passive CCTV systems that are:

- **Costly to maintain** and ineffective for prompt prevention
- Generate **massive volumes of unfiltered footage**, creating data overload
- Rely on **error-prone human monitoring** for review *after* an incident has occurred
- Large-scale government projects (like PSCA) exist but **do not serve small retailers**

> **Smart Lens** aims to fill this gap by using AI to provide continuous threat detection and alerting.

---

## 💡 Solution

Smart Lens is an **AI-powered surveillance solution** designed for small-scale businesses and local markets. The system integrates **deep learning for continuous threat detection** with a **user-friendly mobile application**.

### Core Capabilities

| Feature | Description |
|---------|-------------|
| 🤖 **AI-Driven Threat Detection** | Analyses live CCTV feeds to detect theft, violence, looting, and fire hazards using behavioral pattern recognition |
| 📹 **Motion-Based Recording** | Only saves relevant footage containing suspicious or anomalous events, reducing storage costs |
| 📱 **Instant Mobile Notifications** | Cross-platform mobile app delivers instant alerts with event type, camera location, and video clip |
| 🎥 **Multi-Camera Integration** | Monitor multiple cameras simultaneously from one dashboard |
| 🔧 **Camera Management** | Add, edit, and remove cameras with metadata (location labels) |
| 🚨 **Third-Party Sharing** | Forward alerts to friends, family, community groups, or local law enforcement |

---

## 🏗️ System Architecture

Smart Lens uses a **Modular, Multi-tiered Vertical Architecture**:

```
┌─────────────────────────────────────────────────┐
│  Layer V: Client Tier                           │
│  └── Flutter Mobile App (2FA, Live Feeds, Alerts│
├─────────────────────────────────────────────────┤
│  Layer IV: Persistence Layer                    │
│  ├── Supabase (PostgreSQL) — Metadata, Users    │
│  └── Local HDD — Motion-triggered recordings    │
├─────────────────────────────────────────────────┤
│  Layer III: External Cloud Services             │
│  ├── Backblaze B2 — Evidence-grade threat clips │
│  ├── Firebase (FCM) — Push notifications        │
│  └── SMTP Email API — Alert forwarding          │
├─────────────────────────────────────────────────┤
│  Layer II: Application Tier (FastAPI Server)    │
│  ├── AI & Motion Pipeline (OpenCV + YOLO/CNN)   │
│  ├── Session Manager (JWT)                      │
│  ├── Smart Storage Manager                      │
│  └── Notification Dispatcher                    │
├─────────────────────────────────────────────────┤
│  Layer I: Data Acquisition Layer                │
│  └── IP Cameras (RTSP Protocol)                 │
└─────────────────────────────────────────────────┘
```

---

## 🛠️ Technology Stack

| Component | Technology |
|-----------|-----------|
| **AI Engine** | Python, PyTorch/TensorFlow, YOLOv8 |
| **Video Processing** | OpenCV |
| **Backend Server** | FastAPI (Python) |
| **Mobile Application** | Flutter |
| **Database** | Supabase (PostgreSQL) |
| **Cloud Storage** | Backblaze B2 |
| **Push Notifications** | Firebase Cloud Messaging (FCM) |
| **Authentication** | JWT + 2FA (OTP via Email/SMS) |
| **Version Control** | Git & GitHub |
| **AI Training Data** | Roboflow (YOLOv8 format) |

---

## 👥 Stakeholders

| Stakeholder | Role |
|-------------|------|
| **Primary Users** | Small business owners and shopkeepers in local markets |
| **Secondary Users** | Local security personnel and community watch groups |
| **External Parties** | Local law enforcement who may receive forwarded alerts |
| **Development Team** | Ali Raza Memon, Aadil Shah, Waseem Mazari |
| **Project Supervisor** | Ma'am Faryal Shamsi |

---

## 🎯 Project Objectives

1. **Automate** the manual and inefficient process of monitoring CCTV footage
2. **Smart Detection** of security threats (theft, fire, violence) for prompt intervention
3. **Reduce Storage Costs** via intelligent, motion-based recording
4. **Empower Shopkeepers** with instant, actionable alerts and video evidence via mobile app
5. **Enhanced Safety** — immediate alerts to local law enforcement upon user consent
6. **Affordable & Scalable** solution tailored for small businesses in developing regions

---

## 🚫 Out of Scope

- **Facial Recognition / Biometric Tracking** — focuses on behavioral patterns, not identifying individuals
- **Audio Surveillance** — analysis limited to video feeds only
- **POS Integration** — no integration with cash registers or transaction data

---

## 📁 Repository Structure

```
Smart-Lens-FYP/
├── docs/
│   ├── PROJECT_OVERVIEW.md      ← You are here
│   ├── SRS.md                   ← Software Requirements Specification
│   ├── SDS.md                   ← Software Design Specification
│   └── DATASET.md               ← Dataset & AI Model Info
├── colab_setup.ipynb            ← Google Colab GPU training setup
├── Labeled data api.txt         ← Roboflow dataset API
├── .gitignore
└── README.md
```

---

## 📄 Related Documents

- [SRS.md](SRS.md) — Full Software Requirements Specification
- [SDS.md](SDS.md) — Full Software Design Specification
- [DATASET.md](DATASET.md) — Dataset & AI Model Training Guide
