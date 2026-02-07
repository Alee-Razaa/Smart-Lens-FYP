# 🏗️ Software Design Specification (SDS)

> **Smart Lens CCTV Surveillance System — Technical Blueprint**

---

## 1. Introduction

### 1.1 Purpose

This Software Design Specification (SDS) provides a detailed architectural and component-level design for the Smart Lens system. It translates the functional and non-functional requirements defined in the [SRS](SRS.md) into a structured technical blueprint for the development, implementation, and testing teams.

### 1.2 Definitions & Acronyms

| Term | Definition |
|------|-----------|
| AI | Artificial Intelligence |
| CNN | Convolutional Neural Network |
| RTSP | Real-Time Streaming Protocol (video stream access) |
| FCM | Firebase Cloud Messaging |
| JWT | JSON Web Token (secure session management) |
| 2FA | Two-Factor Authentication (mandatory secondary security layer) |
| Threat Detection | AI engine analyzing video to identify suspicious activities |
| Motion-Based Recording | Storage-saving feature — video recorded only on motion/event detection |

---

## 2. System Architecture

Smart Lens employs a **Modular, Multi-tiered Vertical Architecture** designed for real-time processing and high security.

### 2.1 Architecture Layers

```
┌─────────────────────────────────────────────────────────┐
│  LAYER V — Client Tier                                  │
│  └── Flutter Mobile App                                 │
│      • Live feed monitoring                             │
│      • 2FA-secured alerts                               │
│      • Camera management                                │
├─────────────────────────────────────────────────────────┤
│  LAYER IV — Persistence Layer                           │
│  ├── Supabase (PostgreSQL)                              │
│  │   • Alert Metadata                                   │
│  │   • User & Session Data                              │
│  │   • Audit Logs                                       │
│  └── Local HDD                                          │
│      • High-volume motion-triggered recordings          │
├─────────────────────────────────────────────────────────┤
│  LAYER III — External Cloud Services                    │
│  ├── Backblaze B2 — Evidence-grade threat clips         │
│  ├── Firebase (FCM) — Instant push notifications        │
│  └── SMTP Email API — Alert forwarding to contacts      │
├─────────────────────────────────────────────────────────┤
│  LAYER II — Application Tier (FastAPI Server)           │
│  ├── AI & Motion Pipeline                               │
│  │   • OpenCV (motion filtering)                        │
│  │   • YOLO/CNN Threat Classifier                       │
│  ├── Session Manager (JWT)                              │
│  ├── Smart Storage Manager                              │
│  └── Notification Dispatcher                            │
├─────────────────────────────────────────────────────────┤
│  LAYER I — Data Acquisition Layer                       │
│  └── IP Cameras → RTSP Protocol                         │
└─────────────────────────────────────────────────────────┘
```

---

## 3. Domain Model

### 3.1 Domain Entities

| Entity | Description | Key Attributes |
|--------|-------------|---------------|
| **User (Shopkeeper)** | Primary stakeholder — manages cameras and receives alerts | `userID` (PK), `email`, `passwordHash`, `is_2fa_enabled` |
| **Session** | Manages secure interaction periods via JWT | `sessionID` (PK), `jwtToken`, `expiry` |
| **Camera** | Physical IP camera registered in the system | `cameraID` (PK), `userID` (FK), `streamURL` (RTSP), `status` |
| **AI Model** | Specific version of the detection engine | `modelID` (PK), `version`, `lastTrained` |
| **Alert (Event Log)** | Record of a detected suspicious event | `alertID` (PK), `cameraID` (FK), `eventType`, `confidenceScore` |
| **Video Clip** | Media file recorded upon motion/suspicion detection | `clipID` (PK), `alertID` (FK), `localPath`, `cloudURL` |
| **Audit Log** | Security record tracking all system activities | `logID` (PK), `activityType`, `timestamp` |

### 3.2 Conceptual Relationships

```
User ──┬── maintains ──→ Session (1:many, JWT-authenticated)
       ├── owns ────────→ Camera  (1:many)
       │
Camera ── triggers ─────→ Alert   (1:many)
       │
Alert ─── records ──────→ Video Clip (1:1)
```

- **User ↔ Session**: Every user action authenticated via JWT
- **User ↔ Camera**: Shopkeeper manages multiple streams
- **Camera ↔ Alert**: Detection events logged per camera
- **Alert ↔ Video Clip**: Event metadata maps to physical storage

---

## 4. Class Diagram

### 4.1 Key Classes

| Class | Key Attributes | Key Methods |
|-------|---------------|-------------|
| **UserAccount** | `userID`, `email`, `passwordHash` | `register()` (FR-1.1), `login()` (FR-1.2), `updateProfile()` (FR-1.4), `manageNotificationPrefs()` (FR-1.6) |
| **CameraManager** | `cameraID`, `streamURL`, `locationLabel` | `addCamera()` (FR-2.1), `editMetadata()` (FR-2.2), `removeCamera()` (FR-2.3), `validateConnection()` (FR-2.4) |
| **AI_Engine** | `modelVersion`, `confidenceThresholds` | `analyzeFeed(stream)` (FR-3.2), `detectSuspiciousActivity()` (FR-3.3), `updateModel(file)` (FR-6.3) |
| **Storage_Manager** | `storageLimit`, `currentUsage` | `recordEventClip()` (FR-5.1), `deleteClip()` (FR-5.3), `manageStorage()` (FR-5.5) |
| **Alert_System** | `alertID`, `eventType`, `clipURL` | `generateAlert()` (FR-4.1), `viewClip()` (FR-4.3), `forwardAlert(recipient)` (FR-4.4) |
| **Mobile_App_UI** | `currentCameraStream`, `notificationStatus` | `displayLiveFeed()` (FR-3.1), `displayAlert()`, `searchEvents(filter)` (FR-9.1) |

### 4.2 Key Class Relationships

- **Identity → Services**: `UserAccount` authenticates via `SessionManager` → receives JWT → required for all secure interactions
- **User → Hardware**: `UserAccount` → 1-to-many → `CameraManager` (centralized control over multiple devices)
- **AI Orchestration**: `AI_Engine` depends on `CameraManager` for RTSP streams → triggers `Storage_Manager` for recording + `BackendAPI_Gateway` for metadata logging
- **Hybrid Storage Flow**: `Storage_Manager` mirrors threat-detected clips from local buffer → cloud storage → provides `cloudURL` to `BackendAPI_Gateway`
- **Alert Dispatch**: `BackendAPI_Gateway` → `NotificationDispatcher` → Firebase (FCM) → `Mobile_App_UI`
- **Administrative Control**: `AdministratorAccount` → `AI_Engine` (push model updates, monitor metrics)

---

## 5. Database Design (ERD)

### 5.1 Database Schema

| Table | Attributes | Relationships |
|-------|-----------|---------------|
| **Users** | `user_id` (UUID PK), `email` (Unique), `password_hash`, `is_2fa_enabled` | Base identity table |
| **Sessions** | `session_id` (PK), `user_id` (FK), `jwt_token`, `expires_at` | FK → Users |
| **Cameras** | `camera_id` (PK), `user_id` (FK), `rtsp_url`, `status` | FK → Users (ON DELETE CASCADE) |
| **Alerts** | `alert_id` (PK), `camera_id` (FK), `event_type`, `is_threat` | FK → Cameras |
| **Video_Clips** | `clip_id` (PK), `alert_id` (FK), `local_file_path`, `cloud_url` | FK → Alerts |
| **Audit_Logs** | `log_id` (PK), `user_id` (FK, Nullable), `activity_type` | FK → Users |

### 5.2 Key Design Decisions

1. **Advanced Security**: Passwords stored as salted/hashed values. Sessions table validates all API requests against active JWT.

2. **Two-Factor Authentication (2FA)**: Users table includes `is_2fa_enabled` flag for mandatory secondary security during login.

3. **Hybrid Storage**: Video_Clips table maintains both:
   - `local_file_path` — high-volume motion data on local HDD
   - `cloud_url` — threat-specific evidence mirrored to Backblaze B2
   - *Optimizes cloud costs while ensuring evidence availability*

4. **Governance & Auditability**: Audit_Logs and AI_Models tables provide transparent records of all system modifications — admin actions and user actions fully traceable.

---

## 6. Sequence Diagrams

### 6.1 Threat Detection & Alert Flow

```
IP Camera ──RTSP──→ AI Engine
                      │
                      ├── Process frames (YOLO/CNN)
                      ├── Detect motion → trigger Storage Manager (local recording buffer)
                      ├── Calculate Confidence Score
                      │
                      ├── [Score > Threshold] → "Threat Detected"
                      │
                      ▼
               Storage Manager
                      │
                      ├── Buffer clip to Local HDD (FR-5.1)
                      ├── Mirror threat clip → Backblaze B2 (cloud)
                      │
                      ▼
              Backend Gateway
                      │
                      ├── Log metadata to database
                      ├── Push notification via FCM (FR-4.1)
                      │
                      ▼
               Mobile App
                      │
                      ├── Validate JWT session (FR-7.2)
                      ├── Request evidence link
                      └── Stream video for secure playback
```

### 6.2 Two-Factor Authentication (2FA) & Session Flow

```
User ──→ Mobile App ──→ Backend Gateway ──→ Database
  │          │                │                 │
  │   Enter credentials      │    Validate      │
  │          │──────────────→│───────────────→│
  │          │               │   2FA required   │
  │          │               │──→ External API   │
  │          │               │   (Send 6-digit   │
  │          │               │    OTP via email)  │
  │          │               │                   │
  │  Enter OTP code          │                   │
  │──────→│──────────────→│                   │
  │          │               │  Validate OTP      │
  │          │               │──→ Session Manager  │
  │          │               │   Generate JWT      │
  │          │               │   Store session      │
  │     JWT token received   │                     │
  │←──────│←──────────────│                     │
```

### 6.3 Administrator AI Model Update Flow

```
Administrator ──→ Admin Dashboard ──→ Backend Gateway ──→ AI Engine
      │                 │                   │                │
      │  Login + view   │   Request update  │   Pull latest  │
      │  system health  │──────────────→│   weights file │
      │                 │                   │   (.pt/.weights)│
      │                 │                   │                │
      │                 │                   │  Run validation │
      │                 │                   │  tests          │
      │                 │                   │←───────────│
      │                 │                   │                │
      │                 │  Update metadata  │                │
      │                 │  (Version, Accuracy)               │
      │                 │  + Create Audit Log entry          │
```

---

## 7. System Interface Design

### 7.1 Mobile Application Screens

| Screen | Description | Requirements Met |
|--------|-------------|-----------------|
| **Login / 2FA** | Validates email/password + 6-digit OTP verification | FR-1.2, FR-7.4, NFR-2.1 |
| **Dashboard (Live Monitoring)** | Tiled multi-camera live feeds with low-latency streaming + "Alerts" badge | FR-3.1, NFR-10.3, NFR-1.2 |
| **Alert History & Review** | Chronological, filterable log of detected events (type, timestamp, camera) | FR-5.4, FR-4.2 |
| **Event Detail View** | Plays recorded video clip + "Forward Alert" and "Delete Clip" buttons | FR-4.3, FR-4.4, FR-5.3 |
| **Camera Management** | Register new cameras via RTSP URL + manage existing devices | FR-2.1, FR-2.3 |
| **Alert Recipient Management** | Manage up to 5 trusted contacts for automated notifications | FR-4.4 |

### 7.2 Notification Interface

| Element | Description | Requirements Met |
|---------|-------------|-----------------|
| **Push Notification** | Titled "Smart Lens Security Alert!" with Event Type, Location, Timestamp | NFR-1.1, NFR-5.2 |
| **Notification Action** | Tapping bypasses dashboard → goes directly to Event Detail View | NFR-5.2 |

### 7.3 Interface Design Flow Logic

- **Session Validation**: On app launch → `CheckSession` state → verify JWT → if valid → Dashboard
- **2FA Gateway**: No valid session → `PrimaryAuth` → mandatory `Verify2FA` → no access until OTP validated
- **Contextual Redirection**: Tapping notification → direct to `EventDetail` (bypasses navigation)
- **Forwarding Action**: In `EventDetail` → `ForwardAlert` state → Notification Dispatcher → send evidence to contacts

---

## 8. Test Cases

| Test ID | Requirement | Scenario | Expected Result |
|---------|-------------|----------|-----------------|
| TC-NFR-1.1 | FR-4.1, NFR-1.1 | Measure time from hazard simulation to push notification | Received within **30 seconds** |
| TC-NFR-4.1 | FR-6.5, NFR-4.1 | Monitor normal activity for 2 hours | **Zero** false alerts |
| TC-FR-5.1 | FR-5.1 | Run camera on static, empty scene for 1 hour | **No clips** saved |
| TC-FR-7.4 | FR-7.4, NFR-2.2 | Inspect API calls for encryption | All traffic uses **HTTPS** |
| TC-FR-3.3.1 | FR-3.3.1 | Simulate a theft and verify AI classification | **Correct tag** + high confidence |
| TC-FR-2.4 | FR-2.4 | Attempt to add an invalid RTSP URL | **Error displayed**; no DB record |

---

## 9. Related Documents

- [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md) — High-level project summary
- [SRS.md](SRS.md) — Full Software Requirements Specification
- [DATASET.md](DATASET.md) — Dataset & AI Model Training Guide
