# 📋 Software Requirements Specification (SRS)

> **Smart Lens CCTV Surveillance System**

---

## 1. Introduction

### 1.1 Purpose of Document

This Software Requirement Specification (SRS) document provides a detailed description of the "Smart Lens CCTV Surveillance System". It defines the system's scope, functionalities, and constraints — outlining functional and non-functional requirements, external interfaces, and performance benchmarks that guide design, development, and testing.

### 1.2 Intended Audience

- **Development Team** — Ali Raza Memon, Aadil Shah, Waseem Mazari
- **Supervisor** — Ma'am Faryal Shamsi
- **Evaluation Committee**

### 1.3 Abbreviations & Acronyms

| Acronym | Definition |
|---------|-----------|
| AI | Artificial Intelligence |
| API | Application Programming Interface |
| CCTV | Closed-Circuit Television |
| CNN | Convolutional Neural Network |
| FCM | Firebase Cloud Messaging |
| FR | Functional Requirement |
| HTTPS | Hypertext Transfer Protocol Secure |
| JWT | JSON Web Token |
| NFR | Non-Functional Requirement |
| PSCA | Punjab Safe Cities Authority |
| RTSP | Real-Time Streaming Protocol |
| SSCA | Sindh Safe Cities Authority |
| UI | User Interface |
| 2FA | Two-Factor Authentication |

---

## 2. Overall System Description

### 2.1 Project Background

Small and medium-scale shops in Pakistan face severe security threats — theft, looting, fire, and violence. Traditional CCTV systems are passive, costly, generate massive unfiltered footage, and rely on error-prone human monitoring. Government projects (PSCA, SSCA) don't serve small retailers. Smart Lens fills this gap with affordable, intelligent, proactive surveillance.

### 2.2 Project Scope

**In Scope:**
- AI-Driven Continuous Threat Detection (theft, violence, looting, fire)
- Motion-Based Video Recording (save only suspicious events)
- Multi-Camera Integration
- Camera Management (add cameras with location metadata)
- Instant Mobile Notifications (event type, camera location, video clip)
- Third-Party Alert Sharing (friends, family, law enforcement)

**Out of Scope:**
- Facial Recognition / Biometric Tracking
- Audio Surveillance
- POS Integration

### 2.3 Operating Environment

**Hardware:**
- IP-based CCTV cameras
- Cloud-based GPU server for Deep Learning inference
- User smartphones (Android) with stable internet

**Software:**
- AI Model: Python with OpenCV, TensorFlow/PyTorch
- Backend Server: FastAPI (Python)
- Mobile Application: Flutter
- Database: Supabase (PostgreSQL)

### 2.4 System Constraints

| Constraint | Description |
|-----------|-------------|
| **Cost** | Must be affordable for small-scale businesses |
| **Privacy** | Must avoid facial recognition; adhere to data protection regulations |
| **Accuracy** | Must minimize false positives to prevent alert fatigue |
| **Environment** | Must handle poor lighting, occlusions, crowded scenes |
| **Connectivity** | Depends on stable internet and electric power |
| **User Expertise** | Target users (shopkeepers) have limited technical expertise |
| **Security** | Must be protected against hacking and data breaches |
| **Dataset** | Accuracy constrained by quality of available training datasets |
| **Hardware** | Requires GPU acceleration; no GPU = degraded performance |

### 2.5 Assumptions & Dependencies

**Assumptions:**
- Shopkeepers have stable internet for cameras and server
- Shopkeepers own an Android smartphone
- Standard IP cameras with accessible RTSP stream

**Dependencies:**
- Publicly available datasets (Kaggle, Roboflow)
- Open-source libraries (Python, OpenCV, TensorFlow, Flutter)
- GPU availability for processing camera feeds

---

## 3. External Interface Requirements

### 3.1 Hardware Interfaces

| Interface | Protocol | Description |
|-----------|----------|-------------|
| IP Cameras | RTSP | Live video stream capture |
| Mobile Devices | Android OS | Mobile app UI, alerts, video playback |
| Server/GPU | Cloud VM | Parallel processing of multiple camera streams |

### 3.2 Software Interfaces

| Interface | Description |
|-----------|-------------|
| AI Models | TensorFlow/PyTorch model files for frame processing |
| Database | Supabase (PostgreSQL) for users, cameras, alerts |
| Mobile OS | Flutter app interfaces with Android native capabilities |

### 3.3 Communication Interfaces

| Protocol | Usage |
|----------|-------|
| **RTSP** | Pull live video from IP cameras |
| **HTTPS** | All client-server communication (encrypted) |
| **FCM** | Firebase Cloud Messaging for push notifications |
| **SMTP** | Email forwarding of evidence links |

---

## 4. Functional Requirements

### FR-1: User Management

| ID | Requirement |
|----|-------------|
| FR-1.1 | Register account using email, phone number, and password |
| FR-1.2 | Secure login mechanism using authentication protocols |
| FR-1.3 | Password reset/recovery through email verification |
| FR-1.4 | Edit profile (shop name, contact details, camera location tags) |
| FR-1.5 | Delete account securely |
| FR-1.6 | Manage notification and alert preferences |
| FR-1.7 | Two-Factor Authentication (2FA) with 6-digit OTP via email (5-min expiry) |
| FR-1.8 | Secure session management using JWT (creation, expiration, invalidation) |

### FR-2: Camera Management

| ID | Requirement |
|----|-------------|
| FR-2.1 | Add new IP cameras by RTSP link + location label |
| FR-2.2 | Edit camera details (name, location) |
| FR-2.3 | Remove existing cameras |
| FR-2.4 | Validate camera connectivity before saving |
| FR-2.5 | Store camera metadata (ID, Location, Stream URL) in database |

### FR-3: Continuous Surveillance & Monitoring

| ID | Requirement |
|----|-------------|
| FR-3.1 | Dashboard for viewing live feeds from multiple cameras |
| FR-3.2 | Continuously analyse live video using AI model |
| FR-3.3 | Detect and classify specific events: |
| FR-3.3.1 | → Theft (suspicious movement near sensitive zones) |
| FR-3.3.2 | → Violent activities and crowd disturbances |
| FR-3.3.3 | → Fire and smoke hazards |
| FR-3.3.4 | → Overlay bounding boxes on detected objects/events |
| FR-3.3.5 | → Record only video clips containing motion events |

### FR-4: Alert & Notification System

| ID | Requirement |
|----|-------------|
| FR-4.1 | Send instant mobile alerts when suspicious activity detected |
| FR-4.2 | Include event type, camera location, and timestamp in each alert |
| FR-4.3 | Attach short video clip of detected event |
| FR-4.4 | Allow forwarding alert to external parties (community group, law enforcement) |

### FR-5: Event Recording & Storage Management

| ID | Requirement |
|----|-------------|
| FR-5.1 | Record and save only motion-based events to optimize storage |
| FR-5.2 | Store video clips with metadata (event type, camera ID, timestamp) |
| FR-5.3 | Allow users to view, download, and delete stored clips |
| FR-5.4 | Maintain log of all detected events for audit/review |
| FR-5.5 | Auto-delete older clips when storage limit reached |

### FR-6: AI Engine & Processing

| ID | Requirement |
|----|-------------|
| FR-6.1 | Use trained AI model to analyse live CCTV footage |
| FR-6.2 | Process video frames for detection |
| FR-6.3 | Allow retraining and updating AI model with new datasets |
| FR-6.4 | Output detection results with confidence scores |
| FR-6.5 | Minimize false positives by applying confidence thresholds |

### FR-7: Security & Privacy

| ID | Requirement |
|----|-------------|
| FR-7.1 | Encrypt user credentials in storage and transmission |
| FR-7.2 | Restrict access to video feeds/clips to authorized users only |
| FR-7.3 | Log all user activities for audit (login attempts, data deletion) |
| FR-7.4 | Secure communication using HTTPS for all client-server interactions |
| FR-7.5 | Video feeds not shared publicly except when explicitly initiated by user |

### FR-8: System Administration

| ID | Requirement |
|----|-------------|
| FR-8.1 | Admin interface for managing users, cameras, event logs |
| FR-8.2 | Monitor live system status and AI performance metrics |
| FR-8.3 | Remove or block users violating terms of use |
| FR-8.4 | Access reports summarizing system activity and detections |

### FR-9: Search & Filtering

| ID | Requirement |
|----|-------------|
| FR-9.1 | Search recorded events by date, event type, or camera location |
| FR-9.2 | Filter alerts/videos by event severity |
| FR-9.3 | Sorting options (latest first, by camera, etc.) |

### FR-10: System Performance

| ID | Requirement |
|----|-------------|
| FR-10.1 | Instant alert generation upon event detection |
| FR-10.2 | Maintain video quality during continuous analysis |
| FR-10.3 | Simultaneous streaming from multiple cameras without lag |
| FR-10.4 | 95% uptime under stable internet conditions |

---

## 5. Non-Functional Requirements

### NFR-1: Performance

| ID | Requirement |
|----|-------------|
| NFR-1.1 | Push notification within **30 seconds** of event detection |
| NFR-1.2 | Live video stream loads in **under 5 seconds** on stable connection |
| NFR-1.3 | AI engine supports continuous analysis of **minimum 4 cameras** without lag |
| NFR-1.4 | Minimum video quality of **720p** for clear visual evidence |
| NFR-1.5 | Cloud components maintain **99% uptime** under normal conditions |

### NFR-2: Security

| ID | Requirement |
|----|-------------|
| NFR-2.1 | Secure authentication (JWT, OAuth) for mobile app and admin |
| NFR-2.2 | All data transmission encrypted using **HTTPS/SSL** |
| NFR-2.3 | Sensitive data encrypted at rest using **AES-256** |
| NFR-2.4 | Role-based access control (Shopkeeper vs Administrator) |
| NFR-2.5 | All critical activities logged for audit purposes |

### NFR-3: Privacy

| ID | Requirement |
|----|-------------|
| NFR-3.1 | No facial recognition or biometric tracking |
| NFR-3.2 | Video footage treated as private and confidential |
| NFR-3.3 | Adherence to local data protection and privacy regulations |

### NFR-4: Accuracy & Reliability

| ID | Requirement |
|----|-------------|
| NFR-4.1 | Minimize false positives to prevent alert fatigue |
| NFR-4.2 | Configurable confidence thresholds for alerts |
| NFR-4.3 | Safety-critical events (fire) treated as high-priority |
| NFR-4.4 | AI model robust under variable lighting and partial occlusions |
| NFR-4.5 | AI engine designed for **24/7 continuous operation** |

### NFR-5: Usability

| ID | Requirement |
|----|-------------|
| NFR-5.1 | Simple, intuitive interface for users with limited technical expertise |
| NFR-5.2 | Clear, concise, immediately actionable alert notifications |

### NFR-6: Maintainability

| ID | Requirement |
|----|-------------|
| NFR-6.1 | AI model retrainable with new datasets |
| NFR-6.2 | Administrator dashboard for system health and AI metrics |

### NFR-7: Documentation

| ID | Requirement |
|----|-------------|
| NFR-7.1 | User Manual and video guide for shopkeepers |
| NFR-7.2 | Comprehensive Technical Documentation for developers |

---

## 6. Use Cases Summary

| Use Case ID | Name | Actors |
|-------------|------|--------|
| UC-UM-001 | User Registration | Shopkeeper |
| UC-UM-002 | Secure User Login | Shopkeeper, System |
| UC-CM-001 | Add New Camera | Shopkeeper |
| UC-CM-002 | Remove Registered Camera | Shopkeeper |
| UC-MM-001 | View Multi-Camera Live Feed | Shopkeeper, IP Cameras |
| UC-SM-002 | View Stored Event Clips | Shopkeeper |
| UC-SM-003 | Manage & Delete Stored Clips | Shopkeeper |
| UC-AR-001 | Forward Alert | Shopkeeper, Third Party |
| UC-AR-002 | Receive Suspicious Activity Alert | AI Model, Mobile App, Shopkeeper |
| UC-SA-001 | Admin Update AI Model | Administrator |

### UC-UM-001: User Registration

| Field | Detail |
|-------|--------|
| **Actors** | Shopkeeper |
| **Precondition** | No existing account; has access to mobile app |
| **Main Flow** | 1. Select "Register" → 2. Fill email, phone, password → 3. System validates → 4. Account created, redirect to login |
| **Alternate** | Email already exists → error; Weak password → prompt for correction |
| **Postcondition** | Account created; user redirected to login |

### UC-UM-002: Secure User Login

| Field | Detail |
|-------|--------|
| **Actors** | Shopkeeper |
| **Precondition** | Has registered account |
| **Main Flow** | 1. Enter credentials → 2. System validates + generates JWT → 3. Redirect to Dashboard |
| **Alternate** | Invalid credentials → error + logged; Forgot password → recovery flow |
| **Postcondition** | User securely logged in; valid JWT issued |

### UC-CM-001: Add New Camera

| Field | Detail |
|-------|--------|
| **Actors** | Shopkeeper |
| **Precondition** | Logged in; IP camera on network with valid RTSP URL |
| **Main Flow** | 1. Select "Add Camera" → 2. Enter RTSP URL, name, location → 3. System validates RTSP → 4. Camera registered |
| **Alternate** | Invalid RTSP → connection error; Empty fields → prompt |
| **Postcondition** | Camera registered and streaming to AI Engine |

### UC-AR-002: Receive Suspicious Activity Alert

| Field | Detail |
|-------|--------|
| **Actors** | AI Model, Mobile App, Shopkeeper, Law Enforcement |
| **Precondition** | User logged in; push notifications enabled; AI analyzing feed |
| **Main Flow** | 1. AI detects suspicious activity with high confidence → 2. Video clip recorded + metadata stored → 3. Push notification via FCM → 4. Shopkeeper + law enforcement receive alert → 5. Tap notification → view clip in app |
| **Alternate** | Notifications disabled → event logged only (no push); No suspicious activity → continue surveillance |
| **Postcondition** | Shopkeeper notified; video clip + metadata available for review |

---

## 7. Test Cases

| Test ID | Requirement | Scenario | Expected Result |
|---------|-------------|----------|-----------------|
| TC-NFR-1.1 | FR-4.1, NFR-1.1 | Measure time from hazard to push notification | Received within 30 seconds |
| TC-NFR-4.1 | FR-6.5, NFR-4.1 | Monitor normal activity for 2 hours | Zero false alerts |
| TC-FR-5.1 | FR-5.1 | Run camera on static empty scene for 1 hour | No clips saved |
| TC-FR-7.4 | FR-7.4, NFR-2.2 | Inspect API calls for encryption | All traffic uses HTTPS |
| TC-FR-3.3.1 | FR-3.3.1 | Simulate a theft, verify AI classification | Correct tag + high confidence |
| TC-FR-2.4 | FR-2.4 | Attempt to add invalid RTSP URL | Error displayed; no DB record |

---

## 8. Appendices

### Appendix A: Glossary

| Term | Definition |
|------|-----------|
| **Smart Lens** | The AI-powered surveillance system being developed |
| **Threat Detection** | AI engine analyzing video to identify suspicious activities |
| **Motion-Based Recording** | Only recording video when suspicious events detected |
| **AI Engine** | Software component running AI models on video feeds |
| **Alert Fatigue** | User desensitization from too many false positive alerts |
| **Occlusion** | Object/person partially or fully blocked from camera view |

### Appendix B: Development Tools

| Tool | Usage |
|------|-------|
| Python | AI Engine programming |
| TensorFlow / PyTorch | AI Frameworks |
| OpenCV | Video Processing |
| FastAPI | Backend Server |
| Flutter | Mobile Application |
| Supabase (PostgreSQL) | Database |
| Git & GitHub | Version Control |
