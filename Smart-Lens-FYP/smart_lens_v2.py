"""
Smart Lens v2 — Enhanced Detection Pipeline
=============================================
Motion-gated + Temporal-consistency threat detection.

Key improvements over v1:
  1. Motion Gate       — Skip YOLO on static/idle frames → zero FP on empty scenes
  2. Temporal Filter   — Require N/M recent frames to confirm a threat → kills flickers
  3. Per-class Conf    — Critical classes (Fire/Gun) lower threshold; noisy ones higher
  4. Motion-zone ROI   — Feed only active regions to YOLO → faster + fewer FP
  5. Cooldown Timer    — Suppress duplicate alerts within a time window

Usage:
    python smart_lens_v2.py                                    # Webcam
    python smart_lens_v2.py --source video.mp4                 # Video file
    python smart_lens_v2.py --source rtsp://ip/stream          # RTSP camera
    python smart_lens_v2.py --source video.mp4 --save --log    # Save + log
    python smart_lens_v2.py --mode strict                      # Strict = fewer FP
    python smart_lens_v2.py --mode sensitive                   # Sensitive = fewer misses

Keyboard Controls:
    q   — Quit
    p   — Pause / Resume
    +/- — Adjust base confidence threshold
    m   — Toggle motion overlay visualization
    s   — Screenshot current frame
    d   — Toggle debug info panel
"""

import argparse
import sys
import os
import time
from pathlib import Path
from datetime import datetime
from collections import defaultdict, deque

from ultralytics import YOLO
import cv2
import numpy as np

# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "trained_models",
                          "smart_lens_v1_20260208_0043", "best.pt")

CLASSES = {0: "Fighting", 1: "Fire", 2: "Gun", 3: "Knife"}

CLASS_COLORS = {
    "Fighting": (0, 0, 255),      # Red
    "Fire":     (0, 140, 255),    # Orange
    "Gun":      (0, 255, 255),    # Yellow
    "Knife":    (255, 0, 255),    # Magenta
}

THREAT_SEVERITY = {
    "Fire":     "CRITICAL",
    "Gun":      "CRITICAL",
    "Knife":    "HIGH",
    "Fighting": "MEDIUM",
}

# ─── Per-class confidence thresholds ─────────────────────────────────────────
# Critical threats need lower threshold (better recall), noisy classes higher
CLASS_CONF_THRESHOLDS = {
    "balanced": {
        "Fighting": 0.55,
        "Fire":     0.45,
        "Gun":      0.45,
        "Knife":    0.50,
    },
    "strict": {       # Minimize false positives
        "Fighting": 0.65,
        "Fire":     0.55,
        "Gun":      0.55,
        "Knife":    0.60,
    },
    "sensitive": {    # Minimize missed threats
        "Fighting": 0.40,
        "Fire":     0.30,
        "Gun":      0.35,
        "Knife":    0.40,
    },
}

# ─── Motion detection parameters ─────────────────────────────────────────────
MOTION_HISTORY       = 500      # Background subtractor history (frames)
MOTION_THRESHOLD     = 50       # Variance threshold for background model
MOTION_MIN_AREA      = 1500     # Min contour area to count as motion (pixels²)
MOTION_SCORE_GATE    = 0.005    # Min fraction of frame with motion to trigger YOLO
MOTION_DILATE_KERNEL = 15       # Dilation kernel to merge nearby motion zones

# ─── Temporal consistency parameters ─────────────────────────────────────────
TEMPORAL_WINDOW      = 8        # Look-back window (frames)
TEMPORAL_MIN_HITS    = 3        # Min detections in window to confirm threat
ALERT_COOLDOWN_SEC   = 5.0     # Suppress duplicate alerts within this window


# ═══════════════════════════════════════════════════════════════════════════════
#  MOTION DETECTOR
# ═══════════════════════════════════════════════════════════════════════════════
class MotionDetector:
    """Background-subtraction based motion detector.

    Purpose: Gate the YOLO model — if no significant motion is detected,
    skip inference entirely. This eliminates ALL false positives on static scenes.
    """

    def __init__(self, history=MOTION_HISTORY, threshold=MOTION_THRESHOLD,
                 min_area=MOTION_MIN_AREA, gate_score=MOTION_SCORE_GATE):
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history, varThreshold=threshold, detectShadows=True
        )
        self.min_area = min_area
        self.gate_score = gate_score
        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (MOTION_DILATE_KERNEL, MOTION_DILATE_KERNEL)
        )
        self.warmup_frames = 30  # Let BG model stabilize
        self.frame_count = 0

    def detect(self, frame):
        """Analyze frame for motion.

        Returns:
            motion_score (float):  Fraction of frame pixels with motion [0-1]
            motion_mask (ndarray): Binary mask of motion regions
            motion_zones (list):   List of (x,y,w,h) bounding rects of motion
            has_motion (bool):     Whether motion exceeds the gate threshold
        """
        self.frame_count += 1

        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(frame)

        # Remove shadows (shadow pixels = 127 in MOG2)
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

        # Morphological cleanup: remove noise, merge nearby regions
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN,
                                   cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        fg_mask = cv2.dilate(fg_mask, self.kernel, iterations=2)

        # Compute motion score
        total_pixels = frame.shape[0] * frame.shape[1]
        motion_pixels = cv2.countNonZero(fg_mask)
        motion_score = motion_pixels / total_pixels

        # Find motion zones (contours)
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        motion_zones = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= self.min_area:
                motion_zones.append(cv2.boundingRect(cnt))

        # During warmup, always report motion so BG model stabilizes
        if self.frame_count <= self.warmup_frames:
            has_motion = True
        else:
            has_motion = motion_score >= self.gate_score and len(motion_zones) > 0

        return motion_score, fg_mask, motion_zones, has_motion

    def get_combined_roi(self, motion_zones, frame_shape, padding=50):
        """Merge motion zones into a single expanded ROI for YOLO.

        Returns (x1, y1, x2, y2) or None if no zones.
        """
        if not motion_zones:
            return None

        h, w = frame_shape[:2]
        x_min = min(z[0] for z in motion_zones)
        y_min = min(z[1] for z in motion_zones)
        x_max = max(z[0] + z[2] for z in motion_zones)
        y_max = max(z[1] + z[3] for z in motion_zones)

        # Add padding
        x1 = max(0, x_min - padding)
        y1 = max(0, y_min - padding)
        x2 = min(w, x_max + padding)
        y2 = min(h, y_max + padding)

        # Don't use ROI if it covers >70% of frame (overhead not worth it)
        roi_area = (x2 - x1) * (y2 - y1)
        if roi_area > 0.7 * w * h:
            return None

        return (x1, y1, x2, y2)


# ═══════════════════════════════════════════════════════════════════════════════
#  TEMPORAL CONSISTENCY FILTER
# ═══════════════════════════════════════════════════════════════════════════════
class TemporalFilter:
    """Require multiple detections across recent frames to confirm a threat.

    This eliminates single-frame false positives (flickers, lighting changes,
    hand gestures misclassified as weapons, etc.).
    """

    def __init__(self, window=TEMPORAL_WINDOW, min_hits=TEMPORAL_MIN_HITS,
                 cooldown=ALERT_COOLDOWN_SEC):
        self.window = window
        self.min_hits = min_hits
        self.cooldown = cooldown

        # Per-class: deque of recent frame numbers where class was detected
        self.history = defaultdict(lambda: deque(maxlen=window))

        # Per-class: last confirmed alert timestamp
        self.last_alert_time = defaultdict(float)

        # Current frame counter
        self.current_frame = 0

    def update(self, frame_num, detections):
        """Process new detections and return only temporally confirmed ones.

        Args:
            frame_num: Current frame number
            detections: List of detection dicts with 'class', 'confidence', etc.

        Returns:
            confirmed: List of detections that passed temporal consistency
            pending:   List of detections still accumulating evidence
        """
        self.current_frame = frame_num

        # Record which classes were seen this frame
        seen_classes = set()
        for det in detections:
            cls_name = det["class"]
            seen_classes.add(cls_name)
            self.history[cls_name].append(frame_num)

        confirmed = []
        pending = []

        for det in detections:
            cls_name = det["class"]

            # Count how many of the last N frames had this class
            recent = self.history[cls_name]
            # Only count frames within the window
            hits = sum(1 for f in recent if frame_num - f < self.window)

            det["temporal_hits"] = hits
            det["temporal_required"] = self.min_hits

            if hits >= self.min_hits:
                # Check cooldown
                now = time.time()
                if now - self.last_alert_time[cls_name] >= self.cooldown:
                    self.last_alert_time[cls_name] = now
                    det["alert_status"] = "CONFIRMED"
                    confirmed.append(det)
                else:
                    det["alert_status"] = "SUPPRESSED"
                    confirmed.append(det)  # Still draw, but don't re-alert
            else:
                det["alert_status"] = "PENDING"
                pending.append(det)

        return confirmed, pending

    def get_class_status(self):
        """Return dict of {class: hit_count} for debug display."""
        status = {}
        for cls_name, history in self.history.items():
            hits = sum(1 for f in history if self.current_frame - f < self.window)
            status[cls_name] = hits
        return status


# ═══════════════════════════════════════════════════════════════════════════════
#  DETECTION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════
class SmartLensDetector:
    """Complete detection pipeline: Motion Gate → YOLO → Per-class Filter → Temporal."""

    def __init__(self, model_path, mode="balanced", img_size=640, iou=0.45):
        # Load YOLO model
        if not os.path.exists(model_path):
            print(f"[ERROR] Model not found: {model_path}")
            sys.exit(1)

        print(f"[INFO] Loading model: {model_path}")
        self.model = YOLO(model_path)
        print(f"[INFO] Model loaded | Classes: {list(CLASSES.values())}")

        # Config
        self.mode = mode
        self.class_thresholds = CLASS_CONF_THRESHOLDS[mode].copy()
        self.base_conf = min(self.class_thresholds.values())  # Feed YOLO lowest
        self.img_size = img_size
        self.iou = iou

        # Sub-components
        self.motion = MotionDetector()
        self.temporal = TemporalFilter()

        # Stats
        self.stats = {
            "frames_total": 0,
            "frames_with_motion": 0,
            "frames_yolo_ran": 0,
            "frames_skipped": 0,
            "raw_detections": 0,
            "class_filtered": 0,
            "confirmed_alerts": 0,
            "pending_filtered": 0,
        }

        print(f"[INFO] Mode: {mode}")
        print(f"[INFO] Per-class thresholds: {self.class_thresholds}")

    def process_frame(self, frame, frame_num):
        """Full pipeline: motion → YOLO → class filter → temporal.

        Returns:
            confirmed_dets:  Temporally confirmed detections (draw + alert)
            pending_dets:    Pending detections (draw dimmed, no alert)
            motion_score:    Motion score [0-1]
            motion_mask:     Binary motion mask
            motion_zones:    List of motion zone rects
            yolo_ran:        Whether YOLO was invoked this frame
        """
        self.stats["frames_total"] += 1
        h, w = frame.shape[:2]

        # ── Step 1: Motion Gate ──────────────────────────────────────────
        motion_score, motion_mask, motion_zones, has_motion = \
            self.motion.detect(frame)

        if has_motion:
            self.stats["frames_with_motion"] += 1

        if not has_motion:
            self.stats["frames_skipped"] += 1
            return [], [], motion_score, motion_mask, motion_zones, False

        # ── Step 2: YOLO Inference ───────────────────────────────────────
        self.stats["frames_yolo_ran"] += 1

        # Optional: crop to motion ROI for speed (if ROI is small enough)
        roi = self.motion.get_combined_roi(motion_zones, frame.shape)
        offset_x, offset_y = 0, 0

        if roi is not None:
            rx1, ry1, rx2, ry2 = roi
            roi_frame = frame[ry1:ry2, rx1:rx2]
            offset_x, offset_y = rx1, ry1
            # Only use ROI if it's reasonably sized (min 200px each dimension)
            if roi_frame.shape[0] < 200 or roi_frame.shape[1] < 200:
                roi_frame = frame
                offset_x, offset_y = 0, 0
                roi = None
        else:
            roi_frame = frame

        results = self.model.predict(
            roi_frame, conf=self.base_conf, iou=self.iou,
            imgsz=self.img_size, verbose=False
        )

        # ── Step 3: Per-class Confidence Filter ──────────────────────────
        raw_detections = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = CLASSES.get(cls_id, f"Class_{cls_id}")

                self.stats["raw_detections"] += 1

                # Apply per-class threshold
                cls_threshold = self.class_thresholds.get(cls_name, 0.5)
                if conf < cls_threshold:
                    self.stats["class_filtered"] += 1
                    continue

                # Get bbox (with ROI offset if used)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x1 += offset_x
                y1 += offset_y
                x2 += offset_x
                y2 += offset_y

                # Clamp to frame
                x1 = max(0, min(w, x1))
                y1 = max(0, min(h, y1))
                x2 = max(0, min(w, x2))
                y2 = max(0, min(h, y2))

                # Check if detection overlaps with motion zone (extra validation)
                det_center = ((x1 + x2) // 2, (y1 + y2) // 2)
                in_motion = self._point_in_zones(det_center, motion_zones)

                # For Fire class, skip motion-zone check (fire can be static-ish)
                if cls_name != "Fire" and not in_motion and roi is None:
                    self.stats["class_filtered"] += 1
                    continue

                raw_detections.append({
                    "class": cls_name,
                    "confidence": conf,
                    "severity": THREAT_SEVERITY.get(cls_name, "LOW"),
                    "bbox": (x1, y1, x2, y2),
                })

        # ── Step 4: Temporal Consistency ─────────────────────────────────
        confirmed, pending = self.temporal.update(frame_num, raw_detections)
        self.stats["confirmed_alerts"] += len(
            [d for d in confirmed if d["alert_status"] == "CONFIRMED"])
        self.stats["pending_filtered"] += len(pending)

        return confirmed, pending, motion_score, motion_mask, motion_zones, True

    def _point_in_zones(self, point, zones):
        """Check if a point falls within any motion zone."""
        px, py = point
        for (zx, zy, zw, zh) in zones:
            if zx <= px <= zx + zw and zy <= py <= zy + zh:
                return True
        return False

    def adjust_thresholds(self, delta):
        """Adjust all per-class thresholds by delta."""
        for cls in self.class_thresholds:
            self.class_thresholds[cls] = max(0.1, min(0.95,
                self.class_thresholds[cls] + delta))
        self.base_conf = min(self.class_thresholds.values())

    def get_stats_summary(self):
        s = self.stats
        total = max(s["frames_total"], 1)
        return {
            "skip_rate": f'{100 * s["frames_skipped"] / total:.1f}%',
            "yolo_rate": f'{100 * s["frames_yolo_ran"] / total:.1f}%',
            "raw_dets": s["raw_detections"],
            "class_filtered": s["class_filtered"],
            "confirmed": s["confirmed_alerts"],
            "pending_dropped": s["pending_filtered"],
        }


# ═══════════════════════════════════════════════════════════════════════════════
#  UI RENDERING
# ═══════════════════════════════════════════════════════════════════════════════
def draw_detections_v2(frame, confirmed, pending, show_pending=True):
    """Draw confirmed detections (solid) and pending ones (dashed/dim)."""

    # Draw pending first (underneath)
    if show_pending:
        for det in pending:
            x1, y1, x2, y2 = det["bbox"]
            color = CLASS_COLORS.get(det["class"], (128, 128, 128))
            dim_color = tuple(c // 2 for c in color)

            # Dashed rectangle for pending
            _draw_dashed_rect(frame, (x1, y1), (x2, y2), dim_color, 1, 10)

            label = f"? {det['class']} {det['confidence']:.2f} ({det['temporal_hits']}/{det['temporal_required']})"
            cv2.putText(frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, dim_color, 1)

    # Draw confirmed (solid, prominent)
    for det in confirmed:
        x1, y1, x2, y2 = det["bbox"]
        cls_name = det["class"]
        severity = det["severity"]
        color = CLASS_COLORS.get(cls_name, (0, 255, 0))
        thickness = 3 if severity == "CRITICAL" else 2

        # Solid rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        # Label with severity indicator
        sev_icon = {"CRITICAL": "!!", "HIGH": "!", "MEDIUM": ""}.get(severity, "")
        label = f"{sev_icon} {cls_name} {det['confidence']:.2f}"

        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - lh - 10), (x1 + lw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return frame


def _draw_dashed_rect(img, pt1, pt2, color, thickness, dash_len):
    """Draw a dashed rectangle."""
    x1, y1 = pt1
    x2, y2 = pt2
    for edge in [
        ((x1, y1), (x2, y1)),  # top
        ((x2, y1), (x2, y2)),  # right
        ((x2, y2), (x1, y2)),  # bottom
        ((x1, y2), (x1, y1)),  # left
    ]:
        _draw_dashed_line(img, edge[0], edge[1], color, thickness, dash_len)


def _draw_dashed_line(img, pt1, pt2, color, thickness, dash_len):
    """Draw a dashed line between two points."""
    dist = int(np.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1]))
    if dist == 0:
        return
    dx = (pt2[0] - pt1[0]) / dist
    dy = (pt2[1] - pt1[1]) / dist
    for i in range(0, dist, dash_len * 2):
        s = (int(pt1[0] + dx * i), int(pt1[1] + dy * i))
        e_d = min(i + dash_len, dist)
        e = (int(pt1[0] + dx * e_d), int(pt1[1] + dy * e_d))
        cv2.line(img, s, e, color, thickness)


def draw_status_bar(frame, fps, confirmed_count, pending_count,
                    motion_score, yolo_ran, frame_num, mode):
    """Comprehensive status bar."""
    h, w = frame.shape[:2]
    bar_h = 50
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    # Row 1: System status
    motion_text = f"Motion: {motion_score*100:.1f}%"
    yolo_text = "YOLO: ON" if yolo_ran else "YOLO: SKIP"
    yolo_color = (0, 200, 0) if yolo_ran else (100, 100, 100)
    status = f"Smart Lens v2 [{mode}] | FPS: {fps:.0f} | {motion_text} | F:{frame_num}"
    cv2.putText(frame, status, (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 220, 0), 1)
    cv2.putText(frame, yolo_text, (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, yolo_color, 1)

    # Right side: alert status
    if confirmed_count > 0:
        flash = (0, 0, 255) if (frame_num // 4) % 2 == 0 else (0, 80, 200)
        alert = f"THREAT x{confirmed_count}"
        cv2.putText(frame, alert, (w - 180, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, flash, 2)
    elif pending_count > 0:
        cv2.putText(frame, f"Analyzing... ({pending_count})", (w - 200, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
    else:
        cv2.putText(frame, "CLEAR", (w - 80, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 180, 0), 1)

    return frame


def draw_debug_panel(frame, detector, motion_score, motion_zones, yolo_ran):
    """Draw a debug info panel on the right side."""
    h, w = frame.shape[:2]
    panel_w = 250
    overlay = frame.copy()
    cv2.rectangle(overlay, (w - panel_w, 55), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    x0 = w - panel_w + 10
    y = 75
    line_h = 20

    def put(text, color=(200, 200, 200)):
        nonlocal y
        cv2.putText(frame, text, (x0, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, color, 1)
        y += line_h

    put("--- DEBUG PANEL ---", (0, 200, 255))
    put(f"Motion: {motion_score*100:.2f}%")
    put(f"Motion zones: {len(motion_zones)}")
    put(f"YOLO active: {'Yes' if yolo_ran else 'No'}")
    put("")
    put("Thresholds:", (0, 200, 255))
    for cls, thr in detector.class_thresholds.items():
        put(f"  {cls}: {thr:.2f}")
    put("")
    put("Temporal Status:", (0, 200, 255))
    for cls, hits in detector.temporal.get_class_status().items():
        color = (0, 255, 0) if hits >= TEMPORAL_MIN_HITS else (100, 100, 100)
        put(f"  {cls}: {hits}/{TEMPORAL_MIN_HITS}", color)
    put("")
    stats = detector.get_stats_summary()
    put("Pipeline Stats:", (0, 200, 255))
    put(f"  Skip rate: {stats['skip_rate']}")
    put(f"  YOLO rate: {stats['yolo_rate']}")
    put(f"  Raw dets: {stats['raw_dets']}")
    put(f"  Filtered: {stats['class_filtered']}")
    put(f"  Confirmed: {stats['confirmed']}")

    return frame


def draw_motion_overlay(frame, motion_mask, motion_zones):
    """Overlay semi-transparent motion visualization."""
    # Green tint on motion pixels
    green_mask = np.zeros_like(frame)
    green_mask[:, :, 1] = motion_mask  # Green channel
    cv2.addWeighted(green_mask, 0.3, frame, 1.0, 0, frame)

    # Draw motion zone boxes
    for (x, y, mw, mh) in motion_zones:
        cv2.rectangle(frame, (x, y), (x + mw, y + mh), (0, 255, 0), 1)

    return frame


# ═══════════════════════════════════════════════════════════════════════════════
#  LOGGING
# ═══════════════════════════════════════════════════════════════════════════════
def init_log(source_name):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(SCRIPT_DIR, f"detection_log_{source_name}_{ts}.csv")
    with open(path, "w") as f:
        f.write("timestamp,frame,class,confidence,severity,status,"
                "temporal_hits,x1,y1,x2,y2\n")
    return path


def log_detections(path, frame_num, detections):
    with open(path, "a") as f:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        for d in detections:
            x1, y1, x2, y2 = d["bbox"]
            f.write(f"{ts},{frame_num},{d['class']},{d['confidence']:.4f},"
                    f"{d['severity']},{d.get('alert_status','?')},"
                    f"{d.get('temporal_hits', 0)},{x1},{y1},{x2},{y2}\n")


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def parse_args():
    parser = argparse.ArgumentParser(
        description="Smart Lens v2 — Motion-gated threat detection pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Modes:\n"
               "  balanced   — Default. Good balance of precision/recall\n"
               "  strict     — High confidence required. Fewest false positives\n"
               "  sensitive  — Low thresholds. Fewest missed threats\n"
    )
    parser.add_argument("--source", type=str, default="0",
                        help="Video/image/webcam/RTSP source (default: 0)")
    parser.add_argument("--model", type=str, default=MODEL_PATH,
                        help="Path to model file")
    parser.add_argument("--mode", type=str, default="balanced",
                        choices=["balanced", "strict", "sensitive"],
                        help="Detection mode (default: balanced)")
    parser.add_argument("--conf", type=float, default=None,
                        help="Override all class thresholds with single value")
    parser.add_argument("--iou", type=float, default=0.45,
                        help="NMS IoU threshold (default: 0.45)")
    parser.add_argument("--img-size", type=int, default=640,
                        help="Inference image size (default: 640)")
    parser.add_argument("--save", action="store_true",
                        help="Save output video with detections")
    parser.add_argument("--save-path", type=str, default=None,
                        help="Output save path")
    parser.add_argument("--no-display", action="store_true",
                        help="Headless mode (no GUI window)")
    parser.add_argument("--log", action="store_true",
                        help="Save detection log to CSV")
    parser.add_argument("--debug", action="store_true",
                        help="Show debug panel overlay")
    parser.add_argument("--show-motion", action="store_true",
                        help="Show motion detection overlay")
    return parser.parse_args()


def main():
    args = parse_args()

    # Initialize detector
    detector = SmartLensDetector(
        model_path=args.model, mode=args.mode,
        img_size=args.img_size, iou=args.iou
    )

    # Override per-class thresholds if --conf given
    if args.conf is not None:
        for cls in detector.class_thresholds:
            detector.class_thresholds[cls] = args.conf
        detector.base_conf = args.conf
        print(f"[INFO] Override: all class thresholds set to {args.conf}")

    # Source
    source = args.source
    is_webcam = source.isdigit()
    is_image = source.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))
    is_rtsp = source.lower().startswith('rtsp://')
    source_name = Path(str(source)).stem if not is_webcam else f"webcam_{source}"

    if is_webcam:
        source = int(source)

    # ─── Image mode ──────────────────────────────────────────────────────
    if is_image:
        print(f"[INFO] Image mode: {source}")
        frame = cv2.imread(source)
        if frame is None:
            print(f"[ERROR] Cannot read image: {source}")
            sys.exit(1)

        # For images, skip motion gate, run YOLO directly
        results = detector.model.predict(frame, conf=detector.base_conf,
                                         iou=args.iou, imgsz=args.img_size,
                                         verbose=False)
        detections = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = CLASSES.get(cls_id, f"Class_{cls_id}")
                thr = detector.class_thresholds.get(cls_name, 0.5)
                if conf < thr:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append({
                    "class": cls_name, "confidence": conf,
                    "severity": THREAT_SEVERITY.get(cls_name, "LOW"),
                    "bbox": (x1, y1, x2, y2),
                    "alert_status": "CONFIRMED",
                    "temporal_hits": 1, "temporal_required": 1,
                })

        frame = draw_detections_v2(frame, detections, [])
        print(f"\n{'='*55}")
        print(f"  {len(detections)} threat(s) detected [{args.mode} mode]")
        print(f"{'='*55}")
        for d in detections:
            print(f"  [{d['severity']}] {d['class']}: {d['confidence']:.2f}")
        if not detections:
            print(f"  No threats (thresholds: {detector.class_thresholds})")
        print(f"{'='*55}")

        if args.save:
            sp = args.save_path or f"v2_output_{Path(args.source).stem}.jpg"
            cv2.imwrite(sp, frame)
            print(f"[INFO] Saved: {sp}")
        if not args.no_display:
            cv2.imshow("Smart Lens v2", frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return

    # ─── Video / Webcam / RTSP ───────────────────────────────────────────
    print(f"[INFO] Opening: {'webcam ' + str(source) if is_webcam else source}")
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open: {args.source}")
        sys.exit(1)

    fps_video = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"[INFO] {width}x{height} @ {fps_video:.0f}fps"
          f" | Frames: {total_frames if total_frames > 0 else 'live'}")
    print(f"[INFO] Mode: {args.mode} | Thresholds: {detector.class_thresholds}")
    print(f"[INFO] Controls: q=quit p=pause +/-=conf m=motion d=debug s=screenshot")
    print(f"{'─'*60}")

    # Logging
    log_path = None
    if args.log:
        log_path = init_log(source_name)
        print(f"[INFO] Logging to: {log_path}")

    # Video writer
    writer = None
    if args.save:
        sp = args.save_path or os.path.join(SCRIPT_DIR, f"v2_output_{source_name}.mp4")
        writer = cv2.VideoWriter(sp, cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps_video, (width, height))
        print(f"[INFO] Saving to: {sp}")

    # State
    frame_num = 0
    show_motion = args.show_motion
    show_debug = args.debug
    prev_time = time.time()
    fps_smooth = 0.0
    total_confirmed = 0
    total_new_alerts = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                if is_webcam or is_rtsp:
                    continue
                break

            frame_num += 1

            # ── Pipeline ─────────────────────────────────────────────────
            confirmed, pending, motion_score, motion_mask, motion_zones, yolo_ran = \
                detector.process_frame(frame, frame_num)

            total_confirmed += len(confirmed)
            new_alerts = [d for d in confirmed if d["alert_status"] == "CONFIRMED"]
            total_new_alerts += len(new_alerts)

            # ── Draw ─────────────────────────────────────────────────────
            if show_motion:
                frame = draw_motion_overlay(frame, motion_mask, motion_zones)

            frame = draw_detections_v2(frame, confirmed, pending)

            # FPS (smoothed)
            now = time.time()
            inst_fps = 1.0 / max(now - prev_time, 1e-6)
            fps_smooth = fps_smooth * 0.9 + inst_fps * 0.1 if fps_smooth else inst_fps
            prev_time = now

            frame = draw_status_bar(frame, fps_smooth, len(confirmed),
                                    len(pending), motion_score, yolo_ran,
                                    frame_num, args.mode)

            if show_debug:
                frame = draw_debug_panel(frame, detector, motion_score,
                                         motion_zones, yolo_ran)

            # ── Alerts ───────────────────────────────────────────────────
            for d in new_alerts:
                sev = d["severity"]
                icon = {"CRITICAL": "!!!","HIGH": "!!", "MEDIUM": "!"}.get(sev, "")
                print(f"  [{sev}]{icon} Frame {frame_num}: "
                      f"{d['class']} ({d['confidence']:.2f})")

            # ── Log ──────────────────────────────────────────────────────
            if log_path and (confirmed or pending):
                log_detections(log_path, frame_num, confirmed + pending)

            # ── Save ─────────────────────────────────────────────────────
            if writer:
                writer.write(frame)

            # ── Display ──────────────────────────────────────────────────
            if not args.no_display:
                cv2.imshow("Smart Lens v2 - Enhanced Detection", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('p'):
                    cv2.waitKey(0)
                elif key in (ord('+'), ord('=')):
                    detector.adjust_thresholds(0.05)
                    print(f"[INFO] Thresholds: {detector.class_thresholds}")
                elif key == ord('-'):
                    detector.adjust_thresholds(-0.05)
                    print(f"[INFO] Thresholds: {detector.class_thresholds}")
                elif key == ord('m'):
                    show_motion = not show_motion
                    print(f"[INFO] Motion overlay: {'ON' if show_motion else 'OFF'}")
                elif key == ord('d'):
                    show_debug = not show_debug
                    print(f"[INFO] Debug panel: {'ON' if show_debug else 'OFF'}")
                elif key == ord('s'):
                    ss = f"screenshot_v2_{source_name}_f{frame_num}.jpg"
                    cv2.imwrite(ss, frame)
                    print(f"[INFO] Screenshot: {ss}")

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted")

    finally:
        cap.release()
        if writer:
            writer.release()
        if not args.no_display:
            cv2.destroyAllWindows()

        # ── Summary ──────────────────────────────────────────────────────
        stats = detector.get_stats_summary()
        duration = frame_num / max(fps_video, 1)
        print(f"\n{'═'*60}")
        print(f"  Smart Lens v2 — Session Summary")
        print(f"{'═'*60}")
        print(f"  Source:              {args.source}")
        print(f"  Mode:                {args.mode}")
        print(f"  Duration:            {duration:.1f}s ({frame_num} frames)")
        print(f"  Avg FPS:             {fps_smooth:.0f}")
        print(f"  ─── Pipeline ────────────────────────────")
        print(f"  Frames skipped:      {stats['skip_rate']} (no motion)")
        print(f"  YOLO invoked:        {stats['yolo_rate']}")
        print(f"  Raw detections:      {stats['raw_dets']}")
        print(f"  Class-filtered:      {stats['class_filtered']}")
        print(f"  Temporal-confirmed:  {stats['confirmed']}")
        print(f"  Temporal-pending:    {stats['pending_dropped']}")
        print(f"  ─── Alerts ──────────────────────────────")
        print(f"  New alerts raised:   {total_new_alerts}")
        if log_path:
            print(f"  Log file:            {log_path}")
        print(f"{'═'*60}")


if __name__ == "__main__":
    main()
