"""
Smart Lens - YOLOv8 Model Inference Script
==========================================
Test the trained threat detection model on any video file or webcam.

Usage:
    python test_model.py                          # Webcam (default)
    python test_model.py --source video.mp4       # Video file
    python test_model.py --source rtsp://...      # RTSP stream
    python test_model.py --source image.jpg       # Single image
    python test_model.py --source 0               # Webcam index
"""

import argparse
import sys
import os
import time
from pathlib import Path

from ultralytics import YOLO
import cv2

# ─── Configuration ───────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "trained_models",
                          "smart_lens_v1_20260208_0043", "best.pt")

CLASSES = ["Fighting", "Fire", "Gun", "Knife"]

# Color map per class (BGR)
CLASS_COLORS = {
    "Fighting": (0, 0, 255),     # Red
    "Fire":     (0, 165, 255),   # Orange
    "Gun":      (0, 255, 255),   # Yellow
    "Knife":    (255, 0, 255),   # Magenta
}

DEFAULT_CONF = 0.5    # Confidence threshold
DEFAULT_IOU  = 0.45   # NMS IoU threshold


def parse_args():
    parser = argparse.ArgumentParser(
        description="Smart Lens - Test threat detection model on video/image/webcam"
    )
    parser.add_argument(
        "--source", type=str, default="0",
        help="Video file path, image path, RTSP URL, or webcam index (default: 0)"
    )
    parser.add_argument(
        "--model", type=str, default=MODEL_PATH,
        help=f"Path to .pt model file (default: {MODEL_PATH})"
    )
    parser.add_argument(
        "--conf", type=float, default=DEFAULT_CONF,
        help=f"Confidence threshold (default: {DEFAULT_CONF})"
    )
    parser.add_argument(
        "--iou", type=float, default=DEFAULT_IOU,
        help=f"NMS IoU threshold (default: {DEFAULT_IOU})"
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Save output video with detections"
    )
    parser.add_argument(
        "--save-path", type=str, default=None,
        help="Path for saved output video (default: output_<source_name>.mp4)"
    )
    parser.add_argument(
        "--no-display", action="store_true",
        help="Disable live display window (useful for headless/server)"
    )
    parser.add_argument(
        "--img-size", type=int, default=640,
        help="Inference image size (default: 640)"
    )
    return parser.parse_args()


def load_model(model_path, device=None):
    """Load the trained YOLO model."""
    if not os.path.exists(model_path):
        print(f"[ERROR] Model file not found: {model_path}")
        sys.exit(1)

    print(f"[INFO] Loading model: {model_path}")
    model = YOLO(model_path)

    if device:
        model.to(device)

    print(f"[INFO] Model loaded successfully")
    print(f"[INFO] Classes: {CLASSES}")
    return model


def draw_detections(frame, results, conf_threshold):
    """Draw bounding boxes and labels on the frame."""
    detections = []

    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue

        for box in boxes:
            conf = float(box.conf[0])
            if conf < conf_threshold:
                continue

            cls_id = int(box.cls[0])
            cls_name = CLASSES[cls_id] if cls_id < len(CLASSES) else f"Class_{cls_id}"
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            color = CLASS_COLORS.get(cls_name, (0, 255, 0))
            label = f"{cls_name} {conf:.2f}"

            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Draw label background
            (label_w, label_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(frame, (x1, y1 - label_h - 10),
                          (x1 + label_w, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            detections.append({
                "class": cls_name,
                "confidence": conf,
                "bbox": (x1, y1, x2, y2)
            })

    return frame, detections


def add_status_bar(frame, fps, det_count, frame_num):
    """Add a status bar overlay at the top."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 40), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    status = f"Smart Lens | FPS: {fps:.1f} | Detections: {det_count} | Frame: {frame_num}"
    cv2.putText(frame, status, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Alert indicator when threats detected
    if det_count > 0:
        cv2.putText(frame, "!! THREAT DETECTED !!", (w - 280, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    return frame


def run_inference(args):
    """Main inference loop."""
    model = load_model(args.model)

    # Determine source
    source = args.source
    is_webcam = source.isdigit()
    is_image = source.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))

    if is_webcam:
        source = int(source)
        print(f"[INFO] Opening webcam index: {source}")
    elif is_image:
        print(f"[INFO] Running inference on image: {source}")
    else:
        print(f"[INFO] Opening video source: {source}")

    # ─── Image mode ──────────────────────────────────────────────────────
    if is_image:
        frame = cv2.imread(source)
        if frame is None:
            print(f"[ERROR] Could not read image: {source}")
            sys.exit(1)

        results = model.predict(frame, conf=args.conf, iou=args.iou,
                                imgsz=args.img_size, verbose=False)
        frame, detections = draw_detections(frame, results, args.conf)

        print(f"\n[RESULTS] {len(detections)} detection(s):")
        for d in detections:
            print(f"  - {d['class']}: {d['confidence']:.2f} at {d['bbox']}")

        if args.save:
            save_path = args.save_path or f"output_{Path(args.source).stem}.jpg"
            cv2.imwrite(save_path, frame)
            print(f"[INFO] Saved to: {save_path}")

        if not args.no_display:
            cv2.imshow("Smart Lens - Detection", frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return

    # ─── Video/Webcam mode ───────────────────────────────────────────────
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video source: {args.source}")
        sys.exit(1)

    # Get video properties
    fps_video = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"[INFO] Video: {width}x{height} @ {fps_video:.1f} FPS"
          f" | Total frames: {total_frames if total_frames > 0 else 'stream'}")
    print(f"[INFO] Confidence threshold: {args.conf}")
    print(f"[INFO] Press 'q' to quit, 'p' to pause, '+'/'-' to adjust confidence")

    # Video writer setup
    writer = None
    if args.save:
        save_path = args.save_path or f"output_{Path(str(args.source)).stem}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(save_path, fourcc, fps_video, (width, height))
        print(f"[INFO] Saving output to: {save_path}")

    frame_num = 0
    total_detections = 0
    conf_threshold = args.conf
    prev_time = time.time()
    fps_display = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                if is_webcam:
                    continue
                print("[INFO] End of video")
                break

            frame_num += 1

            # Run detection
            results = model.predict(frame, conf=conf_threshold, iou=args.iou,
                                    imgsz=args.img_size, verbose=False)

            # Draw detections
            frame, detections = draw_detections(frame, results, conf_threshold)
            total_detections += len(detections)

            # Calculate FPS
            curr_time = time.time()
            fps_display = 1.0 / max(curr_time - prev_time, 1e-6)
            prev_time = curr_time

            # Add status bar
            frame = add_status_bar(frame, fps_display, len(detections), frame_num)

            # Print alerts for detected threats
            if detections:
                for d in detections:
                    print(f"  [ALERT] Frame {frame_num}: "
                          f"{d['class']} ({d['confidence']:.2f}) at {d['bbox']}")

            # Save frame
            if writer:
                writer.write(frame)

            # Display
            if not args.no_display:
                cv2.imshow("Smart Lens - Live Detection", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("[INFO] Quit requested")
                    break
                elif key == ord('p'):
                    print("[INFO] Paused. Press any key to resume...")
                    cv2.waitKey(0)
                elif key == ord('+') or key == ord('='):
                    conf_threshold = min(conf_threshold + 0.05, 0.95)
                    print(f"[INFO] Confidence threshold: {conf_threshold:.2f}")
                elif key == ord('-'):
                    conf_threshold = max(conf_threshold - 0.05, 0.05)
                    print(f"[INFO] Confidence threshold: {conf_threshold:.2f}")

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")

    finally:
        cap.release()
        if writer:
            writer.release()
            print(f"[INFO] Output saved to: {save_path}")
        if not args.no_display:
            cv2.destroyAllWindows()

        print(f"\n{'='*50}")
        print(f"  Smart Lens - Session Summary")
        print(f"{'='*50}")
        print(f"  Frames processed: {frame_num}")
        print(f"  Total detections: {total_detections}")
        print(f"  Avg FPS: {fps_display:.1f}")
        print(f"  Confidence threshold: {conf_threshold:.2f}")
        print(f"{'='*50}")


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)
