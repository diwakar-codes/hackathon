"""Stage 1: always-on, cheap candidate flagger.

Runs on every sampled frame. Goal is high recall, not precision -- it just
needs to narrow "all footage" down to "segments worth showing the Stage 2
VLM verifier". Currently covers: motion bursts, stopped vehicles, crowds
and loitering. It does NOT yet cover fire/smoke/flood/road-spill/wrong-way
driving -- those rely on Stage 2's periodic low-duty-cycle catch-all sample
until dedicated signals are added here.
"""

import cv2
from ultralytics import YOLO

from .motion import motion_energy
from .tracker import SimpleTracker, VEHICLE_CLASSES, PERSON_CLASS

MOTION_BURST_THRESHOLD = 25.0
STOPPED_SECONDS_THRESHOLD = 8
CROWD_SIZE_THRESHOLD = 4
LOITER_SECONDS_THRESHOLD = 15


def run_stage1(video_path, sample_fps=2, yolo_model="yolov8n.pt"):
    model = YOLO(yolo_model)
    cap = cv2.VideoCapture(video_path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_interval = max(1, int(src_fps / sample_fps))

    tracker = SimpleTracker()
    prev_gray = None
    candidates = []
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % frame_interval != 0:
            frame_idx += 1
            continue

        t = frame_idx / src_fps
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        reasons = []

        if prev_gray is not None:
            energy = motion_energy(prev_gray, gray)
            if energy > MOTION_BURST_THRESHOLD:
                reasons.append(("motion_burst", energy))
        prev_gray = gray

        result = model(frame, verbose=False)[0]
        detections = []
        person_count = 0
        for box in result.boxes:
            cls_name = model.names[int(box.cls)]
            if cls_name not in VEHICLE_CLASSES and cls_name != PERSON_CLASS:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            detections.append((cx, cy, cls_name))
            if cls_name == PERSON_CLASS:
                person_count += 1

        tracks = tracker.update(detections, t)

        for track in tracks.values():
            stationary_for = t - track["stationary_since"]
            if track["class"] in VEHICLE_CLASSES and stationary_for > STOPPED_SECONDS_THRESHOLD:
                reasons.append(("stopped_vehicle", stationary_for))
            if track["class"] == PERSON_CLASS and stationary_for > LOITER_SECONDS_THRESHOLD:
                reasons.append(("loitering", stationary_for))

        if person_count >= CROWD_SIZE_THRESHOLD:
            reasons.append(("crowd", person_count))

        if reasons:
            candidates.append({"t": t, "frame_idx": frame_idx, "reasons": reasons})

        frame_idx += 1

    cap.release()
    return candidates
