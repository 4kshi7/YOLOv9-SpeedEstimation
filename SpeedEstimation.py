import cv2
import numpy as np
import time
import torch
from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes
from utils.torch_utils import select_device
from scipy.spatial import distance
import pytesseract
from collections import deque  # Sliding window for smoothing speed

# ✅ Configure Tesseract OCR
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ✅ Detect GPU & Use CUDA if Available
device = select_device(0)  # Force CUDA device selection
print(f"Using device: {device}")

# ✅ Load YOLOv9 Model
weights = "weights/yolov9-e.pt"  # Path to YOLOv9 weights
vehicle_model = DetectMultiBackend(weights, device=device)
stride, names, pt = vehicle_model.stride, vehicle_model.names, vehicle_model.pt

# ✅ Load Video
video_path = r"C:\Users\acre 2\Downloads\video.MP4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("[ERROR] Video file could not be loaded. Check the file path!")
    exit()

# ✅ Calibration Settings
fps = cap.get(cv2.CAP_PROP_FPS) or 30
speed_limit_kmh = 60
speed_limit_mps = speed_limit_kmh / 3.6  # Correct conversion

# ✅ Initialize Data Structures
vehicle_tracks = {}  # Stores {vehicle_id: (x, y, timestamp, speed_kmh)}
speed_history = {}  # Stores speed history for smoothing
frame_skip = 2  # Process every 2nd frame for efficiency

# ✅ Define Speed Measurement Lines
line1_y = 300
line2_y = 500
vehicle_classes = {2, 3, 5, 7}  # Car, motorcycle, bus, truck

print("[INFO] Starting Speed Detection and Plate Logging...")

# ✅ Process Video Frames Efficiently
frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()
    frame_count += 1

    # ✅ Skip frames for efficiency (Process every 2nd frame)
    if frame_count % frame_skip != 0:
        continue

    # ✅ Faster YOLO Processing: Avoid unnecessary rescaling
    img = cv2.resize(frame, (640, 384))
    img = img[:, :, ::-1].transpose(2, 0, 1)  # Convert BGR to RGB
    img = np.ascontiguousarray(img, dtype=np.float32) / 255.0  # Normalize and reduce memory overhead

    with torch.no_grad():
        img = torch.from_numpy(img).to(device).unsqueeze(0)
        pred = vehicle_model(img, augment=False, visualize=False)[0]
        pred = non_max_suppression(pred, 0.25, 0.45, classes=None, agnostic=False)

    # ✅ Draw Reference Lines (Once per frame, not per object)
    cv2.line(frame, (0, line1_y), (frame.shape[1], line1_y), (255, 0, 0), 2)
    cv2.line(frame, (0, line2_y), (frame.shape[1], line2_y), (255, 0, 0), 2)

    # ✅ Efficient Object Processing
    for det in pred:
        if len(det):
            det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], frame.shape).round()
            for *xyxy, conf, cls in reversed(det):
                x1, y1, x2, y2 = map(int, xyxy)
                class_id = int(cls)
                confidence = conf

                if class_id in vehicle_classes and confidence > 0.5:
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    vehicle_width = max(x2 - x1, 1)  # Avoid division by zero

                    avg_vehicle_length = 4.5  # Approximate length of a car in meters
                    meters_per_pixel = avg_vehicle_length / vehicle_width

                    # ✅ Improved Vehicle Matching using Spatial Distance
                    vehicle_id = None
                    min_distance = float("inf")
                    for vid, (vx, vy, vt, _) in vehicle_tracks.items():
                        dist = distance.euclidean((vx, vy), (center_x, center_y))
                        if dist < min_distance and dist < 50:  # Reduced threshold for better tracking
                            min_distance = dist
                            vehicle_id = vid

                    if vehicle_id is None:
                        vehicle_id = len(vehicle_tracks) + 1

                    # ✅ Default speed if not calculated yet
                    smooth_speed_kmh = speed_history.get(vehicle_id, deque([0], maxlen=5))[-1]
                    smooth_speed_mps = smooth_speed_kmh / 3.6

                    # ✅ Speed Calculation
                    if vehicle_id in vehicle_tracks:
                        px, py, pt, _ = vehicle_tracks[vehicle_id]
                        dist_meters = distance.euclidean((px, py), (center_x, center_y)) * meters_per_pixel
                        dt = current_time - pt

                        if dt > 0:
                            raw_speed_kmh = (dist_meters / dt) * 3.6
                            raw_speed_mps = raw_speed_kmh / 3.6

                            # ✅ Store speed history
                            if vehicle_id not in speed_history:
                                speed_history[vehicle_id] = deque(maxlen=5)
                            speed_history[vehicle_id].append(raw_speed_kmh)

                            # ✅ Smoothed Speed Calculation (Moving Average)
                            smooth_speed_kmh = np.mean(speed_history[vehicle_id])
                            smooth_speed_mps = smooth_speed_kmh / 3.6

                            # ✅ Display Speed Overlay
                            color = (0, 255, 0) if smooth_speed_kmh <= speed_limit_kmh else (0, 0, 255)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(frame, f"{smooth_speed_kmh:.1f} km/h ({smooth_speed_mps:.1f} m/s)",
                                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                    # ✅ Update Tracking Data
                    vehicle_tracks[vehicle_id] = (center_x, center_y, current_time, smooth_speed_kmh)

                    print(f"[INFO] Vehicle {vehicle_id}: {smooth_speed_kmh:.2f} km/h, {smooth_speed_mps:.2f} m/s")

    # ✅ Display the frame every `frame_skip` frames
    try:
        cv2.imshow("Vehicle Speed Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    except cv2.error:
        print("[WARNING] OpenCV GUI functions are not supported in this environment.")
        break  # Exit loop to prevent further errors

cap.release()
cv2.destroyAllWindows()

print("[INFO] Detection completed.")
