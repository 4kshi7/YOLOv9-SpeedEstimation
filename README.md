# YOLOv9-Based Speed Estimation System

This project is a **YOLOv9-Based Speed Estimation System** that detects vehicles in a video feed and estimates their speed in real time. The system utilizes **YOLOv9 for object detection** and **distance-based calculations** for speed estimation.

This repository is based on [YOLOv9](https://github.com/WongKinYiu/yolov9), and we extend its functionality for **real-time vehicle speed estimation.**

---

## 🔧 Features

- **Real-Time Vehicle Detection** using **YOLOv9**
- **Speed Estimation** based on frame-to-frame distance calculations
- **License Plate Recognition** using **Tesseract OCR** (Optional)
- **Frame Optimization** for efficiency
- **Logs Speed Data** to an Excel file
- **CUDA-Accelerated Inference** for better performance (if available)

---

## 📂 Project Structure

```
YOLOv9-Based-Speed-Estimation-System/
│── classify/
│── data/
│── figure/
│── models/
│── panoptic/
│── scripts/
│── segment/
│── tools/
│── utils/
│── weights/  → YOLOv9 Model Weights (.pt files)
│── SpeedEstimation.py  → Main Speed Estimation Script
│── requirements.txt  → Required Dependencies
│── README.md  → This Document
│── LICENSE.md  → License and Credits
```

---

## 📥 Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/4kshi7/YOLOv9-Based-Speed-Estimation-System.git
cd YOLOv9-Based-Speed-Estimation-System
```

### 2️⃣ Create a Virtual Environment (Recommended)
```bash
python -m venv yolov9_env  # Create virtual environment
source yolov9_env/bin/activate  # Linux/macOS
# OR
yolov9_env\Scripts\activate  # Windows
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Download YOLOv9 Weights
```bash
mkdir weights
curl -L -o weights/yolov9-e.pt "https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-e.pt"
```

---

## 🚀 Usage

### 1️⃣ Run the Speed Estimation Script
```bash
python SpeedEstimation.py
```

### 2️⃣ (Optional) Run with a Custom Video File
Modify `SpeedEstimation.py` and set:
```python
video_path = "path/to/your/video.mp4"
```
Then run:
```bash
python SpeedEstimation.py
```

---

## 🛠 Configuration

| Parameter       | Description |
|----------------|-------------|
| `video_path`   | Path to input video file |
| `weights/yolov9-e.pt` | YOLOv9 weights file |
| `fps`          | Frames per second of input video |
| `speed_limit_kmh` | Speed limit in km/h (for violation detection) |
| `line1_y` and `line2_y` | Position of reference lines for speed calculation |
| `vehicle_classes` | YOLO class IDs for vehicles (car, truck, bus, etc.) |

---

## 📊 Speed Estimation Method

1. Detects a **vehicle** in each frame.
2. Tracks vehicle movement across **two reference lines**.
3. Measures the distance traveled in pixels and converts it to meters.
4. Uses **frame rate (fps)** to calculate speed:
   
   **Speed (m/s) = Distance / Time**
   **Speed (km/h) = (Distance / Time) × 3.6**

5. Displays the estimated speed on the video feed.

---

## ❗ Known Issues

- **GUI Issues on Headless Systems**: If OpenCV’s `imshow` fails, disable the visualization or run in a GUI-enabled environment.
- **CUDA Compatibility**: Ensure your **NVIDIA drivers and PyTorch CUDA** versions match.

---

## 📝 Credits

- This project is based on [YOLOv9](https://github.com/WongKinYiu/yolov9) by **WongKinYiu**.
- Speed estimation logic is an extension built on top of YOLOv9’s vehicle detection.

---

## 📜 License

This project is open-source and follows the **Apache License 2.0**. See `LICENSE.md` for more details.


