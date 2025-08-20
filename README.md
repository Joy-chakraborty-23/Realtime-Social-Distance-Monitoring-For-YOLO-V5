

## ✨ Features

* **YOLOv5-based person detection** for accurate recognition in real-world scenes
* **Real-time social distancing monitoring** from live or recorded video
* **Perspective calibration** for improved distance measurement
* **Bird’s eye view visualization** to clearly display spacing between individuals
* **Risk classification** (High, Medium, Low) based on configurable thresholds
* **Automatic video output** with bounding boxes, distance violations, and risk stats
* **Customizable parameters** for confidence threshold, IOU, and distance limits

---

## 🎯 Use Cases

* **Public safety monitoring** in malls, transport hubs, or event venues
* **Workplace compliance** to enforce distancing guidelines in factories/offices
* **Retail environments** for customer flow and safety analysis
* **Research & education** in computer vision, deep learning, and smart surveillance
* **Prototype development** for smart city or AI-powered monitoring solutions



## 🚀 Getting Started

Follow these steps to set up and run the **Social Distancing Monitor** on your system.

### 1️⃣ Clone the Repository

First, clone this repository and enter the project folder:

```bash
git clone https://github.com/<your-username>/social-distancing-monitor.git
cd social-distancing-monitor
```

### 2️⃣ Install Dependencies

Create a virtual environment (recommended) and install dependencies:

```bash
# Create and activate venv (Linux/Mac)
python3 -m venv venv
source venv/bin/activate

# On Windows
python -m venv venv
venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

### 3️⃣ Clone YOLOv5 (Required for Detection)

This project uses **YOLOv5** as a submodule. Clone it inside the repo:

```bash
git clone https://github.com/ultralytics/yolov5.git
cd yolov5
pip install -r requirements.txt
cd ..
```

### 4️⃣ Download YOLOv5 Weights

By default, the project uses **YOLOv5s** weights (lightweight). Download it automatically with:

```bash
python -c "from pathlib import Path; import torch; \
torch.hub.download_url_to_file('https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt', 'yolov5s.pt')"
```

Alternatively, you can manually download `yolov5s.pt` from [YOLOv5 Releases](https://github.com/ultralytics/yolov5/releases).

### 5️⃣ Prepare Input Video

Place your input video inside the `Input/` directory. Example:

```
social-distancing-monitor/
 ├── Input/
 │    └── test4.mp4
```

### 6️⃣ Run the Monitor

Start the program using:

```bash
python scripts/run_monitor.py
```

During the **calibration step**, you will be asked to click **7 points** on the first video frame:

1. Four corner points of the ground plane (for perspective transformation)
2. Two points that represent the social distance in the scene (e.g., 1 meter apart)
3. One point to confirm and finish calibration

### 7️⃣ View Results

* The processed video with bounding boxes will appear in a window (`Social Distancing Monitor`).
* A bird’s eye view will be displayed in another window (`Bird's Eye View`).
* Results will also be saved in the `Output/` folder as:

  * `perspective.avi` → Video with bounding boxes and risk stats
  * `birds_eye.avi` → Bird’s eye visualization

### 8️⃣ Exit

Press **`q`** anytime to stop processing.


