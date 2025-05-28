
# 🔍 SuperScan: Real-Time Assistive Vision System for the Visually Impaired

SuperScan is a real-time object detection and awareness assistant designed for visually impaired individuals. It uses a USB camera, Coral Edge TPU, and a Raspberry Pi 5 to detect, track, and describe potentially dangerous objects using speech feedback. The system performs inference on-device with Edge TPU–accelerated models, providing low-latency, power-efficient visual assistance.

---

## 🎯 Features

- **Real-Time Object Detection** with SSD MobileNet V2 (Edge TPU-compiled TFLite)
- **Depth Estimation** using MiDaS V2 model quantized for Edge TPU
- **Object Tracking** with SORT Kalman Filter-based tracking
- **Dynamic Object Scoring** based on:
  - Depth proximity
  - Motion trajectory
  - Object type priority
- **Natural Language Generation** for contextual feedback (e.g., *"A car is approaching fast on the left"*)
- **Audio Feedback** using Google Text-to-Speech
- **GStreamer-based Pipeline** for efficient video capture and rendering
- **Runs fully on Raspberry Pi 5 + Coral USB Accelerator**

---

## 🗂️ Project Structure

```
.
├── detecter.py              # Main detection pipeline
├── gstreamer.py             # GStreamer camera integration
├── depth_midas_output.py    # MiDaS depth inference module
├── object_scorer.py         # Computes danger scores
├── tracker.py               # Tracker object factory
├── sort.py                  # SORT tracking algorithm
├── text_generator.py        # Converts object data into spoken descriptions
├── audio.py                 # gTTS audio playback
├── common.py                # Shared utilities and EdgeTPU integration
├── models/                  # TFLite models (SSD, MiDaS)
└── coco_labels.txt          # Label file for SSD model
```

---

## 🚀 How It Works

1. **Frame Capture**  
   GStreamer or OpenCV reads each frame from USB or video file.

2. **Object Detection & Depth Estimation**  
   - Objects are detected using MobileNet-SSD v2 (quantized for EdgeTPU).
   - Depth maps are inferred using MiDaS v2 TFLite model.

3. **Tracking & Scoring**  
   - Objects are tracked across frames (SORT).
   - Each object gets a score based on importance, motion, and proximity.

4. **Text Generation**  
   - Descriptive feedback is created per object (e.g., distance, speed, direction).
   - Alerts are prioritized based on score thresholds.

5. **Speech Output**  
   - Warnings and contextual descriptions are synthesized using `gTTS`.

---

## 📦 Dependencies

- Python 3.9+
- [tflite_runtime](https://www.tensorflow.org/lite/guide/python)
- OpenCV
- GStreamer with Python bindings (`gi`)
- `gTTS`, `mpg123`, `ffmpeg`

To install dependencies:

```bash
sudo apt update
sudo apt install python3-opencv gstreamer1.0-tools                  gstreamer1.0-plugins-{base,good,bad,ugly}                  libgstreamer1.0-dev libglib2.0-dev                  mpg123 ffmpeg
pip install tflite-runtime gTTS numpy
```

---

## 🛠️ Run the System

### Live Camera (Raspberry Pi + Coral USB):

```bash
python3 detecter.py --use_camera
```

### Pre-recorded Video:

```bash
python3 detecter.py --model models/ssd_model.tflite --labels models/coco_labels.txt
```

---

## 🎥 Sample Output

- `detection_output_phone1.mp4` — Video with bounding boxes and arrows
- `depth_output_phone1.mp4` — Depth visualization
- Real-time alerts played via speaker

---

## 🙏 Acknowledgements

- Portions of this project (e.g., `common.py`, `gstreamer.py`, `tracker.py`) are adapted from Coral examples provided by Google Coral (https://coral.ai), licensed under the Apache License 2.0.
- SORT implementation is adapted from the original [SORT repository by Alex Bewley](https://github.com/abewley/sort), licensed under the GPLv3.

---

## 📄 License

- `sort.py` is licensed under the GNU General Public License (GPL v3).
- `common.py`, `gstreamer.py`, and `tracker.py` are adapted from Coral examples licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).

---
