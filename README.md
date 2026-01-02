# 🚗 AI-Based License Plate Detection System

An end-to-end **AI-powered License Plate Detection & Recognition system** that detects vehicle license plates from video streams, performs OCR to extract plate numbers, identifies Indian states, triggers alerts, saves snapshots, and sends events to a backend server.

---

## ✨ Features

* 🎯 **Real-time License Plate Detection** using YOLO (ONNX model)
* 🔍 **OCR with Google Vision API** for accurate plate text extraction
* 🇮🇳 **Indian State Identification** from license plate format
* 🎥 **Supports Video Files & Live Camera Feeds**
* 🧠 **Motion-based & Static Detection Modes**
* 📐 **Zone-based Detection**

  * Line crossing detection
  * ROI (rectangle) based detection
* 🖼️ **Automatic Snapshot Saving** (blur-free plates only)
* 📤 **Backend Integration (Flask API)** for event storage
* 📧 **Email Alerts** on new plate detection
* 🧾 **Duplicate Plate Prevention** (alerts sent only once per plate)

---

## 🏗️ Project Structure

```
├── licenseplate.py          # Main detection & OCR pipeline
├── licensebackend.py        # Flask backend API
├── license_plate_detector.onnx  # Trained YOLO model
├── apikey.json              # Google Vision API credentials
├── lisnaps/                 # Saved plate snapshots (auto-generated)
├── backend_snaps/           # Backend-received images
├── licensevideo.mp4         # Sample input video
└── README.md
```

---

## ⚙️ Requirements

Install the required dependencies:

```bash
pip install opencv-python ultralytics google-cloud-vision flask requests
```

Also ensure:

* Python 3.8+
* Google Vision API enabled
* Service account key (`apikey.json`) configured

---

## 🔑 Google Vision Setup

1. Create a Google Cloud project
2. Enable **Vision API**
3. Create a **Service Account**
4. Download the JSON key and rename it to `apikey.json`
5. Place it in the project root

---

## 🚀 How to Run

### 1️⃣ Start Backend Server

```bash
python licensebackend.py
```

Backend runs at:

```
http://127.0.0.1:5000/plate_event
```

---

### 2️⃣ Run License Plate Detection

```bash
python licenseplate.py --mode motion --zone line --mark enable --alert none
```

#### Available Arguments:

| Argument  | Options                | Description                                |
| --------- | ---------------------- | ------------------------------------------ |
| `--mode`  | `motion`, `static`     | Detection based on motion or static frames |
| `--zone`  | `line`, `rect`, `off`  | Zone-based detection                       |
| `--mark`  | `enable`, `disable`    | Draw bounding boxes                        |
| `--alert` | `email`, `none`        | Email notification                         |
| `--plate` | `all`, `license_plate` | Plate type filter                          |

---

## 📸 Output

* 📁 **Snapshots saved automatically** in `lisnaps/DD-MM-YY/`
* 📁 **Backend images stored** in `backend_snaps/`
* 🖥️ **Live annotated video display**
* 📧 **Email alert (optional)** with detected plate image

---

## 🛡️ Security Note

⚠️ **Do NOT upload `apikey.json` or email credentials to GitHub**

Add the following to `.gitignore`:

```
apikey.json
*.onnx
```

---

## 🔮 Future Enhancements

* 🚦 Vehicle type classification
* 🗄️ Database integration
* 🌐 Web dashboard
* 📊 Analytics & reporting
* 📱 Mobile notifications

---

## 👩‍💻 Author

**Sangareshwari A**
AI Engineer | Computer Vision | Video Analytics

---

⭐ If you like this project, give it a star on GitHub!
