# Face Detection Pipeline ✨
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Build Status](https://img.shields.io/github/actions/workflow/status/amitesh-maurya/face-detection-pipeline/python-app.yml?branch=main)](https://github.com/amitesh-maurya/face-detection-pipeline/actions)

A **modular, production-ready face detection pipeline** that supports:
- **Detectors:** OpenCV (Haar Cascade), MTCNN, Dlib, face_recognition  
- **Inputs:** Image paths, URLs, Base64, webcam streams, raw numpy arrays  
- **Outputs:** Bounding boxes, confidence scores, optional landmarks, metadata  
- **REST API:** Ready-to-use Flask endpoints for easy integration  

---

## ✨ Features
- **Multi-Detector Support** → OpenCV Haar, MTCNN, Dlib, face_recognition  
- **Flexible Input Sources** → Files, URLs, Base64, webcams, numpy arrays  
- **Advanced Preprocessing** → Resize, grayscale, histogram equalization, blurring, gamma correction  
- **Smart Post-processing** → Confidence thresholding, Non-Maximum Suppression (NMS), bounding box refinement  
- **Comprehensive Output** → JSON results, annotated images, and meta data  
- **Logging** → Verbose & production-ready  

---

## 🚀 Installation
**Requirements:** Python 3.8+  

```bash
git clone https://github.com/amitesh-maurya/face-detection-pipeline.git
cd face-detection-pipeline
pip install -r requirements.txt
```

> **Note:** `dlib` & `face_recognition` may need system-level dependencies.

---

## 🖥 Usage

### Python (Script)
```python
from face_detection_pipeline import FaceDetectionPipeline

pipeline = FaceDetectionPipeline(detector_type="opencv")
results = pipeline.process_image("test_image.jpg", input_type="file", save_output=True)
print(results)
```

### URL Input
```python
results = pipeline.process_image("https://path.to/image.jpg", input_type="url")
```

### Webcam
```python
results = pipeline.process_image(0, input_type="webcam", save_output=True)
```

---

## 🌐 REST API

### Run the API
```bash
python your_script.py
# or
export FLASK_APP=face_detection_pipeline.py && flask run
```

API runs at: **http://localhost:5000**

### Endpoints
- **POST** `/api/detect` → `{"image_url": "...", "image_base64": "..."}`  
- **POST** `/api/detect/upload` → Multipart file upload (file field)  
- **GET** `/api/health` → API health status  

#### Example with curl:
```bash
curl -X POST http://localhost:5000/api/detect     -H "Content-Type: application/json"     -d '{"image_url":"https://path.to/image.jpg"}'
```

---

## ⚙️ Configuration
```python
pipeline = FaceDetectionPipeline(
    detector_type="mtcnn",  # opencv | mtcnn | dlib | face_recognition
    confidence_threshold=0.8,
    nms_threshold=0.4,
    target_size=(640, 480)
)
```

---

## 📦 Output Example
```json
{
  "status": "success",
  "timestamp": "2025-08-01T23:55:00Z",
  "face_count": 2,
  "faces": [
    {
      "face_id": "AmiteshFD_0_20250801_235500",
      "bounding_box": {"x": 100, "y": 80, "width": 60, "height": 60},
      "confidence": 0.9921,
      "detector_used": "mtcnn",
      "metadata": {"area": 3600, "aspect_ratio": 1.0, "center": [130, 110]},
      "landmarks": {...}
    }
  ]
}
```

---

## 🛠 Development
- **Linting:** `flake8`, `black`, `isort`  
- **CI/CD:** GitHub Actions workflow included in `.github/workflows/python-app.yml`  
- **Testing:** Please add tests before contributing  

---

## 🤝 Contributing
Pull requests welcome! For feature requests or bug reports, open an issue.

---

## 📄 License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## ❤️ Acknowledgements
- OpenCV  
- Dlib  
- MTCNN  
- face_recognition  

---

> _“Happy face detecting!”_
