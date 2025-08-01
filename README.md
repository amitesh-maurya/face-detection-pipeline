Face Detection Pipeline ✨
A plug-and-play, production-ready face detection pipeline with support for OpenCV (Haar Cascade), MTCNN, Dlib, or face_recognition.
Works with images, URLs, webcam streams, base64 strings, and ships with a REST API (Flask) outta the box.

🚀 Features
Multi-Detector Flex → OpenCV Haar, MTCNN, Dlib, or face_recognition

Any Input You Throw At It → Files, URLs, Base64, webcams, numpy arrays

Pro Preprocessing → Resize, grayscale, histogram equalization, blur, gamma correction

Smart Post-processing → Confidence thresholding, NMS, bounding box refinements

Rich Output → Bounding boxes, confidence scores, optional landmarks, meta info

REST API Ready → Easily integrate w/ your app

Auto Output Saving → Annotated images + JSON results on demand

Verbose Logging → Debug & production safe

🔧 Installation
Requirements: Python 3.8+

bash
Copy
Edit
git clone https://github.com/your-username/your-repo-name.git  
cd your-repo-name  
pip install -r requirements.txt
Heads up: dlib & face_recognition may need system-level packages.

🖥️ Usage
Python Script
python
Copy
Edit
from face_pipeline import FaceDetectionPipeline

pipeline = FaceDetectionPipeline(detector_type="opencv")
results = pipeline.process_image("test.jpg", input_type="file", save_output=True)
print(results)
URL Input
python
Copy
Edit
results = pipeline.process_image("https://path.to/image.jpg", input_type="url")
Webcam
python
Copy
Edit
results = pipeline.process_image(0, input_type="webcam", save_output=True)
🌐 REST API
Run server:

bash
Copy
Edit
python your_script.py  
# or
export FLASK_APP=your_api_module.py && flask run
API default: http://localhost:5000

Endpoints
POST /api/detect → JSON body: { "image_url": "...", "image_base64": "..." }

POST /api/detect/upload → File upload

GET /api/health → Status check

Curl Sample
bash
Copy
Edit
curl -X POST http://localhost:5000/api/detect \
  -H "Content-Type: application/json" \
  -d '{"image_url": "https://path.to/image.jpg"}'
⚙️ Config
python
Copy
Edit
pipeline = FaceDetectionPipeline(
    detector_type="mtcnn",  # opencv | mtcnn | dlib | face_recognition
    confidence_threshold=0.8,
    nms_threshold=0.4,
    target_size=(640, 480)
)
📦 Output Example
json
Copy
Edit
{
  "status": "success",
  "timestamp": "2025-08-01T23:55:00Z",
  "face_count": 2,
  "faces": [
    {
      "face_id": "face_0_20250801_235500",
      "bounding_box": {"x": 100, "y": 80, "width": 60, "height": 60},
      "confidence": 0.9921,
      "detector_used": "mtcnn",
      "metadata": {"area": 3600, "aspect_ratio": 1.0, "center": [130, 110]},
      "landmarks": {...}
    }
  ]
}
🛠 Dev Notes
Linting: flake8 + black + isort

CI/CD: GitHub Actions ready

Tests: Please add before merging

🤝 Contributing
PRs & issues welcome! Got ideas? Drop ‘em in Issues or hit up the maintainer.

📄 License
MIT — basically, do cool stuff with it.

Want me to also:

Add badges (like Python version, license, build passing)?

Throw in cool emoji section headers everywhere?

Make it look like a GitHub trending project (with shields + collapsible sections)?
