import gradio as gr
import numpy as np
from face_detection_pipeline import FaceDetectionPipeline, draw_detections

# Initialize pipelines for all detectors you support
detectors = ["opencv", "mtcnn", "dlib", "face_recognition"]
pipelines = {det: FaceDetectionPipeline(detector_type=det) for det in detectors}

def gradio_face_detect(image: np.ndarray, detector: str):
    pipe = pipelines[detector]
    results = pipe.process_image(image, input_type="array", save_output=False)
    faces = results.get("faces", [])
    annotated = draw_detections(image, [
        {
            "bbox": [
                f["bounding_box"]["x"],
                f["bounding_box"]["y"],
                f["bounding_box"]["width"],
                f["bounding_box"]["height"]
            ],
            "confidence": f["confidence"],
            "landmarks": f.get("landmarks"),
            "detector": detector
        } for f in faces
    ]) if faces else image
    return annotated, results

gr.Interface(
    fn=gradio_face_detect,
    inputs=[
        gr.Image(type="numpy", label="Upload or Webcam"),
        gr.Dropdown(choices=detectors, value="opencv", label="Face Detector")
    ],
    outputs=[
        gr.Image(label="Annotated Image"),
        gr.JSON(label="Detection Results/Metadata")
    ],
    title="Face Detection Demo",
    description="Upload an image or use your webcam. Choose the detection model and run face detection."
).launch()
