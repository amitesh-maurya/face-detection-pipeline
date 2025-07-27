import base64
import json
import logging
import os
from datetime import datetime
from io import BytesIO
from typing import List, Dict, Tuple, Union

import cv2
import dlib
import face_recognition
import numpy as np
from PIL import Image
from mtcnn import MTCNN
from requests import get

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _calculate_face_metrics(faces: List[Dict]) -> List[Dict]:
    """Calculate additional metrics for detected faces."""
    for i, face in enumerate(faces):
        bbox = face['bbox']
        x, y, w, h = bbox

        # Calculate face area
        face['area'] = w * h

        # Calculate aspect ratio
        face['aspect_ratio'] = w / h if h > 0 else 1.0

        # Calculate center point
        face['center'] = [x + w // 2, y + h // 2]

        # Add unique face ID
        face['face_id'] = f"face_{i}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    return faces


def _refine_bounding_boxes(faces: List[Dict]) -> List[Dict]:
    """Refine bounding box coordinates."""
    refined_faces = []

    for face in faces:
        bbox = face['bbox']
        x, y, w, h = bbox

        # Ensure bounding box is within image bounds
        x = max(0, x)
        y = max(0, y)

        # Add some padding around the face
        padding = 0.1
        pad_w = int(w * padding)
        pad_h = int(h * padding)

        refined_bbox = [
            max(0, x - pad_w),
            max(0, y - pad_h),
            w + 2 * pad_w,
            h + 2 * pad_h
        ]

        refined_face = face.copy()
        refined_face['bbox'] = refined_bbox
        refined_faces.append(refined_face)

    return refined_faces


def _detect_face_recognition(image: np.ndarray) -> List[Dict]:
    """Detect faces using face_recognition library[60]."""
    # Convert BGR to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_image, model="hog")

    detected_faces = []
    for (top, right, bottom, left) in face_locations:
        w, h = right - left, bottom - top
        detected_faces.append({
            'bbox': [int(left), int(top), int(w), int(h)],
            'confidence': 0.9,  # face_recognition doesn't provide confidence
            'landmarks': None,
            'detector': 'face_recognition'
        })

    return detected_faces


def _apply_gamma_correction(image: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """Apply gamma correction for illumination normalization[12]."""
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)


def capture_from_webcam(camera_index: int = 0) -> np.ndarray:
    """
    Capture frame from webcam.

    Args:
        camera_index: Camera device index

    Returns:
        Captured frame as numpy array
    """
    try:
        cap = cv2.VideoCapture(camera_index)[5]
        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera {camera_index}")

        ret, frame = cap.read()
        cap.release()

        if not ret:
            raise RuntimeError("Could not capture frame from camera")

        logger.info(f"Captured frame from camera {camera_index}")
        return frame

    except Exception as e:
        logger.error(f"Error capturing from webcam: {e}")
        raise


def load_image_from_url(image_url: str) -> np.ndarray:
    """
    Load image from URL.

    Args:
        image_url: URL of the image

    Returns:
        Loaded image as numpy array
    """
    try:
        response = get(image_url, timeout=10)
        response.raise_for_status()

        image = Image.open(BytesIO(response.content))
        image_array = np.array(image)

        # Convert RGB to BGR for OpenCV
        if len(image_array.shape) == 3:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

        logger.info(f"Loaded image from URL: {image_url}")
        return image_array

    except Exception as e:
        logger.error(f"Error loading image from URL: {e}")
        raise


def load_image_from_base64(base64_string: str) -> np.ndarray:
    """
    Load image from base64 encoded string.

    Args:
        base64_string: Base64 encoded image data

    Returns:
        Loaded image as numpy array
    """
    try:
        # Remove data URL prefix if present
        if base64_string.startswith('data:image'):
            base64_string = base64_string.split(',')[1]

        image_data = base64.b64decode(base64_string)
        image = Image.open(BytesIO(image_data))
        image_array = np.array(image)

        # Convert RGB to BGR for OpenCV
        if len(image_array.shape) == 3:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

        logger.info("Loaded image from base64 string")
        return image_array

    except Exception as e:
        logger.error(f"Error loading image from base64: {e}")
        raise


def load_image_from_file(image_path: str) -> np.ndarray:
    """
    Load image from file path.

    Args:
        image_path: Path to image file

    Returns:
        Loaded image as numpy array
    """
    try:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        image = cv2.imread(image_path)[1][2][5]
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")

        logger.info(f"Loaded image from file: {image_path}")
        return image

    except Exception as e:
        logger.error(f"Error loading image from file: {e}")
        raise


def draw_detections(image: np.ndarray, faces: List[Dict]) -> np.ndarray:
    """
    Draw bounding boxes and labels on image.

    Args:
        image: Input image
        faces: List of detected faces

    Returns:
        Image with drawn detections
    """
    try:
        result_image = image.copy()

        for face in faces:
            bbox = face['bbox']
            x, y, w, h = bbox
            confidence = face['confidence']

            # Draw bounding box
            var = cv2.rectangle ( result_image, (x, y), (x + w, y + h), (0, 255, 0), 2 )[1][2][8]

            # Draw confidence score
            label = f"{confidence:.2f}"
            cv2.putText(result_image, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Draw landmarks if available
            if face.get('landmarks'):
                landmarks = face['landmarks']
                if isinstance(landmarks, dict):
                    for point_name, (px, py) in landmarks.items():
                        cv2.circle(result_image, (int(px), int(py)), 2, (255, 0, 0), -1)

        logger.info(f"Drew detections for {len(faces)} faces")
        return result_image

    except Exception as e:
        logger.error(f"Error drawing detections: {e}")
        return image


def _initialize_detectors(self: object) -> object:
    pass


# noinspection PyTypeChecker
def save_results_to_file(results: Dict, output_path: str):
    """Save detection results to JSON file."""
    try:
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {output_path}")
    except Exception as e:
        logger.error(f"Error saving results: {e}")
        raise


class FaceDetectionPipeline:
    """
    Complete face detection pipeline implementing:
    1. Image Input (files, video frames, live streams)
    2. Preprocessing (resize, normalize, rotate, color space conversion)
    3. Face Detection (Haar, MTCNN, Dlib models)
    4. Post-processing (NMS, confidence filtering, bounding box refinement)
    5. Output API (REST endpoints with face coordinates, scores, landmarks)
    """

    def __init__(self,
                 detector_type: str = "opencv",
                 confidence_threshold: float = 0.5,
                 nms_threshold: float = 0.4,
                 target_size: Tuple[int, int] = (640, 480)) -> object:
        """
        Initialize the face detection pipeline.

        Args:
            detector_type: Type of detector ('opencv', 'mtcnn', 'dlib', 'face_recognition')
            confidence_threshold: Minimum confidence score for face detection
            nms_threshold: Non-maximum suppression threshold
            target_size: Target image size for processing
        """
        self.detector_type = detector_type
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.target_size = target_size

        # Initialize detectors
        # self._initialize_detectors ()
        _initialize_detectors ( self )
        """Initialize face detection models based on detector type."""
        try:
            if self.detector_type == "opencv":
                # Load OpenCV Haar Cascade classifier[1][2]
                haar_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                self.face_cascade = cv2.CascadeClassifier(haar_cascade_path)
                logger.info("OpenCV Haar Cascade detector initialized")

            elif self.detector_type == "mtcnn":
                # Initialize MTCNN detector[51][58][61]
                self.mtcnn_detector = MTCNN()
                logger.info("MTCNN detector initialized")

            elif self.detector_type == "dlib":
                # Initialize Dlib detector[3]
                self.dlib_detector = dlib.get_frontal_face_detector()
                logger.info("Dlib detector initialized")

            elif self.detector_type == "face_recognition":
                # Use face_recognition library[60]
                logger.info("Face recognition library detector initialized")

        except Exception as e:
            logger.error(f"Error initializing detector: {e}")
            raise

    # ==================== STEP 1: IMAGE INPUT ====================

    # ==================== STEP 2: PREPROCESSING ====================

    def preprocess_image(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Apply preprocessing steps to input image[9][12][57].

        Args:
            image: Input image

        Returns:
            Dictionary containing original and processed images
        """
        try:
            original_image = image.copy()
            processed_image = image.copy()

            # 1. Resize image while maintaining aspect ratio
            processed_image = self._resize_image(processed_image)

            # 2. Convert to grayscale for detection
            gray_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)[1][2][5]

            # 3. Apply histogram equalization for better contrast[9][12]
            equalized_image = cv2.equalizeHist(gray_image)

            # 4. Apply Gaussian blur for noise reduction[12]
            blurred_image = cv2.GaussianBlur(equalized_image, (3, 3), 0)

            # 5. Gamma correction for illumination normalization[12]
            gamma_corrected = _apply_gamma_correction(blurred_image, gamma=0.8)

            logger.info("Image preprocessing completed")

            return {
                'original': original_image,
                'processed': processed_image,
                'gray': gray_image,
                'equalized': equalized_image,
                'final': gamma_corrected
            }

        except Exception as e:
            logger.error(f"Error in preprocessing: {e}")
            raise

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        """Resize image to target size while maintaining aspect ratio[9]."""
        h, w = image.shape[:2]
        target_w, target_h = self.target_size

        # Calculate scaling factor
        scale = min(target_w / w, target_h / h)

        # Calculate new dimensions
        new_w = int(w * scale)
        new_h = int(h * scale)

        # Resize image
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        return resized

    # ==================== STEP 3: FACE DETECTION ====================

    def detect_faces(self, image: np.ndarray) -> List[Dict]:
        """
        Detect faces using the specified detector.

        Args:
            image: Input image (can be color or grayscale)

        Returns:
            List of detected faces with bounding boxes and confidence scores
        """
        try:
            # Preprocess image
            processed_images = self.preprocess_image(image)
            detection_image = processed_images['final']

            # Detect faces based on detector type
            if self.detector_type == "opencv":
                faces = self._detect_opencv_haar(detection_image, processed_images['processed'])
            elif self.detector_type == "mtcnn":
                faces = self._detect_mtcnn(processed_images['processed'])
            elif self.detector_type == "dlib":
                faces = self._detect_dlib(detection_image, processed_images['processed'])
            elif self.detector_type == "face_recognition":
                faces = _detect_face_recognition(processed_images['processed'])
            else:
                raise ValueError(f"Unsupported detector type: {self.detector_type}")

            logger.info(f"Detected {len(faces)} faces using {self.detector_type}")
            return faces

        except Exception as e:
            logger.error(f"Error in face detection: {e}")
            raise

    def _detect_opencv_haar(self, gray_image: np.ndarray, color_image: np.ndarray) -> List[Dict]:
        """Detect faces using OpenCV Haar Cascade[1][2][5]."""
        faces = self.face_cascade.detectMultiScale(
            gray_image,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )[1][2][14]

        detected_faces = []
        for (x, y, w, h) in faces:
            detected_faces.append({
                'bbox': [int(x), int(y), int(w), int(h)],
                'confidence': 0.9,  # Haar cascade doesn't provide confidence
                'landmarks': None,
                'detector': 'opencv_haar'
            })

        return detected_faces

    def _detect_mtcnn(self, image: np.ndarray) -> List[Dict]:
        """Detect faces using MTCNN[51][58][61]."""
        # Convert BGR to RGB for MTCNN
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = self.mtcnn_detector.detect_faces(rgb_image)

        detected_faces = []
        for result in results:
            if result['confidence'] >= self.confidence_threshold:
                bbox = result['box']
                detected_faces.append({
                    'bbox': [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])],
                    'confidence': float(result['confidence']),
                    'landmarks': result['keypoints'],
                    'detector': 'mtcnn'
                })

        return detected_faces

    def _detect_dlib(self, gray_image: np.ndarray, color_image: np.ndarray) -> List[Dict]:
        """Detect faces using Dlib[3]."""
        faces = self.dlib_detector(gray_image, 1)

        detected_faces = []
        for face in faces:
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            detected_faces.append({
                'bbox': [int(x), int(y), int(w), int(h)],
                'confidence': 0.9,  # Dlib doesn't provide confidence scores
                'landmarks': None,
                'detector': 'dlib'
            })

        return detected_faces

    # ==================== STEP 4: POST-PROCESSING ====================

    def post_process_detections(self, faces: List[Dict]) -> List[Dict]:
        """
        Apply post-processing to refine detections[29][32][35].

        Args:
            faces: List of detected faces

        Returns:
            List of refined face detections
        """
        try:
            if not faces:
                return faces

            # 1. Filter by confidence threshold
            filtered_faces = self._filter_by_confidence(faces)

            # 2. Apply Non-Maximum Suppression to remove duplicates
            nms_faces = self._apply_nms(filtered_faces)

            # 3. Refine bounding boxes
            refined_faces = _refine_bounding_boxes(nms_faces)

            # 4. Calculate additional metrics
            final_faces = _calculate_face_metrics(refined_faces)

            logger.info(f"Post-processing completed: {len(faces)} -> {len(final_faces)} faces")
            return final_faces

        except Exception as e:
            logger.error(f"Error in post-processing: {e}")
            raise

    def _filter_by_confidence(self, faces: List[Dict]) -> List[Dict]:
        """Filter faces by confidence threshold[31][34]."""
        return [face for face in faces if face['confidence'] >= self.confidence_threshold]

    def _apply_nms(self, faces: List[Dict]) -> List[Dict]:
        """Apply Non-Maximum Suppression to remove overlapping detections[32][35]."""
        if len(faces) <= 1:
            return faces

        # Convert to format expected by cv2.dnn.NMSBoxes
        boxes = []
        confidences = []

        for face in faces:
            bbox = face['bbox']
            boxes.append([bbox[0], bbox[1], bbox[2], bbox[3]])
            confidences.append(face['confidence'])

        # Apply NMS
        indices = cv2.dnn.NMSBoxes(
            boxes,
            confidences,
            self.confidence_threshold,
            self.nms_threshold
        )

        # Return filtered faces
        if len(indices) > 0:
            indices = indices.flatten()
            return [faces[i] for i in indices]
        else:
            return []

    # ==================== STEP 5: OUTPUT API ====================

    def format_detection_results(self, faces: List[Dict],
                                 include_landmarks: bool = True,
                                 include_metadata: bool = True) -> Dict:
        """
        Format detection results for API output[30][33][36].

        Args:
            faces: List of detected faces
            include_landmarks: Whether to include facial landmarks
            include_metadata: Whether to include metadata

        Returns:
            Formatted results dictionary
        """
        try:
            results = {
                'status': 'success',
                'timestamp': datetime.now().isoformat(),
                'face_count': len(faces),
                'faces': []
            }

            for face in faces:
                face_result = {
                    'face_id': face.get('face_id', 'unknown'),
                    'bounding_box': {
                        'x': face['bbox'][0],
                        'y': face['bbox'][1],
                        'width': face['bbox'][2],
                        'height': face['bbox'][3]
                    },
                    'confidence': round(face['confidence'], 4),
                    'detector_used': face.get('detector', self.detector_type)
                }

                # Add landmarks if available and requested
                if include_landmarks and face.get('landmarks'):
                    face_result['landmarks'] = face['landmarks']

                # Add metadata if requested
                if include_metadata:
                    face_result['metadata'] = {
                        'area': face.get('area', 0),
                        'aspect_ratio': round(face.get('aspect_ratio', 1.0), 2),
                        'center': face.get('center', [0, 0])
                    }

                results['faces'].append(face_result)

            logger.info(f"Formatted results for {len(faces)} faces")
            return results

        except Exception as e:
            logger.error(f"Error formatting results: {e}")
            return {
                'status': 'error',
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'face_count': 0,
                'faces': []
            }

    # ==================== MAIN PIPELINE METHOD ====================

    def process_image(self, input_source: Union[str, np.ndarray],
                      input_type: str = "file",
                      save_output: bool = False,
                      output_dir: str = "output") -> Dict:
        """
        Complete face detection pipeline.

        Args:
            input_source: Image file path, URL, base64 string, or numpy array
            input_type: Type of input ("file", "url", "base64", "array", "webcam")
            save_output: Whether to save output images and results
            output_dir: Directory to save outputs

        Returns:
            Detection results dictionary
        """
        try:
            # Step 1: Load image based on input type
            if input_type == "file":
                image = load_image_from_file(input_source)
            elif input_type == "url":
                image = load_image_from_url(input_source)
            elif input_type == "base64":
                image = load_image_from_base64(input_source)
            elif input_type == "array":
                image = input_source
            elif input_type == "webcam":
                image = capture_from_webcam(input_source if isinstance(input_source, int) else 0)
            else:
                raise ValueError(f"Unsupported input type: {input_type}")

            # Step 2 & 3: Detect faces (includes preprocessing)
            detected_faces = self.detect_faces(image)

            # Step 4: Post-process detections
            processed_faces = self.post_process_detections(detected_faces)

            # Step 5: Format results
            results = self.format_detection_results(processed_faces)

            # Optional: Save outputs
            if save_output:
                os.makedirs(output_dir, exist_ok=True)

                # Save results JSON
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                json_path = os.path.join(output_dir, f"detection_results_{timestamp}.json")
                save_results_to_file(results, json_path)

                # Save annotated image
                annotated_image = draw_detections(image, processed_faces)
                img_path = os.path.join(output_dir, f"annotated_image_{timestamp}.jpg")
                cv2.imwrite(img_path, annotated_image)

                results['output_files'] = {
                    'results_json': json_path,
                    'annotated_image': img_path
                }

            return results

        except Exception as e:
            logger.error(f"Error in face detection pipeline: {e}")
            return {
                'status': 'error',
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'face_count': 0,
                'faces': []
            }


# ==================== REST API WRAPPER ====================

from flask import Flask, request, jsonify
import tempfile


class FaceDetectionAPI:
    """REST API wrapper for face detection pipeline[30][33][36]."""

    def __init__(self, pipeline: FaceDetectionPipeline):
        self.app = Flask(__name__)
        self.pipeline = pipeline
        self._setup_routes()

    def _setup_routes(self):
        """Setup API routes."""

        @self.app.route('/api/detect', methods=['POST'])
        def detect_faces():
            """Face detection endpoint."""
            try:
                data = request.get_json()

                if 'image_url' in data:
                    results = self.pipeline.process_image(data['image_url'], input_type="url")
                elif 'image_base64' in data:
                    results = self.pipeline.process_image(data['image_base64'], input_type="base64")
                else:
                    return jsonify({'error': 'No image data provided'}), 400

                return jsonify(results)

            except Exception as e:
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/detect/upload', methods=['POST'])
        def upload_and_detect():
            """File upload and detection endpoint."""
            try:
                if 'file' not in request.files:
                    return jsonify({'error': 'No file provided'}), 400

                file = request.files['file']
                if file.filename == '':
                    return jsonify({'error': 'No file selected'}), 400

                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                    file.save(tmp_file.name)
                    results = self.pipeline.process_image(tmp_file.name, input_type="file")

                # Clean up temporary file
                os.unlink(tmp_file.name)

                return jsonify(results)

            except Exception as e:
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/health', methods=['GET'])
        def health_check():
            """Health check endpoint."""
            return jsonify({
                'status': 'healthy',
                'detector_type': self.pipeline.detector_type,
                'timestamp': datetime.now().isoformat()
            })

    def run(self, host='0.0.0.0', port=5000, debug=False):
        """Run the API server."""
        self.app.run(host=host, port=port, debug=debug)


# ==================== USAGE EXAMPLES ====================

def main():
    """Example usage of the face detection pipeline."""

    # Initialize pipeline with different detectors
    pipeline_opencv = FaceDetectionPipeline(detector_type="opencv", confidence_threshold=0.5)
    pipeline_mtcnn = FaceDetectionPipeline(detector_type="mtcnn", confidence_threshold=0.9)

    # Example 1: Process image from file
    print("=== Processing image from file ===")
    results = pipeline_opencv.process_image("test_image.jpg", input_type="file", save_output=True)
    print(f"Detected {results['face_count']} faces")

    # Example 2: Process image from URL
    print("\n=== Processing image from URL ===")
    image_url = "https://amiteshmaurya.com/images/amitesh-maurya-july25.jpg"
    results = pipeline_mtcnn.process_image(image_url, input_type="url")
    print(json.dumps(results, indent=2))

    # Example 3: Process webcam capture
    print("\n=== Processing webcam capture ===")
    results = pipeline_opencv.process_image(0, input_type="webcam", save_output=True)
    print(f"Detected {results['face_count']} faces from webcam")

    # Example 4: Start REST API server
    print("\n=== Starting REST API server ===")
    api = FaceDetectionAPI(pipeline_opencv)
    print("API server starting on http://localhost:5000")
    # api.run(port=5000, debug=True)  # Uncomment to run server


if __name__ == "__main__":
    main()
from face_pipeline import FaceDetectionPipeline, FaceDetectionAPI

if __name__ == "__main__":
    pipeline = FaceDetectionPipeline(detector_type="opencv")  # or use "mtcnn" as needed
    api = FaceDetectionAPI(pipeline)
    api.run(host="0.0.0.0", port=5000, debug=True)
