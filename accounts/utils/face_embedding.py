import os
import cv2
import numpy as np
import onnxruntime as ort
from django.conf import settings
import logging

from .face_detector import detect_faces

logger = logging.getLogger(__name__)

_onnx_session = None


def _get_onnx_session():
    """Lazily load ArcFace / MobileFaceNet ONNX model."""
    global _onnx_session
    if _onnx_session is None:
        models_dir = os.path.join(settings.BASE_DIR, "models")
        os.makedirs(models_dir, exist_ok=True)
        model_path = os.path.join(models_dir, "mobilefacenet.onnx")

        # Auto-download onnx model if not present locally
        if not os.path.exists(model_path):
            import urllib.request
            logger.info("Downloading lightweight face embedding model to %s...", model_path)
            model_url = "https://media.githubusercontent.com/media/onnx/models/main/validated/vision/body_analysis/arcface/model/arcfaceresnet100-8.onnx"
            try:
                urllib.request.urlretrieve(model_url, model_path)
                logger.info("Face model downloaded successfully.")
            except Exception as e:
                logger.error("Failed to auto-download face model: %s", e)
                raise FileNotFoundError(f"ONNX face model missing at {model_path} and auto-download failed.")

        # CPU-optimized options
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        _onnx_session = ort.InferenceSession(model_path, sess_options=opts, providers=['CPUExecutionProvider'])
    return _onnx_session


def preprocess_face(face_crop):
    """
    Preprocess cropped face to 112x112 normalized tensor for ArcFace.
    """
    # Resize to ArcFace standard input size (112, 112)
    face_resized = cv2.resize(face_crop, (112, 112))
    rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
    tensor = np.transpose(rgb, (2, 0, 1)).astype(np.float32)
    # Normalize: (pixel - 127.5) / 128.0
    tensor = (tensor - 127.5) / 128.0
    # Add batch dimension: (1, 3, 112, 112)
    return np.expand_dims(tensor, axis=0)


def get_embedding(frame, face_location=None):
    """
    Returns normalized 512-D face embedding using ArcFace ONNX.
    Zero dlib dependencies.
    
    Args:
        frame: BGR image (OpenCV format)
        face_location: Optional (top, right, bottom, left)
        
    Returns:
        1D numpy array of 512 floats, or None if detection fails.
    """
    if frame is None:
        return None

    try:
        h, w = frame.shape[:2]

        if face_location is None:
            locations = detect_faces(frame)
            if len(locations) != 1:
                return None
            face_location = locations[0]

        top, right, bottom, left = face_location

        # Safe bounds check with slight padding
        pad_y = int((bottom - top) * 0.1)
        pad_x = int((right - left) * 0.1)
        y1 = max(0, top - pad_y)
        y2 = min(h, bottom + pad_y)
        x1 = max(0, left - pad_x)
        x2 = min(w, right + pad_x)

        face_crop = frame[y1:y2, x1:x2]
        if face_crop.size == 0:
            return None

        # Preprocess and Run ONNX Inference
        blob = preprocess_face(face_crop)
        session = _get_onnx_session()
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name

        output = session.run([output_name], {input_name: blob})[0]  # Shape: (1, 512)
        embedding = output[0]

        # L2-normalize embedding vector
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    except Exception as e:
        logger.error("Error generating face embedding: %s", e)
        return None


def average_embeddings(embeddings):
    """Averages and re-normalizes multiple embeddings."""
    if not embeddings:
        return None
    mean_emb = np.mean(embeddings, axis=0)
    norm = np.linalg.norm(mean_emb)
    return mean_emb / norm if norm > 0 else mean_emb
