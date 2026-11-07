import os
import cv2
import numpy as np
import onnxruntime as ort
from django.conf import settings
import logging

from .face_detector import detect_faces
from .face_matcher import match_face

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
    face_resized = cv2.resize(face_crop, (112, 112))
    rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
    tensor = np.transpose(rgb, (2, 0, 1)).astype(np.float32)
    # Normalize: (pixel - 127.5) / 128.0
    tensor = (tensor - 127.5) / 128.0
    return np.expand_dims(tensor, axis=0)


def extract_embedding_from_crop(face_crop):
    """Generates a 512-D L2-normalized embedding vector from a cropped face image."""
    if face_crop is None or face_crop.size == 0:
        return None
    try:
        blob = preprocess_face(face_crop)
        session = _get_onnx_session()
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name

        output = session.run([output_name], {input_name: blob})[0]
        embedding = output[0]

        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding
    except Exception as e:
        logger.error("Error generating face crop embedding: %s", e)
        return None


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
        return extract_embedding_from_crop(face_crop)

    except Exception as e:
        logger.error("Error generating face embedding: %s", e)
        return None


def get_all_embeddings(frame):
    """
    Detects all faces in the frame and returns their bounding boxes and embeddings.
    Used for multi-face presence detection (e.g. teacher + students in frame).
    
    Returns:
        list of tuples: [(face_location, embedding), ...]
    """
    if frame is None:
        return []

    results = []
    locations = detect_faces(frame)
    if not locations:
        return []

    h, w = frame.shape[:2]
    for loc in locations:
        top, right, bottom, left = loc
        pad_y = int((bottom - top) * 0.1)
        pad_x = int((right - left) * 0.1)
        y1 = max(0, top - pad_y)
        y2 = min(h, bottom + pad_y)
        x1 = max(0, left - pad_x)
        x2 = min(w, right + pad_x)

        face_crop = frame[y1:y2, x1:x2]
        emb = extract_embedding_from_crop(face_crop)
        if emb is not None:
            results.append((loc, emb))

    return results


def find_target_face_in_frame(frame, target_embedding, threshold=0.45):
    """
    Scans ALL detected faces in the frame (teacher + any students/colleagues)
    and checks if the target teacher's face is present.
    
    Args:
        frame: OpenCV BGR image
        target_embedding: Teacher's registered 512-D embedding
        threshold: Max distance cutoff (default 0.45)
        
    Returns:
        (found: bool, best_match_loc: tuple or None, total_faces: int)
    """
    all_faces = get_all_embeddings(frame)
    if not all_faces:
        return False, None, 0

    best_match = None
    best_dist = 1.0

    for loc, emb in all_faces:
        u = np.asarray(target_embedding, dtype=np.float32)
        v = np.asarray(emb, dtype=np.float32)
        
        sim = float(np.dot(u, v))
        dist = 1.0 - sim

        if dist < threshold and dist < best_dist:
            best_dist = dist
            best_match = loc

    if best_match is not None:
        return True, best_match, len(all_faces)

    return False, None, len(all_faces)


def average_embeddings(embeddings):
    """Averages and re-normalizes multiple embeddings."""
    if not embeddings:
        return None
    mean_emb = np.mean(embeddings, axis=0)
    norm = np.linalg.norm(mean_emb)
    return mean_emb / norm if norm > 0 else mean_emb
