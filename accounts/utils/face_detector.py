import os
import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

_mp_face_detection = None
_detector = None


def _get_detector():
    """Lazily initialize MediaPipe face detector singleton."""
    global _mp_face_detection, _detector
    if _detector is None:
        try:
            import mediapipe as mp
            # Support both classic mediapipe solutions and modern bindings
            if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_detection"):
                _mp_face_detection = mp.solutions.face_detection
                _detector = _mp_face_detection.FaceDetection(
                    model_selection=0,  # 0 for short range (<2m, webcam)
                    min_detection_confidence=0.5
                )
            else:
                _detector = None
        except Exception as e:
            logger.error("Error initializing MediaPipe face detector: %s", e)
            _detector = None
    return _detector


def detect_faces(frame):
    """
    Detect faces using MediaPipe.
    
    Args:
        frame: BGR image from OpenCV
        
    Returns:
        List of bounding boxes in (top, right, bottom, left) format.
    """
    if frame is None:
        return []

    h, w = frame.shape[:2]
    detector = _get_detector()

    if detector is not None:
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = detector.process(rgb)
            locations = []
            if results.detections:
                for detection in results.detections:
                    box = detection.location_data.relative_bounding_box
                    xmin = max(0, int(box.xmin * w))
                    ymin = max(0, int(box.ymin * h))
                    width = int(box.width * w)
                    height = int(box.height * h)
                    xmax = min(w, xmin + width)
                    ymax = min(h, ymin + height)

                    # Return in (top, right, bottom, left) convention
                    locations.append((ymin, xmax, ymax, xmin))
            return locations
        except Exception as e:
            logger.error("MediaPipe detection error: %s", e)

    # Lightweight OpenCV Haar Cascade fallback if detector unavailable
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(cascade_path)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        return [(y, x + fw, y + fh, x) for (x, y, fw, fh) in faces]
    except Exception as e:
        logger.error("Fallback detection error: %s", e)
        return []
