import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

# MediaPipe Face Mesh landmark indices
# Left eye indices: top=159, bottom=145, left=33, right=133
LEFT_EYE_TOP = 159
LEFT_EYE_BOTTOM = 145
LEFT_EYE_LEFT = 33
LEFT_EYE_RIGHT = 133

# Right eye indices: top=386, bottom=374, left=362, right=263
RIGHT_EYE_TOP = 386
RIGHT_EYE_BOTTOM = 374
RIGHT_EYE_LEFT = 362
RIGHT_EYE_RIGHT = 263

# Key 3D facial depth points:
# Nose tip = 1, Forehead = 10, Chin = 152, Left cheek = 234, Right cheek = 454
NOSE_TIP = 1
FOREHEAD = 10
CHIN = 152
LEFT_CHEEK = 234
RIGHT_CHEEK = 454

_face_mesh = None


def _get_face_mesh():
    """Lazily load MediaPipe Face Mesh for fast landmark tracking."""
    global _face_mesh
    if _face_mesh is None:
        try:
            import mediapipe as mp
            if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
                _face_mesh = mp.solutions.face_mesh.FaceMesh(
                    static_image_mode=False,
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
        except Exception as e:
            logger.error("Error initializing MediaPipe FaceMesh: %s", e)
            _face_mesh = None
    return _face_mesh


def _calc_ear(landmarks, top_idx, bottom_idx, left_idx, right_idx, w, h):
    """Calculate eye aspect ratio."""
    top = np.array([landmarks[top_idx].x * w, landmarks[top_idx].y * h])
    bottom = np.array([landmarks[bottom_idx].x * w, landmarks[bottom_idx].y * h])
    left = np.array([landmarks[left_idx].x * w, landmarks[left_idx].y * h])
    right = np.array([landmarks[right_idx].x * w, landmarks[right_idx].y * h])

    vertical = np.linalg.norm(top - bottom)
    horizontal = np.linalg.norm(left - right)

    if horizontal == 0:
        return 0.0
    return float(vertical / horizontal)


def analyze_face_frame(frame):
    """
    Extracts comprehensive geometric, 3D depth, and landmark metrics using MediaPipe FaceMesh.
    
    Returns:
        dict with keys:
            - 'has_face': bool
            - 'ear': float (Eye Aspect Ratio)
            - 'z_range': float (3D depth variance across facial features)
            - 'yaw': float (horizontal turn angle)
            - 'landmarks': raw landmarks list or None
    """
    if frame is None:
        return {'has_face': False, 'ear': 0.0, 'z_range': 0.0, 'yaw': 0.0, 'landmarks': None}

    mesh = _get_face_mesh()
    h, w = frame.shape[:2]

    if mesh is not None:
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = mesh.process(rgb)
            if results and results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                
                # 1. EAR
                left_ear = _calc_ear(landmarks, LEFT_EYE_TOP, LEFT_EYE_BOTTOM, LEFT_EYE_LEFT, LEFT_EYE_RIGHT, w, h)
                right_ear = _calc_ear(landmarks, RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM, RIGHT_EYE_LEFT, RIGHT_EYE_RIGHT, w, h)
                avg_ear = (left_ear + right_ear) / 2.0

                # 2. 3D Depth (Z-variance)
                # Flat photo/screens have very compressed/artificial Z-gradients compared to real curved human faces
                z_vals = [p.z for p in landmarks]
                z_range = float(max(z_vals) - min(z_vals))

                # 3. Head Yaw (horizontal orientation: nose relative to midpoint between cheeks)
                left_x = landmarks[LEFT_CHEEK].x * w
                right_x = landmarks[RIGHT_CHEEK].x * w
                nose_x = landmarks[NOSE_TIP].x * w
                mid_x = (left_x + right_x) / 2.0
                face_width = abs(right_x - left_x)
                yaw_ratio = float((nose_x - mid_x) / face_width) if face_width > 0 else 0.0

                return {
                    'has_face': True,
                    'ear': avg_ear,
                    'z_range': z_range,
                    'yaw': yaw_ratio,
                    'landmarks': landmarks
                }
        except Exception as e:
            logger.error("MediaPipe FaceMesh analysis error: %s", e)

    return {'has_face': False, 'ear': 0.0, 'z_range': 0.0, 'yaw': 0.0, 'landmarks': None}


def detect_screen_texture_spoof(frame):
    """
    Detects 2D screen playback or printed photo attacks via frequency domain & color space analysis.
    
    Screens and phone displays exhibit:
    - High frequency moiré patterns or pixel grids
    - Excessive brightness saturation / glare reflections
    - Lower color diversity in skin tones compared to natural lighting
    """
    if frame is None:
        return True, "Invalid frame"

    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 1. Glare / Bright specular reflection spots (common on phone glass)
        _, thresh = cv2.threshold(gray, 252, 255, cv2.THRESH_BINARY)
        glare_ratio = np.sum(thresh == 255) / float(gray.size)
        if glare_ratio > 0.08:
            return True, "Screen glare detected. Please do not show phone screens."

        # 2. Laplacian Blur / Flatness check
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian_var < 35.0:
            return True, "Image too blurry or low quality."

        # 3. Color saturation check in HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        sat_std = np.std(hsv[:, :, 1])
        if sat_std < 12.0:
            return True, "Flat color detected (photo printout/screen suspected)."

        return False, "Natural texture"
    except Exception as e:
        logger.error("Texture spoof check error: %s", e)
        return False, "Error in texture check"


def check_liveness_challenge(frames, expected_challenge='blink'):
    """
    Multi-frame challenge-response verification to defeat static photos & pre-recorded video loops.
    
    Args:
        frames: Temporal sequence of BGR frames (6-10 frames captured over 1.5s - 2s).
        expected_challenge: 'blink' (natural eye closure) or 'head_turn' (slight head turn).
        
    Returns:
        (is_live: bool, message: str, best_frame: np.ndarray or None)
    """
    if not frames or len(frames) < 3:
        return False, "Insufficient video frames for live verification. Please try again.", None

    frame_metrics = []
    frames_with_faces = []

    for frame in frames:
        # Step A: Anti-spoof texture & screen check
        is_spoof, spoof_reason = detect_screen_texture_spoof(frame)
        if is_spoof:
            return False, f"Anti-spoofing alert: {spoof_reason}", None

        # Step B: 3D FaceMesh Landmark Analysis
        metrics = analyze_face_frame(frame)
        if metrics['has_face']:
            frame_metrics.append(metrics)
            frames_with_faces.append(frame)

    if len(frame_metrics) < 3:
        return False, "No stable face detected. Please ensure good lighting and face the camera directly.", None

    ears = [m['ear'] for m in frame_metrics]
    yaws = [m['yaw'] for m in frame_metrics]
    z_ranges = [m['z_range'] for m in frame_metrics]

    # Select best frame with open eyes and center pose for face recognition
    best_idx = int(np.argmax(ears))
    best_frame = frames_with_faces[best_idx]

    # Check 1: 3D Depth Check (screens are 2D flat surfaces; real faces have curved 3D depth)
    avg_z_range = np.mean(z_ranges)
    if avg_z_range < 0.08:
        return False, "2D spoof detected (flat photo or digital display). Live 3D face required.", None

    # Check 2: Dynamic Blink Detection
    min_ear = min(ears)
    max_ear = max(ears)
    ear_delta = max_ear - min_ear

    # If verifying a blink: EAR must visibly drop during the sequence
    # Real eye closure drops EAR below 0.19, while open eye is >= 0.25
    has_blinked = (min_ear < 0.20 and max_ear > 0.24) or (ear_delta >= 0.07)

    # Check 3: Head micro-movement / natural biological motion
    yaw_delta = max(yaws) - min(yaws)
    has_natural_movement = yaw_delta >= 0.035

    if expected_challenge == 'blink':
        if has_blinked:
            return True, "Liveness verified (natural blink detected).", best_frame
        elif has_natural_movement and max_ear >= 0.22:
            # Active live subject with micro head movement
            return True, "Liveness verified (biological movement confirmed).", best_frame
        else:
            return False, "Liveness check failed: Static photo or replay detected. Please blink your eyes naturally.", None

    # Default fallback
    if has_blinked or has_natural_movement:
        return True, "Liveness verified.", best_frame

    return False, "Anti-spoofing alert: Please blink naturally or turn your head slightly.", None


def check_single_frame_liveness(frame, min_ear=0.18):
    """
    Continuous passive check for live class monitoring (every 5 seconds).
    Blocks 2D photo screens and validates 3D human presence in <15ms.
    """
    if frame is None:
        return False, "Empty frame"

    # 1. Texture & Screen reflection check
    is_spoof, spoof_reason = detect_screen_texture_spoof(frame)
    if is_spoof:
        return False, spoof_reason

    # 2. 3D Face Geometry & Open Eyes
    metrics = analyze_face_frame(frame)
    if not metrics['has_face']:
        return False, "No face detected in live feed"

    # Reject if eyes closed or unnatural posture
    if metrics['ear'] < min_ear:
        return False, "Eyes closed or unnatural posture"

    # Reject if flat 2D display held up to camera
    if metrics['z_range'] < 0.08:
        return False, "2D screen/photo detected in front of camera"

    return True, "Live human verified"
