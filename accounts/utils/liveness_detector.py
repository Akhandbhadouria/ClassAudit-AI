"""
accounts/utils/liveness_detector.py
───────────────────────────────────
Biometric Anti-Spoofing & Active Presence Verification Engine.

Defeats:
1. 2D Printed Paper Photos
2. Digital Display / Phone Screen Replays
3. Static Cutouts and Frozen Camera Frames
4. Hand-Held Photo Wiggling

Uses:
- Google MediaPipe 468-point 3D FaceMesh
- Specular reflection & Screen Glare Analysis (HSV)
- Inter-frame Pixel Mean Absolute Difference (MAD)
- Dynamic Eye Aspect Ratio (EAR) & Mouth Aspect Ratio (MAR) variance
"""

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

# Lips / Mouth indices for Mouth Aspect Ratio (MAR)
# Upper lip center = 13, Lower lip center = 14
# Left mouth corner = 61, Right mouth corner = 291
UPPER_LIP = 13
LOWER_LIP = 14
MOUTH_LEFT = 61
MOUTH_RIGHT = 291

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


def _calc_mar(landmarks, w, h):
    """Calculate mouth aspect ratio (speech and jaw dynamics)."""
    top = np.array([landmarks[UPPER_LIP].x * w, landmarks[UPPER_LIP].y * h])
    bottom = np.array([landmarks[LOWER_LIP].x * w, landmarks[LOWER_LIP].y * h])
    left = np.array([landmarks[MOUTH_LEFT].x * w, landmarks[MOUTH_LEFT].y * h])
    right = np.array([landmarks[MOUTH_RIGHT].x * w, landmarks[MOUTH_RIGHT].y * h])

    vertical = np.linalg.norm(top - bottom)
    horizontal = np.linalg.norm(left - right)

    if horizontal == 0:
        return 0.0
    return float(vertical / horizontal)


def _extract_roi_crop(frame, face_box):
    """Safely extracts face ROI with a 15% margin."""
    if frame is None or face_box is None:
        return frame

    h, w = frame.shape[:2]
    top, right, bottom, left = face_box
    y1, y2 = min(top, bottom), max(top, bottom)
    x1, x2 = min(left, right), max(left, right)

    pad_h = int((y2 - y1) * 0.15)
    pad_w = int((x2 - x1) * 0.15)

    ny1 = max(0, y1 - pad_h)
    ny2 = min(h, y2 + pad_h)
    nx1 = max(0, x1 - pad_w)
    nx2 = min(w, x2 + pad_w)

    if ny2 > ny1 + 10 and nx2 > nx1 + 10:
        return frame[ny1:ny2, nx1:nx2]
    return frame


def detect_screen_texture_spoof(frame, face_box=None):
    """
    Detects 2D screen playback or printed photo attacks via frequency domain & color space analysis.
    Focuses specifically on the face region where phone glare and screen artifacts concentrate.
    """
    if frame is None:
        return True, "Invalid frame"

    try:
        roi = _extract_roi_crop(frame, face_box)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # 1. Glare / Bright specular reflection spots (common on phone glass and glossy photo paper)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]

        # Phone screen reflections: high brightness, low saturation specular hotspots
        specular_mask = (val > 250) & (sat < 35)
        glare_ratio = np.sum(specular_mask) / float(gray.size)
        if glare_ratio > 0.025:  # Over 2.5% of face ROI is specular glare
            return True, "Screen glare detected. Mobile screens and tablets are strictly prohibited."

        # 2. Laplacian Blur / Low-quality print check
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian_var < 20.0:
            return True, "Image too blurry or low quality."

        # 3. Color saturation distribution
        sat_std = np.std(sat)
        if sat_std < 10.0:
            return True, "Flat color detected (photo printout/screen suspected)."

        return False, "Natural texture"
    except Exception as e:
        logger.error("Texture spoof check error: %s", e)
        return False, "Error in texture check"


def analyze_face_frame(frame, face_box=None):
    """
    Extracts comprehensive geometric, 3D depth, landmark, and motion metrics using MediaPipe FaceMesh.
    
    Returns:
        dict with keys:
            - 'has_face': bool
            - 'ear': float (Eye Aspect Ratio)
            - 'mar': float (Mouth Aspect Ratio)
            - 'z_range': float (3D depth variance across facial features)
            - 'yaw': float (horizontal turn angle)
            - 'pitch': float (vertical head tilt angle)
            - 'face_crop': 64x64 normalized grayscale thumbnail
            - 'landmarks': raw landmarks list or None
    """
    empty_res = {
        'has_face': False, 'ear': 0.0, 'mar': 0.0, 'z_range': 0.0,
        'yaw': 0.0, 'pitch': 0.0, 'face_crop': None, 'landmarks': None
    }
    if frame is None:
        return empty_res

    mesh = _get_face_mesh()
    if mesh is None:
        return empty_res

    try:
        target_img = _extract_roi_crop(frame, face_box) if face_box is not None else frame
        h, w = target_img.shape[:2]

        rgb = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
        results = mesh.process(rgb)

        if results and results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark

            # 1. Eye Aspect Ratio (EAR)
            left_ear = _calc_ear(landmarks, LEFT_EYE_TOP, LEFT_EYE_BOTTOM, LEFT_EYE_LEFT, LEFT_EYE_RIGHT, w, h)
            right_ear = _calc_ear(landmarks, RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM, RIGHT_EYE_LEFT, RIGHT_EYE_RIGHT, w, h)
            avg_ear = float((left_ear + right_ear) / 2.0)

            # 2. Mouth Aspect Ratio (MAR)
            mar = _calc_mar(landmarks, w, h)

            # 3. 3D Depth (Z-variance)
            z_vals = [p.z for p in landmarks]
            z_range = float(max(z_vals) - min(z_vals))

            # 4. Head Yaw (horizontal turn)
            left_x = landmarks[LEFT_CHEEK].x * w
            right_x = landmarks[RIGHT_CHEEK].x * w
            nose_x = landmarks[NOSE_TIP].x * w
            mid_x = (left_x + right_x) / 2.0
            face_width = abs(right_x - left_x)
            yaw_ratio = float((nose_x - mid_x) / face_width) if face_width > 0 else 0.0

            # 5. Head Pitch (vertical tilt: forehead to nose vs nose to chin)
            fh_y = landmarks[FOREHEAD].y * h
            nose_y = landmarks[NOSE_TIP].y * h
            chin_y = landmarks[CHIN].y * h
            top_h = abs(nose_y - fh_y)
            bot_h = abs(chin_y - nose_y)
            pitch_ratio = float(top_h / bot_h) if bot_h > 0 else 1.0

            # 6. Normalized 64x64 grayscale face crop for temporal pixel delta
            gray = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
            face_crop = cv2.resize(gray, (64, 64)).astype(np.float32)

            return {
                'has_face': True,
                'ear': avg_ear,
                'mar': mar,
                'z_range': z_range,
                'yaw': yaw_ratio,
                'pitch': pitch_ratio,
                'face_crop': face_crop,
                'landmarks': landmarks
            }
    except Exception as e:
        logger.error("MediaPipe FaceMesh analysis error: %s", e)

    return empty_res


def check_active_presence_telemetry(frame, previous_state=None, face_box=None):
    """
    Active Presence Telemetry for Live Classroom Monitoring (evaluated every 5 seconds).
    
    Guarantees that a real, physically active living teacher is in front of the camera,
    strictly blocking:
    - Printed photo cards on a stand/desk
    - Phone or tablet screens held up to the webcam
    - Hand-held photo shaking
    - Pre-recorded frozen camera feeds

    Args:
        frame: Current BGR image from webcam
        previous_state: Dict storing metrics from previous verification checks
        face_box: (top, right, bottom, left) of the matched teacher face

    Returns:
        (is_live: bool, message: str, new_state: dict)
    """
    if frame is None:
        return False, "Empty frame received", previous_state

    # ── Step 1: Face ROI texture & screen glare check ──
    is_spoof, spoof_reason = detect_screen_texture_spoof(frame, face_box=face_box)
    if is_spoof:
        return False, spoof_reason, previous_state

    # ── Step 2: Facial geometric and landmark analysis ──
    metrics = analyze_face_frame(frame, face_box=face_box)
    if not metrics['has_face']:
        return False, "No face detected in live feed", previous_state

    # Eyes must be open in live view
    if metrics['ear'] < 0.12:
        return False, "Eyes closed or unnatural posture", previous_state

    # ── Step 3: Initialize or update rolling state ──
    prev_crops = previous_state.get('crop_history', []) if previous_state else []
    prev_ears = previous_state.get('ears_history', []) if previous_state else []
    prev_mars = previous_state.get('mars_history', []) if previous_state else []
    prev_yaws = previous_state.get('yaws_history', []) if previous_state else []
    consecutive_frozen = previous_state.get('consecutive_frozen', 0) if previous_state else 0

    new_ears = (prev_ears + [metrics['ear']])[-5:]
    new_mars = (prev_mars + [metrics['mar']])[-5:]
    new_yaws = (prev_yaws + [metrics['yaw']])[-5:]
    new_crops = (prev_crops + [metrics['face_crop']])[-3:]

    new_state = {
        'ear': metrics['ear'],
        'mar': metrics['mar'],
        'yaw': metrics['yaw'],
        'pitch': metrics['pitch'],
        'face_crop': metrics['face_crop'],
        'crop_history': new_crops,
        'ears_history': new_ears,
        'mars_history': new_mars,
        'yaws_history': new_yaws,
        'consecutive_frozen': consecutive_frozen,
    }

    # ── Step 4: Temporal anti-static verification ──
    if previous_state and previous_state.get('face_crop') is not None:
        prev_crop = previous_state['face_crop']
        curr_crop = metrics['face_crop']

        # A. Mean Absolute Difference (MAD) between consecutive face thumbnails
        mad = float(np.mean(np.abs(curr_crop - prev_crop)))

        # B. Internal landmark delta
        delta_ear = abs(metrics['ear'] - previous_state.get('ear', metrics['ear']))
        delta_mar = abs(metrics['mar'] - previous_state.get('mar', metrics['mar']))
        delta_yaw = abs(metrics['yaw'] - previous_state.get('yaw', metrics['yaw']))

        # Condition 1: Static Image Detection (Frozen on desk, stand, or screen)
        # Static photos with typical webcam sensor noise have MAD < 1.8
        is_frozen = (mad < 1.8)

        if is_frozen:
            consecutive_frozen += 1
            new_state['consecutive_frozen'] = consecutive_frozen
            if consecutive_frozen >= 1:
                return (
                    False,
                    "Static image detected — no biological movement. Live physical presence required.",
                    new_state
                )
        else:
            new_state['consecutive_frozen'] = 0

        # Condition 2: Hand-held Photo Wiggling Detection (Rolling 15-20s window)
        # If someone holds a photo card by hand, the overall image moves (MAD > 1.8),
        # but the internal facial geometry is 100% frozen:
        # The eyes never blink, and the mouth/lips never open.
        if len(new_ears) >= 3:
            ear_variance = float(np.ptp(new_ears))
            mar_variance = float(np.ptp(new_mars))
            has_blinked = any(e < 0.20 for e in new_ears) or (ear_variance >= 0.04)
            has_spoken_or_moved_lips = any(m > 0.06 for m in new_mars) or (mar_variance >= 0.02)
            has_head_motion = float(np.ptp(new_yaws)) >= 0.035

            if not (has_blinked or has_spoken_or_moved_lips or has_head_motion):
                return (
                    False,
                    "No natural biological activity (blink, speech, or head turn) detected over 15s. Please blink or move naturally.",
                    new_state
                )

    return True, "Live human verified", new_state


def check_single_frame_liveness(frame, min_ear=0.18, face_box=None):
    """
    Convenience wrapper for backward compatibility.
    Runs single-frame anti-spoofing and geometric validation.
    """
    if frame is None:
        return False, "Empty frame"

    is_spoof, spoof_reason = detect_screen_texture_spoof(frame, face_box=face_box)
    if is_spoof:
        return False, spoof_reason

    metrics = analyze_face_frame(frame, face_box=face_box)
    if not metrics['has_face']:
        return False, "No face detected in live feed"

    if metrics['ear'] < min_ear:
        return False, "Eyes closed or unnatural posture"

    return True, "Live human verified"


def check_liveness_challenge(frames, expected_challenge='blink'):
    """
    Multi-frame challenge-response verification for Login and Enrollment.
    Defeats static photos & pre-recorded video loops via temporal burst analysis.
    """
    if not frames or len(frames) < 3:
        return False, "Insufficient video frames for live verification. Please try again.", None

    frame_metrics = []
    frames_with_faces = []

    for frame in frames:
        is_spoof, spoof_reason = detect_screen_texture_spoof(frame)
        if is_spoof:
            return False, f"Anti-spoofing alert: {spoof_reason}", None

        metrics = analyze_face_frame(frame)
        if metrics['has_face']:
            frame_metrics.append(metrics)
            frames_with_faces.append(frame)

    if len(frame_metrics) < 3:
        return False, "No stable face detected. Please ensure good lighting and face the camera directly.", None

    ears = [m['ear'] for m in frame_metrics]
    yaws = [m['yaw'] for m in frame_metrics]

    best_idx = int(np.argmax(ears))
    best_frame = frames_with_faces[best_idx]

    # Dynamic Blink Detection
    min_ear = min(ears)
    max_ear = max(ears)
    ear_delta = max_ear - min_ear
    has_blinked = (min_ear < 0.20 and max_ear > 0.24) or (ear_delta >= 0.07)

    # Head micro-movement
    yaw_delta = max(yaws) - min(yaws)
    has_natural_movement = yaw_delta >= 0.035

    if expected_challenge == 'blink':
        if has_blinked:
            return True, "Liveness verified (natural blink detected).", best_frame
        elif has_natural_movement and max_ear >= 0.22:
            return True, "Liveness verified (biological movement confirmed).", best_frame
        else:
            return False, "Liveness check failed: Static photo or replay detected. Please blink your eyes naturally.", None

    if has_blinked or has_natural_movement:
        return True, "Liveness verified.", best_frame

    return False, "Anti-spoofing alert: Please blink naturally or turn your head slightly.", None
