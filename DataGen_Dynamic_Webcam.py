import cv2
import mediapipe as mp
import numpy as np
import os
import random

# ==========================================
#             CONFIGURATION
# ==========================================

# 1. Output Directory
DATASET_DIR = "dataset_dynamic_aligned"

# 2. Capture Source (0 = Webcam)
CAPTURE_SOURCE = 0

# --- DYNAMIC CURVE CONFIGURATION ---
BASE_CLOSED_THRESH = 0.08   
BASE_OPEN_THRESH   = 0.21

# Sensitivity for saving both eyes. 
MAX_Z_DIFF_FOR_BOTH = 0.05

# Maximum Limits (Hard Caps)
MAX_CLOSED_THRESH = 0.120
MIN_OPEN_THRESH   = 0.180

# --- CENTER GAP VETO (UPDATED) ---
# We relaxed this from 0.05 to 0.15. 
# This means the center gap must be smaller than 15% of the eye width.
CENTER_CLOSURE_THRESH = 0.08 

# --- CENTER GAP OPEN GATE ---
# Prevents false "open" labels when the eyelids are actually touching (common during
# hard blinks and under head turns where EAR can be noisy).
OPEN_CENTER_MIN = 0.12

# --- HARD CENTER LOCK ---
# If the center gap is extremely small, force "Closed" regardless of EAR.
CENTER_HARD_LOCK = 0.06

# --- SAFETY OVERRIDE ---
# If EAR is lower than this, we force "Closed" regardless of the veto.
EAR_HARD_LOCK = 0.05

BLUR_THRESHOLD = 35.0

# Angle Curve Config
ANGLE_MAX_MAG = 1.2          
ANGLE_CLOSED_BOOST_MAX = 0.015  
ANGLE_OPEN_DROP_MAX   = 0.020  

# Scale Curve Config
SCALE_REF_NEAR  = 0.04
SCALE_REF_MID   = 0.11
SCALE_REF_FAR   = 0.22
SCALE_CLOSED_NEAR = 0.030
SCALE_CLOSED_MID  = 0.110
SCALE_CLOSED_FAR  = 0.125
OPEN_BAND_OFFSET = 0.09   

# Balancing
OPEN_EYE_SAVE_PROB = 0.15 
PATCH_SIZE = (64, 64)     
OPTIMAL_FACE_RATIO = SCALE_REF_MID

# ==========================================
#           SETUP & HELPERS
# ==========================================

# Create directories
for state in ["open", "closed"]:
    os.makedirs(os.path.join(DATASET_DIR, state), exist_ok=True)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Landmarks
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Center Eyelid Pairs (for Veto Check)
LEFT_CENTER_PAIR = [159, 145]
RIGHT_CENTER_PAIR = [386, 374]

NOSE_TIP = 1
CHIN = 152
FOREHEAD = 10

def get_3d_point(landmark, w, h):
    return np.array([landmark.x * w, landmark.y * h, landmark.z * w])

def compute_3D_EAR(landmarks, eye_indices, w, h):
    p1 = get_3d_point(landmarks[eye_indices[0]], w, h)
    p2 = get_3d_point(landmarks[eye_indices[1]], w, h)
    p3 = get_3d_point(landmarks[eye_indices[2]], w, h)
    p4 = get_3d_point(landmarks[eye_indices[3]], w, h)
    p5 = get_3d_point(landmarks[eye_indices[4]], w, h)
    p6 = get_3d_point(landmarks[eye_indices[5]], w, h)

    vertical_1 = np.linalg.norm(p2 - p6)
    vertical_2 = np.linalg.norm(p3 - p5)
    horizontal = np.linalg.norm(p1 - p4)

    if horizontal == 0: return 0.0, [], 0.0
    EAR = (vertical_1 + vertical_2) / (2.0 * horizontal)
    min_EAR = min(vertical_1, vertical_2) / horizontal
    pts_2d = [(int(p[0]), int(p[1])) for p in [p1, p2, p3, p4, p5, p6]]
    return EAR, pts_2d, min_EAR

def get_center_gap_ratio(landmarks, center_pair, corner_indices, w, h):
    """Calculates the gap between the exact center of the eyelids."""
    p_top = get_3d_point(landmarks[center_pair[0]], w, h)
    p_bot = get_3d_point(landmarks[center_pair[1]], w, h)
    
    p_left = get_3d_point(landmarks[corner_indices[0]], w, h)
    p_right = get_3d_point(landmarks[corner_indices[1]], w, h)
    
    center_dist = np.linalg.norm(p_top - p_bot)
    width_dist = np.linalg.norm(p_left - p_right)
    
    if width_dist == 0: return 1.0
    return center_dist / width_dist

def get_head_pose_ratios(landmarks):
    nose = landmarks[NOSE_TIP]
    left_outer = landmarks[33]
    right_outer = landmarks[263]
    eye_mid_x = (left_outer.x + right_outer.x) / 2
    face_width = abs(right_outer.x - left_outer.x)
    yaw_ratio = (nose.x - eye_mid_x) / (face_width + 1e-6)
    forehead = landmarks[FOREHEAD]
    chin = landmarks[CHIN]
    face_height = abs(chin.y - forehead.y)
    face_mid_y = (forehead.y + chin.y) / 2
    pitch_ratio = (nose.y - face_mid_y) / (face_height + 1e-6)
    return yaw_ratio, pitch_ratio

def get_face_scale_ratio(landmarks):
    left_outer = landmarks[33]
    right_outer = landmarks[263]
    return abs(right_outer.x - left_outer.x)

def get_eye_scale_ratio(landmarks, corner_indices):
    left_corner = landmarks[corner_indices[0]]
    right_corner = landmarks[corner_indices[1]]
    return abs(right_corner.x - left_corner.x)

def classify_eye_state(ear, ear_min, center_ratio, closed_thresh, open_thresh):
    """Returns 'open', 'closed', or None (no-save/ambiguous). Uses min-EAR to catch hard squints."""
    # Strong, geometry-based closure checks.
    if center_ratio < CENTER_HARD_LOCK:
        return "closed"
    if ear_min < EAR_HARD_LOCK:
        return "closed"

    # Regular closed band + squint veto.
    if ear_min < closed_thresh:
        if center_ratio < CENTER_CLOSURE_THRESH:
            return "closed"
        return None

    # Open band is gated by center gap to avoid false opens.
    if ear > open_thresh and ear_min > open_thresh and center_ratio > OPEN_CENTER_MIN:
        return "open"

    return None

# --- FIXED ANGLE FACTOR ---
def angle_factor(angle_mag):
    k = 1.5
    x = max(angle_mag, 0.0)
    raw_val = 1.0 - np.exp(-k * x)
    max_raw = 1.0 - np.exp(-k * ANGLE_MAX_MAG)
    if max_raw <= 0: return 0.0
    return np.clip(raw_val / max_raw, 0.0, 1.0)

def closed_thresh_from_scale(scale):
    s = float(max(scale, 0.0))
    if s <= SCALE_REF_NEAR: return SCALE_CLOSED_NEAR
    if s >= SCALE_REF_FAR: return SCALE_CLOSED_FAR
    if s <= SCALE_REF_MID:
        t = (s - SCALE_REF_NEAR) / (SCALE_REF_MID - SCALE_REF_NEAR)
        return SCALE_CLOSED_NEAR + t * (SCALE_CLOSED_MID - SCALE_CLOSED_NEAR)
    t = (s - SCALE_REF_MID) / (SCALE_REF_FAR - SCALE_REF_MID)
    return SCALE_CLOSED_MID + t * (SCALE_CLOSED_FAR - SCALE_CLOSED_MID)

def open_thresh_from_closed(closed_thresh):
    return np.clip(closed_thresh + OPEN_BAND_OFFSET, MIN_OPEN_THRESH, BASE_OPEN_THRESH)

def get_aligned_eye_crop(frame, landmarks, eye_indices, w, h):
    pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye_indices]
    x_vals = [p[0] for p in pts]
    y_vals = [p[1] for p in pts]
    cx = int(np.mean(x_vals))
    cy = int(np.mean(y_vals))
    
    sorted_x = sorted(pts, key=lambda k: k[0])
    angle = np.degrees(np.arctan2(sorted_x[-1][1] - sorted_x[0][1], sorted_x[-1][0] - sorted_x[0][0]))
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    
    eye_width = np.sqrt((sorted_x[-1][0] - sorted_x[0][0])**2 + (sorted_x[-1][1] - sorted_x[0][1])**2)
    crop_size = max(int(eye_width * 2.0), 32)
    rotated_frame = cv2.warpAffine(frame, M, (w, h))
    half = crop_size // 2
    return rotated_frame[max(cy-half, 0):min(cy+half, h), max(cx-half, 0):min(cx+half, w)]

def process_for_save(crop, size):
    """
    Grayscales, Checks for Blur, and Resizes.
    Returns None if the image is too blurry.
    """
    if crop.size == 0: return None
    
    try:
        # 1. Convert to Gray
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        # 2. BLUR CHECK (Inserted Here)
        # We calculate sharpness BEFORE resizing, as resizing can hide blur.
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        if blur_score < BLUR_THRESHOLD:
            # Optional: Print to console so you know it's happening
            # print(f"Rejected Blur: {blur_score:.1f}") 
            return None

        # 3. Resize and Return
        return cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
        
    except Exception as e:
        print(f"Error processing patch: {e}")
        return None

# ==========================================
#           MAIN CAPTURE LOOP
# ==========================================

cap = cv2.VideoCapture(CAPTURE_SOURCE)
frame_index = 0

print(f"Starting Webcam Capture... Output: {DATASET_DIR}")
print("Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    h, w, _ = frame.shape
    results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    status_text = "Searching..."
    status_color = (200, 200, 200)

    # Per-eye metrics (default)
    left_ear = 0.0
    right_ear = 0.0
    left_ear_min = 0.0
    right_ear_min = 0.0
    left_center_ratio = 1.0
    right_center_ratio = 1.0
    left_pts = []
    right_pts = []

    left_label = None
    right_label = None

    left_closed_thresh = BASE_CLOSED_THRESH
    left_open_thresh = BASE_OPEN_THRESH
    right_closed_thresh = BASE_CLOSED_THRESH
    right_open_thresh = BASE_OPEN_THRESH

    if results.multi_face_landmarks:
        face = results.multi_face_landmarks[0].landmark

        # 1. Head pose factor (still useful globally)
        yaw, pitch = get_head_pose_ratios(face)
        ang_f = angle_factor(abs(yaw) + abs(pitch))

        # 2. Per-eye thresholds (uses per-eye scale so head turns don't skew thresholds)
        left_scale = get_eye_scale_ratio(face, [33, 133])
        right_scale = get_eye_scale_ratio(face, [362, 263])

        left_closed_thresh = min(closed_thresh_from_scale(left_scale) + (ANGLE_CLOSED_BOOST_MAX * ang_f), MAX_CLOSED_THRESH)
        left_open_thresh = max(open_thresh_from_closed(left_closed_thresh) - (ANGLE_OPEN_DROP_MAX * ang_f), MIN_OPEN_THRESH)
        right_closed_thresh = min(closed_thresh_from_scale(right_scale) + (ANGLE_CLOSED_BOOST_MAX * ang_f), MAX_CLOSED_THRESH)
        right_open_thresh = max(open_thresh_from_closed(right_closed_thresh) - (ANGLE_OPEN_DROP_MAX * ang_f), MIN_OPEN_THRESH)

        # 3. Compute per-eye metrics
        left_ear, left_pts, left_ear_min = compute_3D_EAR(face, LEFT_EYE, w, h)
        right_ear, right_pts, right_ear_min = compute_3D_EAR(face, RIGHT_EYE, w, h)
        left_center_ratio = get_center_gap_ratio(face, LEFT_CENTER_PAIR, [33, 133], w, h)
        right_center_ratio = get_center_gap_ratio(face, RIGHT_CENTER_PAIR, [362, 263], w, h)

        # 4. Classify each eye independently
        left_label = classify_eye_state(left_ear, left_ear_min, left_center_ratio, left_closed_thresh, left_open_thresh)
        right_label = classify_eye_state(right_ear, right_ear_min, right_center_ratio, right_closed_thresh, right_open_thresh)

        # 5. Save per-eye (keeps balancing for open)
        saved_any = False
        for indices, suffix, eye_label in [
            (LEFT_EYE, "L", left_label),
            (RIGHT_EYE, "R", right_label),
        ]:
            if eye_label is None:
                continue
            if eye_label == "open" and random.random() >= OPEN_EYE_SAVE_PROB:
                continue

            crop_bgr = get_aligned_eye_crop(frame, face, indices, w, h)
            final_patch = process_for_save(crop_bgr, PATCH_SIZE)
            if final_patch is None:
                continue

            filename = f"{eye_label}_Webcam_{frame_index}_{PATCH_SIZE[0]}x{PATCH_SIZE[1]}_dyn_{suffix}.jpg"
            cv2.imwrite(os.path.join(DATASET_DIR, eye_label, filename), final_patch)
            saved_any = True

        # 6. Status text
        if saved_any:
            if (left_label == "closed") or (right_label == "closed"):
                status_color = (0, 0, 255)
            elif (left_label == "open") or (right_label == "open"):
                status_color = (0, 255, 0)
            status_text = f"SAVING  L:{left_label or '-'}  R:{right_label or '-'}"
        else:
            # Helpful debug when something is getting vetoed as squint
            if (left_ear < left_closed_thresh and left_center_ratio >= CENTER_CLOSURE_THRESH) or (
                right_ear < right_closed_thresh and right_center_ratio >= CENTER_CLOSURE_THRESH
            ):
                status_text = f"VETOED (Squint)  Lc:{left_center_ratio:.2f} Rc:{right_center_ratio:.2f}"
                status_color = (0, 165, 255)
            else:
                status_text = "Tracking..."
                status_color = (200, 200, 200)

        for pt in left_pts:
            cv2.circle(frame, pt, 1, (0, 255, 0), -1)
        for pt in right_pts:
            cv2.circle(frame, pt, 1, (0, 255, 0), -1)

    # UI Overlay
    cv2.putText(frame, f"EAR  L:{left_ear:.3f}/{left_ear_min:.3f}  R:{right_ear:.3f}/{right_ear_min:.3f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Center  L:{left_center_ratio:.3f}  R:{right_center_ratio:.3f}", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    
    cv2.putText(frame, status_text, (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

    frame_index += 1
    cv2.imshow("Webcam DataGen", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"): break

cap.release()
cv2.destroyAllWindows()