import cv2
import numpy as np
import onnxruntime as ort
import time
import mediapipe as mp
import json
import win32com.client # Direct Windows SAPI access (more stable than pyttsx3)
import threading
import queue
import pythoncom # Required for stable TTS in threads on Windows
import textwrap
import winsound
import os
from datetime import datetime

# ==========================================
#           CONFIGURATION
# ==========================================
CNN_MODEL_PATH = "eye_state_mobilenet.onnx"
LSTM_MODEL_PATH = "blink_lstm.onnx"
LABEL_MAP_PATH = "lstm_word_map.json"
IMAGE_SIZE = (64, 64)

# --- NEW UI DIMENSIONS & LAYOUT ---
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720

# Panel Rects (x, y, w, h)
HEADER_HEIGHT = 70
VIDEO_PANEL = (20, 90, 800, 420)
TRANS_PANEL = (20, 520, 800, 180)
HISTORY_PANEL = (840, 90, 420, 610)

# Button Config (Header)
# Added Theme button to the left of Audio
BTN_THEME_RECT = (700, 20, 130, 40)
BTN_AUDIO_RECT = (840, 20, 130, 40)
BTN_MODE_RECT = (980, 20, 130, 40)
BTN_CMD_RECT = (1120, 20, 130, 40)

# --- THEMES (BGR Colors) ---
THEMES = {
    "dark": {
        "bg": (42, 23, 15),          # #0f172a (Slate 950)
        "panel_bg": (59, 41, 30),    # #1e293b (Slate 800)
        "panel_header": (85, 65, 51),# #334155 (Slate 700)
        "panel_border": (105, 85, 71),# #475569 (Slate 600)
        
        "text_main": (252, 250, 248), # #f8fafc (Slate 50)
        "text_muted": (184, 163, 148),# #94a3b8 (Slate 400)
        
        "accent_primary": (248, 189, 56), # #38bdf8 (Sky 400) - Blue
        "accent_success": (153, 211, 52), # #34d399 (Emerald 400) - Green
        "accent_warning": (36, 191, 251), # #fbbf24 (Amber 400) - Yellow
        "accent_danger": (113, 113, 248), # #f87171 (Red 400) - Red
        
        "overlay_bg": (30, 20, 10), # Dark overlay
        "btn_active_text": (42, 23, 15) # Dark text on active button
    },
    "light": {
        "bg": (245, 245, 245),       # Very Light Gray
        "panel_bg": (255, 255, 255), # White
        "panel_header": (230, 230, 230), # Light Gray Header
        "panel_border": (200, 200, 200), # Medium Gray Border
        
        "text_main": (40, 40, 40),   # Dark Gray Text
        "text_muted": (100, 100, 100), # Medium Gray Text
        
        "accent_primary": (200, 100, 0),   # Darker Blue for visibility
        "accent_success": (50, 180, 50),   # Darker Green
        "accent_warning": (0, 160, 220),   # Darker Amber
        "accent_danger": (50, 50, 220),    # Red
        
        "overlay_bg": (240, 240, 240), # Light overlay
        "btn_active_text": (255, 255, 255) # White text on active button
    }
}

current_theme_name = "dark"
COLORS = THEMES["dark"]

# Timing Thresholds
CHAR_PAUSE_THRESHOLD = 1.0  
WORD_PAUSE_THRESHOLD = 2.0  
# Word pause only starts after being idle for this long (no committed letters)
WORD_PAUSE_GRACE = 2.0
SENTENCE_SPEAK_BLINK_SEC = 2.0  # long blink gesture: speak the full current transcript
BATCH_COMMIT_BLINK_SEC = 2.0    # long blink gesture: decode buffered Morse
MIN_OPEN_STABILITY = 0.1 # seconds; debounce time to ignore blink glitches
DOT_DASH_THRESHOLD = 0.55  # split between dot and dash durations (sec)

# Evaluation / logging (SOP4 support)
EVAL_PROMPT_MODE = False           # If True: show prompts, score attempts, log metrics
EVAL_FORCE_CHAR_MODE = True        # If True: lock to CHAR mode during prompt eval
EVAL_PROMPTS_PATH = "eval_prompts.txt"  # one prompt per line

LOGGING_ENABLED = True
LOG_DIR = "session_logs"

# Head-gesture controls
# Uses a simple yaw ratio from FaceMesh landmarks.
# Note: depending on camera mirroring, you may need to flip RIGHT/LEFT.
HEAD_BACKSPACE_ENABLED = True
HEAD_BACKSPACE_DIRECTION = "RIGHT"  # "RIGHT" or "LEFT"
HEAD_BACKSPACE_YAW_THRESHOLD = 0.25  # higher = more turn required
HEAD_BACKSPACE_YAW_RESET = 0.20      # hysteresis reset threshold (must return below this)
HEAD_BACKSPACE_HOLD_SEC = 0.15       # must hold the turn for this long
HEAD_BACKSPACE_COOLDOWN_SEC = 1   # minimum time between backspaces

# After a backspace (keyboard or head gesture), ignore blink inputs briefly.
POST_BACKSPACE_INPUT_BLOCK_SEC = 0.75

# CNN output handling
# The exported MobileNet model emits a single logit: logit > 0 -> open.
# Adjust this threshold if you later want a stricter closed detection.
OPEN_PROB_THRESHOLD = 0.5

# ==========================================
#           TEXT TO SPEECH SETUP
# ==========================================
speech_queue = queue.Queue()

# Audio output mode
# - TTS: speak words via SAPI
# - MUTE: no audio
# - BEEP: play short/long beeps matching Morse input (dot/dash)
current_audio_mode = "TTS"  # "TTS" | "MUTE" | "BEEP"

# TTS speed (SAPI Rate range is roughly -10..10; positive is faster)
TTS_RATE = 4

# Morse beep settings (Windows)
BEEP_FREQ_HZ = 880
DOT_BEEP_MS = 120
DASH_BEEP_MS = 360
BEEP_GAP_MS = 60

beep_queue = queue.Queue()

def beep_worker():
    """Background thread that plays queued beeps so the main loop stays real-time."""
    while True:
        item = beep_queue.get()
        if item is None:
            break
        try:
            freq_hz, dur_ms = item
            winsound.Beep(int(freq_hz), int(dur_ms))
            if BEEP_GAP_MS > 0:
                time.sleep(BEEP_GAP_MS / 1000.0)
        except Exception as e:
            print(f"Beep error: {e}")
        finally:
            beep_queue.task_done()

beep_thread = threading.Thread(target=beep_worker, daemon=True)
beep_thread.start()

def speech_worker():
    """Persistent background thread that handles all speech requests"""
    # CRITICAL: Initialize COM for this thread on Windows
    pythoncom.CoInitialize()
    
    try:
        # Initialize the native Windows voice engine directly
        speaker = win32com.client.Dispatch("SAPI.SpVoice")
        # Speed up speech for words, letters, and dot/dash cues
        try:
            speaker.Rate = TTS_RATE
        except Exception:
            pass
    except Exception as e:
        print(f"SAPI Initialization Error: {e}")
        return
    
    while True:
        text = speech_queue.get()
        if text is None: break 
        
        try:
            # Direct Speak call is much faster and more stable in threads
            speaker.Speak(text)
        except Exception as e:
            print(f"Speech error: {e}")
        
        speech_queue.task_done()
    
    pythoncom.CoUninitialize()

speech_thread = threading.Thread(target=speech_worker, daemon=True)
speech_thread.start()

def speak_text(text):
    """Adds text to the speech queue"""
    if current_audio_mode != "TTS":
        return
    if text and text != "?":
        speech_queue.put(text)

def play_morse_beep(duration_sec: float):
    """Enqueue a dot/dash beep based on the same threshold used for rendering."""
    if current_audio_mode != "BEEP":
        return
    is_dot = duration_sec < DOT_DASH_THRESHOLD
    beep_queue.put((BEEP_FREQ_HZ, DOT_BEEP_MS if is_dot else DASH_BEEP_MS))


def speak_unit(duration_sec: float):
    """Optional spoken feedback for dot/dash when in TTS mode."""
    if current_audio_mode != "TTS":
        return
    is_dot = duration_sec < DOT_DASH_THRESHOLD
    speak_text("dot" if is_dot else "dash")


def _normalize_for_eval(text: str) -> str:
    # Normalize whitespace/case to reduce trivial mismatches
    return " ".join((text or "").strip().lower().split())


def _levenshtein_distance(a: str, b: str) -> int:
    # Classic DP; good enough for short prompts.
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            ins = curr[j - 1] + 1
            dele = prev[j] + 1
            sub = prev[j - 1] + (0 if ca == cb else 1)
            curr.append(min(ins, dele, sub))
        prev = curr
    return prev[-1]


def _cer(pred: str, truth: str) -> float:
    truth_n = _normalize_for_eval(truth)
    pred_n = _normalize_for_eval(pred)
    if not truth_n:
        return 0.0 if not pred_n else 1.0
    dist = _levenshtein_distance(pred_n, truth_n)
    return dist / max(1, len(truth_n))


def _wer(pred: str, truth: str) -> float:
    truth_words = _normalize_for_eval(truth).split()
    pred_words = _normalize_for_eval(pred).split()
    if not truth_words:
        return 0.0 if not pred_words else 1.0

    # Word-level Levenshtein
    prev = list(range(len(pred_words) + 1))
    for i, tw in enumerate(truth_words, start=1):
        curr = [i]
        for j, pw in enumerate(pred_words, start=1):
            ins = curr[j - 1] + 1
            dele = prev[j] + 1
            sub = prev[j - 1] + (0 if tw == pw else 1)
            curr.append(min(ins, dele, sub))
        prev = curr
    return prev[-1] / max(1, len(truth_words))


MORSE_TABLE = {
    ".-": "A",
    "-...": "B",
    "-.-.": "C",
    "-..": "D",
    ".": "E",
    "..-.": "F",
    "--.": "G",
    "....": "H",
    "..": "I",
    ".---": "J",
    "-.-": "K",
    ".-..": "L",
    "--": "M",
    "-.": "N",
    "---": "O",
    ".--.": "P",
    "--.-": "Q",
    ".-.": "R",
    "...": "S",
    "-": "T",
    "..-": "U",
    "...-": "V",
    ".--": "W",
    "-..-": "X",
    "-.--": "Y",
    "--..": "Z",
    "-----": "0",
    ".----": "1",
    "..---": "2",
    "...--": "3",
    "....-": "4",
    ".....": "5",
    "-....": "6",
    "--...": "7",
    "---..": "8",
    "----.": "9",
}

# Fixed Morse codes for each LSTM word index (code is immutable; word text is editable)
WORD_COMMAND_CODES = {
    0: ".-..-.",
    1: ".----.",
    2: "-.--.",
    3: "-.--.-",
    4: ".-.-.",
    5: "--..--",
    6: "-....-",
    7: ".-.-.-",
    8: "-..-.",
    9: "-----",
    10: ".----",
    11: "..---",
    12: "...--",
    13: "....-",
    14: ".....",
    15: "-....",
    16: "--...",
    17: "---..",
    18: "----.",
    19: "---...",
    20: "-...-",
    21: "..--..",
    22: ".--.-.",
    23: ".-",
    24: "-...",
    25: "-.-.",
    26: "-..",
    27: ".",
    28: "..-.",
    29: "--.",
    30: "....",
    31: "..",
    32: ".---",
    33: "-.-",
    34: ".-..",
    35: "--",
    36: "-.",
    37: "---",
    38: ".--.",
    39: "--.-",
    40: ".-.",
    41: "...",
    42: "-",
    43: "..-",
    44: "...-",
    45: ".--",
    46: "-..-",
    47: "-.--",
    48: "--..",
}


def decode_morse_tokens(tokens: list[str]) -> str:
    """Decode a list of morse letter tokens into text.

    Tokens are typically like ['.-', '.-.', '---'].
    Use '/' token to indicate a word boundary.
    """
    out: list[str] = []
    for tok in tokens:
        t = (tok or "").strip()
        if not t:
            continue
        if t in ("/", "|"):
            # Word boundary
            if out and out[-1] != " ":
                out.append(" ")
            continue
        out.append(MORSE_TABLE.get(t, "?"))

    # Collapse any repeated spaces
    return "".join(out).replace("  ", " ").strip()

# ==========================================
#           LOAD MODELS
# ==========================================
print("Loading models...")

# Load BOTH maps for switching
try:
    with open("lstm_word_map.json", "r") as f:
        word_map = json.load(f)
        idx_to_word = {v: k for k, v in word_map.items()}
except Exception as e:
    print(f"Error loading word map: {e}")
    idx_to_word = {}

try:
    with open("lstm_label_map.json", "r") as f:
        char_map = json.load(f)
        idx_to_char = {v: k for k, v in char_map.items()}
except Exception as e:
    print(f"Error loading label map: {e}")
    idx_to_char = {}

# State for Mode Switching
current_mode = "WORD"  # WORD | CHAR | BUFFER | LETTERS
idx_to_label = idx_to_word
show_word_commands = False
pending_click = None

def toggle_audio_mode():
    global current_audio_mode
    modes = ["TTS", "MUTE", "BEEP"]
    current_audio_mode = modes[(modes.index(current_audio_mode) + 1) % len(modes)]
    print(f"Switched to {current_audio_mode} audio")

def toggle_mode():
    global current_mode, idx_to_label, show_word_commands
    modes = ["WORD", "CHAR", "BUFFER", "LETTERS"]
    current_mode = modes[(modes.index(current_mode) + 1) % len(modes)]
    idx_to_label = idx_to_char if current_mode != "WORD" else idx_to_word
    if current_mode != "WORD":
        show_word_commands = False
    print(f"Switched to {current_mode} mode")

# --- NEW UI HELPERS ---
def toggle_theme():
    global COLORS, current_theme_name
    if current_theme_name == "dark":
        current_theme_name = "light"
        COLORS = THEMES["light"]
    else:
        current_theme_name = "dark"
        COLORS = THEMES["dark"]
    print(f"Switched to {current_theme_name} theme")

def is_point_in_rect(point, rect):
    px, py = point
    rx, ry, rw, rh = rect
    return rx <= px <= rx + rw and ry <= py <= ry + rh

def draw_panel(canvas, rect, title=None, badge=None):
    x, y, w, h = rect
    # Panel BG
    cv2.rectangle(canvas, (x, y), (x+w, y+h), COLORS["panel_bg"], -1)
    # Panel Border
    cv2.rectangle(canvas, (x, y), (x+w, y+h), COLORS["panel_border"], 1)
    
    if title:
        # Header BG
        header_h = 35
        cv2.rectangle(canvas, (x, y), (x+w, y+header_h), COLORS["panel_header"], -1)
        cv2.rectangle(canvas, (x, y), (x+w, y+header_h), COLORS["panel_border"], 1)
        # Title Text
        cv2.putText(canvas, title.upper(), (x+15, y+23), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["text_muted"], 1, cv2.LINE_AA)
        
        if badge:
            # Badge pill
            (tw, th), _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            bx = x + w - tw - 25
            by = y + 8
            cv2.rectangle(canvas, (bx, by), (bx+tw+10, by+20), (50, 60, 20), -1) # Dark green pill
            cv2.rectangle(canvas, (bx, by), (bx+tw+10, by+20), COLORS["accent_success"], 1)
            cv2.putText(canvas, badge, (bx+5, by+14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLORS["accent_success"], 1, cv2.LINE_AA)

def draw_button(canvas, rect, text, active=False, color=None):
    x, y, w, h = rect
    
    # Fill
    if active:
        bg = COLORS["accent_primary"]
        txt_col = COLORS["btn_active_text"] # Adaptive text color
        border_col = COLORS["accent_primary"]
    else:
        bg = COLORS["bg"]
        txt_col = COLORS["text_muted"]
        border_col = COLORS["panel_border"]
    
    if color: # Override color for specific modes
        bg = color
        txt_col = COLORS["btn_active_text"]
    
    cv2.rectangle(canvas, (x, y), (x+w, y+h), bg, -1)
    cv2.rectangle(canvas, (x, y), (x+w, y+h), border_col, 1)
    
    # Text
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, 0.5, 1)
    tx = x + (w - tw) // 2
    ty = y + (h + th) // 2
    cv2.putText(canvas, text, (tx, ty), font, 0.5, txt_col, 1, cv2.LINE_AA)

def draw_progress_bar(canvas, rect, progress, threshold=DOT_DASH_THRESHOLD):
    x, y, w, h = rect
    # Track
    cv2.rectangle(canvas, (x, y), (x+w, y+h), (70, 70, 70), -1)
    
    # Fill
    fill_w = int(w * min(progress, 1.0))
    fill_col = COLORS["accent_primary"]
    if progress > threshold:
        fill_col = COLORS["accent_warning"] # Amber for dash
        
    if fill_w > 0:
        cv2.rectangle(canvas, (x, y), (x+fill_w, y+h), fill_col, -1)
    
    # Threshold marker
    marker_x = x + int(w * threshold)
    cv2.line(canvas, (marker_x, y-2), (marker_x, y+h+2), (0,0,0), 2)

def wrap_text_for_width(text: str, max_width: int, font, font_scale: float, thickness: int) -> list[str]:
    """Wrap text into multiple lines so it fits within max_width for the given font settings."""
    words = text.split()
    lines: list[str] = []
    current = ""
    for w in words:
        candidate = w if not current else f"{current} {w}"
        (cw, _), _ = cv2.getTextSize(candidate, font, font_scale, thickness)
        if cw <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = w
    if current:
        lines.append(current)
    return lines if lines else ["_"]

def mouse_callback(event, x, y, flags, param):
    global pending_click, show_word_commands
    if event == cv2.EVENT_LBUTTONDOWN:
        pending_click = (x, y)
        if is_point_in_rect((x, y), BTN_THEME_RECT): toggle_theme()
        elif is_point_in_rect((x, y), BTN_AUDIO_RECT): toggle_audio_mode()
        elif is_point_in_rect((x, y), BTN_MODE_RECT): toggle_mode()
        elif current_mode == "WORD" and is_point_in_rect((x, y), BTN_CMD_RECT): 
            show_word_commands = not show_word_commands

# Load models with flexible provider selection
def create_session(path):
    try:
        return ort.InferenceSession(path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    except:
        return ort.InferenceSession(path, providers=["CPUExecutionProvider"])

lstm_sess = create_session(LSTM_MODEL_PATH)
lstm_inputs = {i.name: i for i in lstm_sess.get_inputs()}
lstm_providers = lstm_sess.get_providers()
lstm_call_count = 0
last_lstm_debug = {
    "ts": 0.0,
    "mode": "",
    "seq_len": 0,
    "pred_idx": None,
    "pred_label": "",
    "blink_seq": [],
    "provider": lstm_providers,
}

ort_session = create_session(CNN_MODEL_PATH)
cnn_input_name = ort_session.get_inputs()[0].name

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
LEFT_EYE = [33, 160, 158, 133, 153, 144]

NOSE_TIP = 1
LEFT_EYE_OUTER = 33
RIGHT_EYE_OUTER = 263

# ==========================================
#           HELPER FUNCTIONS
# ==========================================
def preprocess_cnn(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, IMAGE_SIZE)
    norm = resized.astype(np.float32) / 255.0
    norm = (norm - 0.5) / 0.5
    return np.expand_dims(np.expand_dims(norm, axis=0), axis=0)

def predict_letter(raw_durations, mode_name: str):
    """Run the blink-duration LSTM and capture debug context so we can verify it runs."""
    global lstm_call_count, last_lstm_debug
    if not raw_durations:
        return ""

    # Shape to (1, T, 1) as expected by LSTM
    arr = np.array([[min(d, 2.0) for d in raw_durations]], dtype=np.float32)[:, :, None]
    feed = {"input": arr}
    if "lengths" in lstm_inputs:
        feed["lengths"] = np.array([arr.shape[1]], dtype=np.int64)
    
    logits = lstm_sess.run(None, feed)[0]
    pred_idx = int(np.argmax(logits, axis=1)[0])
    pred = idx_to_label.get(pred_idx, "?")

    lstm_call_count += 1
    last_lstm_debug = {
        "ts": time.time(),
        "mode": mode_name,
        "seq_len": int(arr.shape[1]),
        "pred_idx": pred_idx,
        "pred_label": pred,
        "blink_seq": [round(float(d), 3) for d in raw_durations],
        "provider": lstm_providers,
    }

    # Console hint so it's obvious the LSTM is executing
    print(f"LSTM inference #{lstm_call_count} [{mode_name}] -> {pred} (idx {pred_idx}) seq_len={arr.shape[1]} durations={last_lstm_debug['blink_seq']}")
    return pred

def get_eye_crop(frame, landmarks, w, h):
    pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in LEFT_EYE]
    x_vals, y_vals = [p[0] for p in pts], [p[1] for p in pts]
    cx, cy = (min(x_vals) + max(x_vals)) // 2, (min(y_vals) + max(y_vals)) // 2
    max_dim = max(max(x_vals) - min(x_vals), max(y_vals) - min(y_vals))
    side = int(max_dim * 2.0) # Slightly larger crop for better features
    half = side // 2
    x1, y1 = max(cx - half, 0), max(cy - half, 0)
    x2, y2 = min(cx + half, w), min(cy + half, h)
    return frame[y1:y2, x1:x2]

def get_yaw_ratio(landmarks):
    """Approx head yaw estimate: nose horizontal offset from eye-mid, normalized by face width."""
    nose = landmarks[NOSE_TIP]
    left_outer = landmarks[LEFT_EYE_OUTER]
    right_outer = landmarks[RIGHT_EYE_OUTER]

    eye_mid_x = (left_outer.x + right_outer.x) / 2.0
    face_width = abs(right_outer.x - left_outer.x)
    if face_width < 1e-6:
        return 0.0
    return (nose.x - eye_mid_x) / face_width

# ==========================================
#           MAIN LOOP
# ==========================================
def main():
    global show_word_commands, pending_click, idx_to_label
    current_cam_idx = 0
    cap = cv2.VideoCapture(current_cam_idx)
    is_closed = False
    closed_start_time = 0
    potential_open_start = None
    last_open_time = time.time()
    last_token_time = last_open_time  # last time we committed a letter/word
    current_blink_sequence = []
    last_blink_duration = 0.0
    decoded_history = [] # List of strings instead of single string
    selected_command_idx = None
    editing_active = False
    edit_buffer = ""
    row_hitboxes = []

    # Word/sentence assembly (CHAR mode builds words from letters)
    current_word = ""
    transcript_words = []  # list[str]

    # Buffered Morse mode state
    buffered_tokens: list[str] = []  # completed Morse tokens (".-", "--" ... or "/")
    current_morse_token = ""       # in-progress token while blinking in BUFFER mode

    input_block_until = 0.0

    last_sentence_speak_time = 0.0

    # --- Session logging / evaluation state ---
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_events_f = None
    session_summary_path = None

    if LOGGING_ENABLED or EVAL_PROMPT_MODE:
        os.makedirs(LOG_DIR, exist_ok=True)
        log_events_path = os.path.join(LOG_DIR, f"mc_session_{session_id}.jsonl")
        session_summary_path = os.path.join(LOG_DIR, f"mc_session_{session_id}_summary.json")
        log_events_f = open(log_events_path, "a", encoding="utf-8")

        summary = {
            "session_id": session_id,
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "config": {
                "CNN_MODEL_PATH": CNN_MODEL_PATH,
                "LSTM_MODEL_PATH": LSTM_MODEL_PATH,
                "CHAR_PAUSE_THRESHOLD": CHAR_PAUSE_THRESHOLD,
                "WORD_PAUSE_THRESHOLD": WORD_PAUSE_THRESHOLD,
                "WORD_PAUSE_GRACE": WORD_PAUSE_GRACE,
                "SENTENCE_SPEAK_BLINK_SEC": SENTENCE_SPEAK_BLINK_SEC,
                "MIN_OPEN_STABILITY": MIN_OPEN_STABILITY,
                "OPEN_PROB_THRESHOLD": OPEN_PROB_THRESHOLD,
            },
            "eval": {
                "enabled": EVAL_PROMPT_MODE,
                "force_char_mode": EVAL_FORCE_CHAR_MODE,
                "prompts_path": EVAL_PROMPTS_PATH,
                "attempts": [],
            },
        }
    else:
        summary = None

    eval_prompts = []
    eval_prompt_idx = 0
    if EVAL_PROMPT_MODE:
        try:
            if os.path.exists(EVAL_PROMPTS_PATH):
                with open(EVAL_PROMPTS_PATH, "r", encoding="utf-8") as f:
                    eval_prompts = [line.strip() for line in f.readlines() if line.strip()]
        except Exception as e:
            print(f"Prompt load error: {e}")
            eval_prompts = []

        # Fallback prompts if file missing/empty
        if not eval_prompts:
            eval_prompts = [
                "hello world",
                "this is a test",
                "openai",
                "morse decoder",
            ]

        if summary is not None:
            summary["eval"]["prompts"] = eval_prompts

        if EVAL_FORCE_CHAR_MODE:
            # Lock to CHAR mode for consistent scoring across subjects.
            global current_mode, idx_to_label
            current_mode = "CHAR"
            idx_to_label = idx_to_char

    # Head-turn backspace state
    head_turn_start = None
    head_backspace_armed = True
    head_backspace_cooldown_until = 0.0

    def handle_backspace(now_ts: float):
        """Delete last entry: BUFFER edits Morse, otherwise edits transcript."""
        nonlocal current_word, transcript_words, last_token_time, input_block_until, buffered_tokens, current_morse_token, current_blink_sequence

        if current_mode == "BUFFER":
            if current_morse_token:
                current_morse_token = current_morse_token[:-1]
                current_blink_sequence = []
                last_token_time = now_ts
                input_block_until = max(input_block_until, now_ts + POST_BACKSPACE_INPUT_BLOCK_SEC)
                return
            if buffered_tokens:
                buffered_tokens.pop()
                current_blink_sequence = []
                last_token_time = now_ts
                input_block_until = max(input_block_until, now_ts + POST_BACKSPACE_INPUT_BLOCK_SEC)
                return
            # Fall through to transcript editing if buffer is empty (post-decode edits)

        if current_mode in ("CHAR", "LETTERS"):
            if current_word:
                current_word = current_word[:-1]
            elif transcript_words:
                last = transcript_words[-1]
                if len(last) <= 1:
                    transcript_words.pop()
                else:
                    transcript_words[-1] = last[:-1]
        else:
            if transcript_words:
                transcript_words.pop()
            elif current_word:
                current_word = ""

        # Prevent immediate auto-commit after editing
        last_token_time = now_ts

        # Prevent accidental dot/dash inputs immediately after a backspace gesture.
        input_block_until = max(input_block_until, now_ts + POST_BACKSPACE_INPUT_BLOCK_SEC)

    def commit_word_edit():
        nonlocal editing_active, edit_buffer, selected_command_idx
        global idx_to_label
        if selected_command_idx is None:
            return
        new_word = edit_buffer.strip()
        if not new_word:
            editing_active = False
            return

        old_word = idx_to_word.get(selected_command_idx, "")
        if new_word == old_word:
            editing_active = False
            return

        # Drop any existing entries that would conflict, then set new text for this idx
        for key in list(word_map.keys()):
            if word_map[key] == selected_command_idx or key == new_word:
                word_map.pop(key, None)

        word_map[new_word] = selected_command_idx
        idx_to_word[selected_command_idx] = new_word
        if current_mode == "WORD":
            idx_to_label = idx_to_word

        try:
            with open(LABEL_MAP_PATH, "w", encoding="utf-8") as f:
                json.dump(word_map, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Failed to save word map: {e}")

        editing_active = False
    
    print("System Ready! SAPI TTS & Timers Enabled.")
    
    cv2.namedWindow("Blink Morse Decoder", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Blink Morse Decoder", WINDOW_WIDTH, WINDOW_HEIGHT)
    cv2.setMouseCallback("Blink Morse Decoder", mouse_callback)

    while True:
        if not cap.isOpened():
            print(f"Camera {current_cam_idx} failed. Resetting to 0.")
            current_cam_idx = 0
            cap = cv2.VideoCapture(current_cam_idx)
            if not cap.isOpened(): break

        ret, frame = cap.read()
        if not ret: break
        h_frame, w_frame, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        
        eye_state = "OPEN"
        cnn_val = 1 # Debug raw value
        now = time.time()

        if current_mode != "WORD":
            show_word_commands = False
            editing_active = False
            selected_command_idx = None
        elif not show_word_commands:
            editing_active = False
            selected_command_idx = None

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark

            # Head-turn backspace gesture (runs even if eye crop fails)
            if HEAD_BACKSPACE_ENABLED:
                yaw = get_yaw_ratio(landmarks)
                # Choose direction. If mirrored, swap RIGHT/LEFT or negate yaw.
                if HEAD_BACKSPACE_DIRECTION.upper() == "RIGHT":
                    turn_val = yaw
                else:
                    turn_val = -yaw

                if turn_val > HEAD_BACKSPACE_YAW_THRESHOLD and now >= head_backspace_cooldown_until and head_backspace_armed:
                    if head_turn_start is None:
                        head_turn_start = now
                    elif (now - head_turn_start) >= HEAD_BACKSPACE_HOLD_SEC:
                        handle_backspace(now)
                        head_backspace_cooldown_until = now + HEAD_BACKSPACE_COOLDOWN_SEC
                        head_backspace_armed = False
                        head_turn_start = None
                else:
                    # Not currently above threshold; clear timer.
                    head_turn_start = None

                # Re-arm only once the head returns near center (hysteresis)
                if not head_backspace_armed and turn_val < HEAD_BACKSPACE_YAW_RESET:
                    head_backspace_armed = True

            crop = get_eye_crop(frame, landmarks, w_frame, h_frame)
            
            if crop.size != 0:
                # 1. Run CNN
                cnn_out = ort_session.run(None, {cnn_input_name: preprocess_cnn(crop)})
                logits = np.asarray(cnn_out[0])

                # Support both 2-class logits (legacy) and single-logit (current MobileNet export).
                flat = logits.reshape(-1)
                if flat.size >= 2:
                    # 2-class path: 0=Closed, 1=Open
                    cnn_val = int(np.argmax(flat[:2]))
                elif flat.size == 1:
                    # 1-logit path: sigmoid(logit) = P(open)
                    logit = float(flat[0])
                    open_prob = 1.0 / (1.0 + np.exp(-logit))
                    cnn_val = 1 if open_prob >= OPEN_PROB_THRESHOLD else 0
                else:
                    # Unexpected output shape; default to OPEN to avoid false blinks
                    cnn_val = 1
                
                if cnn_val == 0: 
                    eye_state = "CLOSED"
                    # Treat a detected closure as "activity" so we don't trigger CHAR_PAUSE while
                    # a blink/hold is still in progress.
                    last_open_time = now
                    potential_open_start = None
                    if not is_closed:
                        is_closed = True
                        closed_start_time = now
                else:
                    eye_state = "OPEN"
                    if is_closed:
                        if potential_open_start is None:
                            potential_open_start = now
                        
                        if (now - potential_open_start) > MIN_OPEN_STABILITY:
                            # Blink finished
                            duration = potential_open_start - closed_start_time
                            last_blink_duration = duration
                            is_closed = False
                            potential_open_start = None
                            if now >= input_block_until:
                                # Long-blink actions
                                if current_mode == "BUFFER" and duration >= BATCH_COMMIT_BLINK_SEC:
                                    # Finalize any in-progress Morse token
                                    if current_morse_token:
                                        buffered_tokens.append(current_morse_token)
                                        current_morse_token = ""

                                    decoded_text = decode_morse_tokens(buffered_tokens)
                                    buffered_tokens = []
                                    current_blink_sequence = []

                                    if decoded_text:
                                        transcript_words.append(decoded_text)
                                        decoded_history.append(decoded_text)
                                        speak_text(decoded_text)
                                        print(f"Buffered decode: {decoded_text}")
                                        last_token_time = now

                                        if log_events_f is not None:
                                            log_events_f.write(json.dumps({
                                                "ts": now,
                                                "type": "buffer_decode",
                                                "decoded": decoded_text,
                                                "mode": current_mode,
                                            }, ensure_ascii=False) + "\n")
                                            log_events_f.flush()
                                # Long-blink command gesture: speak full sentence (current transcript)
                                elif duration >= SENTENCE_SPEAK_BLINK_SEC and (now - last_sentence_speak_time) >= 0.5:
                                    transcript_text_now = " ".join(transcript_words + ([current_word] if current_word else []))
                                    speak_text(transcript_text_now)
                                    print(f"Speak sentence: {transcript_text_now}")
                                    last_sentence_speak_time = now

                                    if log_events_f is not None:
                                        log_events_f.write(json.dumps({
                                            "ts": now,
                                            "type": "gesture_speak_sentence",
                                            "blink_duration_sec": duration,
                                            "mode": current_mode,
                                            "transcript": transcript_text_now,
                                            "prompt": (eval_prompts[eval_prompt_idx] if EVAL_PROMPT_MODE and eval_prompts else None),
                                        }, ensure_ascii=False) + "\n")
                                        log_events_f.flush()

                                    # If running eval prompts: treat long blink as "submit attempt".
                                    if EVAL_PROMPT_MODE and eval_prompts:
                                        target = eval_prompts[eval_prompt_idx]
                                        pred = transcript_text_now
                                        attempt = {
                                            "prompt_index": eval_prompt_idx,
                                            "prompt": target,
                                            "predicted": pred,
                                            "cer": _cer(pred, target),
                                            "wer": _wer(pred, target),
                                            "submitted_at_ts": now,
                                        }
                                        if summary is not None:
                                            summary["eval"]["attempts"].append(attempt)

                                        # Advance to next prompt and clear transcript so attempts are separated.
                                        eval_prompt_idx = (eval_prompt_idx + 1) % len(eval_prompts)
                                        transcript_words = []
                                        current_word = ""
                                        last_token_time = now
                                elif duration > 0.05:
                                    current_blink_sequence.append(duration)
                                    print(f"Blink recorded: {duration:.2f}s")
                                    play_morse_beep(duration)
                                    speak_unit(duration)

                                    if current_mode == "BUFFER":
                                        # Build the Morse token without decoding yet
                                        if duration < DOT_DASH_THRESHOLD:
                                            current_morse_token += "."
                                        else:
                                            current_morse_token += "-"

                                    if log_events_f is not None:
                                        log_events_f.write(json.dumps({
                                            "ts": now,
                                            "type": "blink_recorded",
                                            "blink_duration_sec": duration,
                                        }, ensure_ascii=False) + "\n")
                                        log_events_f.flush()
                            else:
                                # Discard any blink that completes during the post-backspace lockout.
                                current_blink_sequence = []
                            last_open_time = now

        # 3. Prediction Timer Logic
        time_since_last_blink = now - last_open_time
        
        # Trigger Prediction (commit a token after a "letter pause")
        if now >= input_block_until and len(current_blink_sequence) > 0 and time_since_last_blink > CHAR_PAUSE_THRESHOLD:
            if current_mode == "BUFFER":
                if current_morse_token:
                    buffered_tokens.append(current_morse_token)
                    current_morse_token = ""
                    last_token_time = now
            else:
                predicted = predict_letter(current_blink_sequence, current_mode)

                if predicted and predicted != "?":
                    if current_mode in ("CHAR", "LETTERS"):
                        # Build words from letters; speak in LETTERS mode for audible confirmation
                        current_word += predicted
                        decoded_history.append(predicted)
                        print(f"Letter: {predicted}")
                        if current_mode == "LETTERS":
                            speak_text(predicted)
                    else:
                        # WORD mode: treat prediction as a full word token
                        transcript_words.append(predicted)
                        decoded_history.append(predicted)
                        speak_text(predicted)
                        print(f"Word: {predicted}")

                    if log_events_f is not None:
                        transcript_text_now = " ".join(transcript_words + ([current_word] if current_word else []))
                        log_events_f.write(json.dumps({
                            "ts": now,
                            "type": "token_committed",
                            "mode": current_mode,
                            "token": predicted,
                            "blink_sequence_sec": list(current_blink_sequence),
                            "transcript": transcript_text_now,
                            "prompt": (eval_prompts[eval_prompt_idx] if EVAL_PROMPT_MODE and eval_prompts else None),
                        }, ensure_ascii=False) + "\n")
                        log_events_f.flush()

                    last_token_time = now

            current_blink_sequence = []
            last_open_time = now  # reset pause timer after committing a token

        # Word boundary:
        # Only start counting a word-pause after we've been idle (no committed letters) for WORD_PAUSE_GRACE seconds.
        time_since_last_token = now - last_token_time
        if (
            now >= input_block_until
            and current_mode == "CHAR"
            and current_word
            and len(current_blink_sequence) == 0
            and time_since_last_token > (WORD_PAUSE_GRACE + WORD_PAUSE_THRESHOLD)
        ):
            transcript_words.append(current_word)
            speak_text(current_word)  # speak whole word
            print(f"Word committed: {current_word}")

            if log_events_f is not None:
                transcript_text_now = " ".join(transcript_words)
                log_events_f.write(json.dumps({
                    "ts": now,
                    "type": "word_boundary_commit",
                    "word": current_word,
                    "mode": current_mode,
                    "transcript": transcript_text_now,
                    "prompt": (eval_prompts[eval_prompt_idx] if EVAL_PROMPT_MODE and eval_prompts else None),
                }, ensure_ascii=False) + "\n")
                log_events_f.flush()

            current_word = ""
            last_token_time = now

        # Word boundary hint for BUFFER mode: insert a "/" separator after a long pause
        if (
            now >= input_block_until
            and current_mode == "BUFFER"
            and not current_morse_token
            and len(buffered_tokens) > 0
            and len(current_blink_sequence) == 0
            and time_since_last_token > (WORD_PAUSE_GRACE + WORD_PAUSE_THRESHOLD)
        ):
            buffered_tokens.append("/")
            last_token_time = now

        # Human-readable transcript
        transcript_text = " ".join(transcript_words + ([current_word] if current_word else []))

        # --- DRAWING THE NEW UI ---
        canvas = np.zeros((WINDOW_HEIGHT, WINDOW_WIDTH, 3), dtype=np.uint8)
        canvas[:] = COLORS["bg"]

        # 1. Header
        cv2.rectangle(canvas, (0, 0), (WINDOW_WIDTH, HEADER_HEIGHT), COLORS["panel_bg"], -1)
        cv2.rectangle(canvas, (0, 0), (WINDOW_WIDTH, HEADER_HEIGHT), COLORS["panel_border"], 1)
        
        cv2.putText(canvas, "BLINK MORSE DECODER", (30, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLORS["text_main"], 2, cv2.LINE_AA)
        cv2.putText(canvas, "Thesis by Group 2 BSCS 4A SY 2025-2026", (30, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLORS["text_muted"], 1, cv2.LINE_AA)
        
        if EVAL_PROMPT_MODE and eval_prompts:
            prompt_text = f"PROMPT: {eval_prompts[eval_prompt_idx]}"
            cv2.putText(canvas, prompt_text, (350, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS["accent_warning"], 1, cv2.LINE_AA)

        # Buttons
        draw_button(canvas, BTN_THEME_RECT, f"THEME: {current_theme_name.upper()}", active=False)
        draw_button(canvas, BTN_AUDIO_RECT, f"AUDIO: {current_audio_mode}", active=(current_audio_mode=="TTS"))
        
        # Mode Button Color Logic
        mode_active = True
        mode_col = None
        if current_mode == "WORD": mode_col = COLORS["accent_primary"]
        elif current_mode == "CHAR": mode_col = COLORS["accent_warning"]
        elif current_mode == "BUFFER": mode_col = COLORS["accent_success"]
        draw_button(canvas, BTN_MODE_RECT, current_mode, active=mode_active, color=mode_col)
        
        cmd_active = (current_mode == "WORD" and show_word_commands)
        draw_button(canvas, BTN_CMD_RECT, "LIBRARY", active=cmd_active)

        # 2. Video Panel
        vx, vy, vw, vh = VIDEO_PANEL
        draw_panel(canvas, VIDEO_PANEL, "Live Feed")
        
        # Fit Video - STRETCH TO FILL
        display_w = vw - 4
        display_h = vh - 38 # Account for header (35) and borders
        
        res_frame = cv2.resize(frame, (display_w, display_h))
        
        # Overlay eye state on video pixels
        if eye_state == "CLOSED":
            cv2.rectangle(res_frame, (0,0), (display_w, display_h), (0,0,255), 3) 
        
        # Position just below header
        dy = vy + 36
        dx = vx + 2
        canvas[dy:dy+display_h, dx:dx+display_w] = res_frame

        # HUD Overlay on top of video area
        state_col = COLORS["accent_success"] if eye_state == "OPEN" else COLORS["accent_danger"]
        
        # Draw background for HUD pill
        cv2.rectangle(canvas, (vx+15, vy+50), (vx+130, vy+80), COLORS["overlay_bg"], -1)
        cv2.putText(canvas, f"STATE: {eye_state}", (vx+20, vy+72), cv2.FONT_HERSHEY_SIMPLEX, 0.5, state_col, 1, cv2.LINE_AA)

        # Blink Meter (Bottom of video panel)
        meter_w = 300
        meter_h = 8
        mx = vx + (vw - meter_w) // 2
        my = vy + vh - 25
        
        # Determine meter value
        meter_val = 0.0
        if is_closed:
             meter_val = min((now - closed_start_time)/1.0, 1.0)
        else:
             # Decay effect for visualization or just show 0
             meter_val = 0.0
        
        draw_progress_bar(canvas, (mx, my, meter_w, meter_h), meter_val)
        dur_txt = f"{(now - closed_start_time):.2f}s" if is_closed else "0.00s"
        cv2.putText(canvas, dur_txt, (mx + meter_w//2 - 20, my - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLORS["text_main"], 1, cv2.LINE_AA)

        # 3. Translation Panel
        tx, ty, tw, th = TRANS_PANEL
        draw_panel(canvas, TRANS_PANEL, "Input Buffer", badge="Active")
        
        # Sequence String
        seq_str = ""
        if current_mode == "BUFFER":
            parts = buffered_tokens.copy()
            if current_morse_token: parts.append(current_morse_token)
            seq_str = " ".join(parts)
        else:
            for d in current_blink_sequence:
                seq_str += " _ " if d >= DOT_DASH_THRESHOLD else " . "
    
        cv2.putText(canvas, seq_str, (tx+20, ty+64), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS["text_muted"], 1, cv2.LINE_AA)
    
        # Current Word/Text (wrapped to fit panel)
        full_line = " ".join(transcript_words[-6:] + ([current_word] if current_word else []))
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.1
        thickness = 2
        max_text_width = tw - 40
        wrapped_lines = wrap_text_for_width(full_line or "_", max_text_width, font, font_scale, thickness)
        line_y = ty + 98
        line_spacing = 28
        for line in wrapped_lines[:4]:  # cap lines to avoid overflow
            cv2.putText(canvas, line, (tx+20, line_y), font, font_scale, COLORS["text_main"], thickness, cv2.LINE_AA)
            line_y += line_spacing

        # LSTM usage breadcrumb so we can visually verify the model is being invoked
        if last_lstm_debug["ts"] > 0:
            dbg = last_lstm_debug
            dbg_time = datetime.fromtimestamp(dbg["ts"]).strftime("%H:%M:%S")
            dbg_line1 = f"LSTM {dbg['mode']} #{lstm_call_count}: {dbg['pred_label']} (idx {dbg['pred_idx']}, len {dbg['seq_len']})"
            seq_preview = " ".join([f"{d:.2f}" for d in dbg["blink_seq"][-6:]])
            dbg_line2 = f"{dbg_time} seq: {seq_preview}" if seq_preview else f"{dbg_time} seq: -"
            cv2.putText(canvas, dbg_line1, (tx+20, ty+128), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLORS["text_muted"], 1, cv2.LINE_AA)
            cv2.putText(canvas, dbg_line2, (tx+20, ty+142), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLORS["text_muted"], 1, cv2.LINE_AA)

        # 4. History Panel
        hx, hy, hw, hh = HISTORY_PANEL
        draw_panel(canvas, HISTORY_PANEL, "Session Log")
        
        # Draw list
        visible_hist = (transcript_words + ([current_word] if current_word else []))[-18:]
        start_y = hy + 60
        for i, w in enumerate(visible_hist):
            col = COLORS["text_muted"]
            if i == len(visible_hist)-1 and current_word: 
                col = COLORS["accent_primary"]
                # Active highlight
                cv2.rectangle(canvas, (hx+2, start_y-20), (hx+hw-2, start_y+10), COLORS["overlay_bg"], -1)
                cv2.rectangle(canvas, (hx, start_y-20), (hx+3, start_y+10), COLORS["accent_primary"], -1)
            
            ts = datetime.fromtimestamp(now).strftime("%H:%M:%S") # Mock time for items just for visuals, or track real time
            cv2.putText(canvas, f"{w}", (hx+30, start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 1, cv2.LINE_AA)
            start_y += 30

        # 5. Command Overlay
        if show_word_commands and current_mode == "WORD":
            # Darken bg
            overlay = canvas.copy()
            cv2.rectangle(overlay, (0,0), (WINDOW_WIDTH, WINDOW_HEIGHT), COLORS["overlay_bg"], -1)
            cv2.addWeighted(overlay, 0.7, canvas, 0.3, 0, canvas)
            
            # Draw Centered Modal
            mw, mh = 800, 600
            mx = (WINDOW_WIDTH - mw) // 2
            my = (WINDOW_HEIGHT - mh) // 2
            
            cv2.rectangle(canvas, (mx, my), (mx+mw, my+mh), COLORS["panel_bg"], -1)
            cv2.rectangle(canvas, (mx, my), (mx+mw, my+mh), COLORS["panel_border"], 1)
            
            # Header
            cv2.rectangle(canvas, (mx, my), (mx+mw, my+50), COLORS["panel_header"], -1)
            cv2.putText(canvas, "COMMAND LIBRARY", (mx+20, my+35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS["text_main"], 2)
            cv2.putText(canvas, "Click to edit. ESC to close.", (mx+mw-250, my+35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["text_muted"], 1)
            
            # Grid
            row_hitboxes = []
            cols = 3
            rows = 17
            cw = (mw - 40) // cols
            rh = 30
            
            cmds = sorted(WORD_COMMAND_CODES.items())
            for i, (idx, code) in enumerate(cmds):
                c = i // rows
                r = i % rows
                
                bx = mx + 20 + c * cw
                by = my + 70 + r * rh
                
                w_txt = idx_to_word.get(idx, "[Empty]")
                
                # Highlight
                is_sel = (selected_command_idx == idx)
                txt_col = COLORS["accent_primary"] if is_sel else COLORS["text_muted"]
                if is_sel:
                    # Highlight box
                    cv2.rectangle(canvas, (bx, by-20), (bx+cw-10, by+5), COLORS["overlay_bg"], -1)
                
                disp = f"{idx:02d} {code:6s} {w_txt}"
                cv2.putText(canvas, disp, (bx+5, by), cv2.FONT_HERSHEY_SIMPLEX, 0.45, txt_col, 1, cv2.LINE_AA)
                
                row_hitboxes.append((idx, (bx, by-20, cw-10, 25)))

            if editing_active:
                cv2.rectangle(canvas, (mx, my+mh-50), (mx+mw, my+mh), COLORS["overlay_bg"], -1)
                cv2.putText(canvas, f"EDITING ID {selected_command_idx}: {edit_buffer}_", (mx+20, my+mh-15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS["accent_warning"], 2)

        # Handle Overlay Clicks
        if pending_click:
            px, py = pending_click
            pending_click = None
            if show_word_commands and current_mode == "WORD":
                for idx, r in row_hitboxes:
                    rx, ry, rw, rh = r
                    if rx <= px <= rx+rw and ry <= py <= ry+rh:
                        selected_command_idx = idx
                        edit_buffer = idx_to_word.get(idx, "")
                        editing_active = True
        
        cv2.imshow("Blink Morse Decoder", canvas)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break

        # Allow ESC to close the Library overlay even when not editing
        if key == 27 and show_word_commands:
            show_word_commands = False
            editing_active = False
            selected_command_idx = None
            continue
        
        # Typing logic
        if editing_active and show_word_commands:
            if key == 27: editing_active = False
            elif key == 13: # Enter
                commit_word_edit()
            elif key in (8, 127): edit_buffer = edit_buffer[:-1]
            elif 32 <= key <= 126: edit_buffer += chr(key)
            continue
            
        # Backspace manual
        if key in (8, 127):
            handle_backspace(now)
            continue
        if key == ord('c'):
            cap.release()
            current_cam_idx += 1
            if current_cam_idx > 3: current_cam_idx = 0 # Cycle 0-3
            cap = cv2.VideoCapture(current_cam_idx)
            print(f"Switching to camera {current_cam_idx}...")

        # Optional: skip to next prompt (eval mode)
        if key == ord('n') and EVAL_PROMPT_MODE and eval_prompts:
            eval_prompt_idx = (eval_prompt_idx + 1) % len(eval_prompts)
            transcript_words = []
            current_word = ""
            last_token_time = now

    # Exit
    beep_queue.put(None)
    speech_queue.put(None)
    cap.release()
    cv2.destroyAllWindows()

    if summary is not None and session_summary_path is not None:
        summary["ended_at"] = datetime.now().isoformat(timespec="seconds")
        try:
            with open(session_summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Failed to write session summary: {e}")

    if log_events_f is not None:
        try:
            log_events_f.close()
        except Exception:
            pass

if __name__ == "__main__":
    main()