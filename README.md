# CNN‑LSTM Blink Morse Decoder

A Windows-focused, real-time Morse-code input system that uses a CNN to detect eye openness and an LSTM to translate blink durations into characters/words. The app renders a dashboard UI (OpenCV), supports speech/beep feedback, command editing, and head-turn backspace gestures.

## Upstream
Forked from **MizBitz/CNN-LSTM-Blink-Morse-Decoder**: https://github.com/MizBitz/CNN-LSTM-Blink-Morse-Decoder

## Features
- **Live blink → Morse → text** decoding with CNN (eye state) + LSTM (sequence).
- **Multiple input modes:** WORD, CHAR, BUFFER (raw Morse), LETTERS (audible per-letter).
- **Audio feedback:** Windows SAPI TTS, mute, or Morse beeps.
- **Gesture controls:** long-blink actions, head-turn backspace, library editing for word commands.
- **UI theming:** dark/light toggle, live feed, input buffer, and session log panels.
- **Session logging:** JSONL event log plus optional evaluation summaries.

## Requirements
- **OS:** Windows (uses `win32com.client`, `pythoncom`, `winsound`, SAPI voice).
- **Python:** 3.9+ (tested versions may vary).
- **Dependencies:** `opencv-python`, `numpy`, `onnxruntime` (CPU/CUDA), `mediapipe`, `pywin32`, `textwrap`, `json`, `threading`, `queue`.
- **Hardware:** Webcam.

## Model & data files (place alongside `mc_inference.py`)
- `eye_state_mobilenet.onnx` — CNN eye-state model.
- `blink_lstm.onnx` — LSTM blink-sequence model.
- `lstm_word_map.json` — word label map (word → index).
- `lstm_label_map.json` — char label map (char → index) for non-WORD modes.
- Optional: `eval_prompts.txt` — one prompt per line for evaluation mode.

## Quick start
1. Install deps (example):
   ```bash
   pip install opencv-python numpy onnxruntime mediapipe pywin32
   ```
   - For GPU, install `onnxruntime-gpu` and ensure CUDA is available.
2. Place the ONNX and JSON map files in the project root (see above).
3. Run:
   ```bash
   python mc_inference.py
   ```
4. Ensure your webcam is available; the window opens with live decoding.

## Controls & gestures
- **Mouse buttons (header):**
  - THEME: toggle dark/light.
  - AUDIO: cycle TTS → MUTE → BEEP.
  - MODE: cycle WORD → CHAR → BUFFER → LETTERS.
  - LIBRARY (WORD mode): open command library overlay (edit word labels).
- **Keyboard:**
  - `q` — quit.
  - `c` — cycle camera index (0–3).
  - `Backspace` — delete (context-aware: buffer or transcript).
  - In library edit overlay: `Enter` to save, `Esc` to cancel, regular typing to edit, `Backspace` to delete.
- **Head-turn backspace:** turn head past yaw threshold (`HEAD_BACKSPACE_YAW_THRESHOLD`, default 0.25) toward `HEAD_BACKSPACE_DIRECTION` (default RIGHT), hold for `HEAD_BACKSPACE_HOLD_SEC` (0.15s); cooldown applies.
- **Blink gestures:**
  - **Dot/Dash:** short/long blink split at `DOT_DASH_THRESHOLD` (0.52s).
  - **Commit token (auto):** after `CHAR_PAUSE_THRESHOLD` of openness (1.0s).
  - **Word boundary (CHAR mode):** idle longer than `WORD_PAUSE_GRACE + WORD_PAUSE_THRESHOLD` (2.0s + 2.0s).
  - **Long blink (BUFFER mode):** if ≥ `BATCH_COMMIT_BLINK_SEC` (2.0s), finalize buffered Morse and decode.
  - **Long blink (any mode):** if ≥ `SENTENCE_SPEAK_BLINK_SEC` (2.0s), speak full transcript.
  - **Buffer word separator:** in BUFFER mode, long idle inserts `/`.

## Modes (what commits)
- **WORD:** Each blink sequence → word via LSTM word map.
- **CHAR:** Sequences → letters; words auto-commit after pause.
- **BUFFER:** Raw Morse assembly; long blink decodes buffered tokens.
- **LETTERS:** Like CHAR but speaks each letter as committed.

## Audio modes
- **TTS:** Speaks committed tokens (words or letters) and long-blink sentence.
- **BEEP:** Plays dot/dash beeps (`winsound.Beep`); respects `DOT_BEEP_MS`, `DASH_BEEP_MS`, `BEEP_GAP_MS`.
- **MUTE:** No audio.

## UI layout
- **Header:** App title, thesis credit, buttons (theme/audio/mode/library).
- **Live Feed panel:** camera feed with blink meter and eye-state overlay.
- **Input Buffer panel:** shows current Morse/dot-dash sequence and active text line.
- **Session Log panel:** recent transcript items (highlighted active word).
- **Library overlay (WORD mode):** grid of command indices/code/text; click to edit and save to `lstm_word_map.json`.

## Configuration highlights (edit in `mc_inference.py`)
- Model paths: `CNN_MODEL_PATH`, `LSTM_MODEL_PATH`, `LABEL_MAP_PATH`.
- Timings: `DOT_DASH_THRESHOLD`, `CHAR_PAUSE_THRESHOLD`, `WORD_PAUSE_THRESHOLD`, `WORD_PAUSE_GRACE`, `MIN_OPEN_STABILITY`.
- Long-blink actions: `SENTENCE_SPEAK_BLINK_SEC`, `BATCH_COMMIT_BLINK_SEC`.
- Audio: `current_audio_mode`, `BEEP_FREQ_HZ`, `DOT_BEEP_MS`, `DASH_BEEP_MS`.
- Head backspace: `HEAD_BACKSPACE_*` constants, `POST_BACKSPACE_INPUT_BLOCK_SEC`.
- Themes: `THEMES` dict and `current_theme_name`.
- Logging: `LOGGING_ENABLED`, `LOG_DIR`; evaluation: `EVAL_PROMPT_MODE`, `EVAL_FORCE_CHAR_MODE`, `EVAL_PROMPTS_PATH`.

## Logging & evaluation
- When logging/eval enabled, writes JSONL event logs to `session_logs/mc_session_<timestamp>.jsonl` and optional summary JSON.
- Eval mode cycles prompts from `eval_prompts.txt`, locks CHAR mode if `EVAL_FORCE_CHAR_MODE=True`, and records CER/WER per submission (long blink).

## Troubleshooting
- **No TTS / crashes:** Ensure Windows, `pywin32` installed, SAPI available. TTS only works in TTS mode.
- **No camera:** Check webcam permissions; use `c` to cycle devices.
- **Model load errors:** Verify ONNX and JSON map files exist at configured paths.
- **GPU issues:** If CUDA provider fails, code falls back to CPU; ensure correct onnxruntime package.

## Run reminder
```bash
python mc_inference.py
```
