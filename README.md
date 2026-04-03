# Eternix - Real-Time Emotion Detection and AI Companion

Eternix is a Flask + Socket.IO web app that reads webcam frames in real time, detects emotion from face cues, and uses that context to guide AI chat responses.

This README is aligned to the current codebase (not legacy docs).

## What It Does

- Streams webcam frames from browser to backend over Socket.IO.
- Runs image preprocessing before inference:
  - Adaptive gamma correction (optional, config driven)
  - CLAHE on LAB L-channel
- Runs emotion inference with two paths:
  - GPU path (DeepFace `extract_faces` + HSEmotion classifier)
  - CPU fallback path (DeepFace `analyze` with backend fallback logic)
- Stabilizes output with:
  - EMA smoothing
  - Confidence gating
  - Hysteresis (sticky emotion switching)
  - Optional frame skipping (`analyze_every_n_frames`)
- Tracks face boxes across short misses to reduce UI flicker.
- Exposes health/readiness/metrics APIs.
- Supports AI chat via DeepSeek API.

## Technical Decisions

### Why EMA smoothing instead of raw model output?

Raw per-frame emotion predictions are noisy — a model can output "happy" one frame and "neutral" the next even when the user's expression hasn't changed. Instead of displaying this jitter, Eternix maintains an Exponential Moving Average (EMA) of emotion scores across frames. EMA was chosen over alternatives like Kalman filtering because:

- Emotion scores aren't a single continuous variable — they're a probability distribution across 7 classes. Kalman filters assume Gaussian state transitions, which doesn't map cleanly to categorical distributions.
- EMA is computationally trivial (important at 5-8 FPS real-time processing).
- The decay factor (`alpha: 0.4`) is intuitive to tune: lower = smoother but slower to react.

### Why hysteresis (asymmetric switch-in/switch-out)?

Even with EMA, the dominant emotion can oscillate between two close-scoring emotions (e.g., "sad" and "neutral"). Hysteresis solves this by making it **harder to leave** the current emotion than to **enter** it:

- `switch_in_frames: 3` — a new emotion must dominate for 3 consecutive frames before the label changes (~0.6s at 5 FPS).
- `switch_out_frames: 4` — the current emotion gets an extra frame of "benefit of the doubt."
- `min_dominance_margin: 2.0` — the new emotion's EMA score must lead the runner-up by at least 2 points.

This mirrors how human perception works — we don't perceive someone as "switching" emotions on every micro-expression.

### Why a dual inference pipeline (GPU + CPU fallback)?

The GPU path (YOLOv8 face detection → HSEmotion classifier) is faster and more accurate, but requires CUDA + PyTorch. The CPU fallback (DeepFace with TensorFlow) ensures the app works on any machine. The fallback is automatic — if the GPU path fails for any reason (missing CUDA, model load error, inference exception), the system silently falls back to CPU within the same frame processing call.

### Why disgust guardrails?

Facial emotion models trained on FER/AffectNet datasets have a well-documented bias: they over-predict "disgust" on neutral/resting faces, especially for certain demographic groups. Rather than retraining the model, Eternix applies post-inference guardrails:

- Reject "disgust" predictions below a minimum raw score (43/100 — found empirically via the calibration script).
- If "happy" scores ≥82% of the "disgust" score, override to "happy" (catches smirks misclassified as disgust).
- If any other emotion scores ≥90% of disgust, prefer that emotion instead.

These thresholds were calibrated using `scripts/evaluate_emotion_dataset.py` against labeled samples to find the point where false positives drop sharply while true positives are retained.

### Why DeepSeek over OpenAI/Claude for the therapist?

Cost. DeepSeek Chat is ~100x cheaper than GPT-4 and ~20x cheaper than GPT-3.5-turbo for equivalent output quality in empathetic conversation. For a real-time system where every emotion update could trigger an API call, cost per token matters significantly. The OpenAI-compatible SDK also means switching providers later requires only changing the base URL and API key.

## Current Runtime Behavior

- App always runs on port `5000` (hardcoded for Socket.IO + CORS consistency).
- Optional auto-kill behavior can terminate existing listeners on `5000` before startup (`app.auto_kill_existing_port_process`).

## High-Level Architecture

1. Browser captures video (`640x480` by default) and sends JPEG frames over Socket.IO.
2. Backend decodes frame and queues processing in a thread pool (`emotion.worker_count`, clamped `1..4`).
3. Preprocess frame (`services/emotion_pipeline.py`):
   - BGR -> LAB
   - Adaptive gamma correction using target brightness
   - CLAHE
   - Optional denoise/sharpen
4. Emotion inference:
   - If `gpu_emotion.enabled: true`: detect face + classify with HSEmotion.
   - On GPU path failure: fallback to DeepFace CPU path.
   - CPU path uses detector strategy (`opencv`/`ssd` in balanced mode by default).
5. Runtime calibration and guardrails:
   - Optional class-bias calibration artifact
   - Confidence calibration
   - Disgust guardrails
6. Smoother (`emotion_smoother.py`) updates EMA and stable emotion state.
7. Server emits `emotion_update` with raw and smoothed payloads.
8. Frontend renders box, chart, confidence, and emotion-aware chat context.

## Project Layout

```text
app.py
config.yaml
emotion_smoother.py
services/
  chat_service.py
  emotion_pipeline.py
  evaluation.py
  face_tracker.py
  runtime_calibration.py
scripts/
  benchmark_streaming.py
  evaluate_emotion_dataset.py
templates/
  index.html
static/
  css/style.css
  js/main.js
tests/
  test_*.py
restart_5000.ps1
requirements.txt
requirements-cpu.txt
requirements-gpu.txt
```

## Requirements

Recommended:

- Python `3.10.x` (TensorFlow 2.10 compatibility)
- Windows/Linux/macOS
- Webcam and browser camera permission

Optional for GPU path:

- NVIDIA GPU + CUDA-compatible PyTorch install
- `ultralytics`, `hsemotion`, `timm`, `torch`

## Setup

### 1) Create and activate virtual environment

Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python -m venv venv        
source venv/bin/activate
```

### 2) Install dependencies

```bash
# Full install (CPU + GPU):
pip install -r requirements.txt

# Or choose one:
pip install -r requirements-cpu.txt   # DeepFace + TensorFlow only (no GPU needed)
pip install -r requirements-gpu.txt   # Adds HSEmotion + PyTorch (requires CUDA)
```

### 3) Configure environment variables

Create or update `.env`:

```env
FLASK_SECRET_KEY=replace_with_strong_secret
DEEPSEEK_API_KEY=your_deepseek_api_key_here
```

The only supported AI chat provider is **DeepSeek**. The `DEEPSEEK_API_KEY` environment variable is required for the AI therapist chat feature to work.

### 4) Review `config.yaml`

Important keys:

- `gpu_emotion.enabled`
- `emotion.analyze_every_n_frames`
- `emotion.smoothing.*`
- `image_preprocessing.gamma_correction.*`
- `privacy.require_camera_consent`
- `security.cors_origins`

## Run

### Standard

```bash
python app.py
```

Open:

- `http://127.0.0.1:5000`

### Windows helper (kill old listener and restart)

```powershell
powershell -ExecutionPolicy Bypass -File .\restart_5000.ps1
```

### CLI options

```bash
python app.py --optimize-speed
python app.py --model deepseek-chat
python app.py --no-server-camera
```

Notes:

- `--model` overrides configured chat model.
- `--optimize-speed` selects speed-oriented detection behavior.
- `--no-server-camera` is accepted for compatibility.

## API Endpoints

HTTP:

- `GET /` -> UI
- `GET /api/emotion` -> latest emotion + confidence
- `GET /api/emotion/history` -> recent history + summary
- `GET /api/metrics` -> runtime counters/timings
- `GET /api/health` -> health snapshot
- `GET /api/readiness` -> readiness (`200` ready, `503` not ready)
- `GET /api/calibration/status` -> artifact/calibration status
- `POST /api/calibration/reload` -> reload calibration artifact
- `POST /api/chat` -> synchronous chat endpoint

Socket.IO events:

- Client -> Server: `frame`, `user_message`, `camera_consent`, `clear_emotion_state`, `toggle_lock`
- Server -> Client: `emotion_update`, `ai_typing`, `ai_response`, `camera_consent_required`, `camera_consent_ack`, `emotion_memory_cleared`

## Emotion Payload Shape

`emotion_update` includes:

- `emotion`
- `confidence` (display calibrated)
- `confidence_raw`
- `raw_emotions` (per-class scores)
- `face_detected`
- `face_tracked`
- `face_location`
- `smoothed` object with stable emotion/confidence/scores
- `quality.low_confidence` flags

## Calibration and Dataset Evaluation

Build calibration artifact from labeled predictions:

```bash
python scripts/evaluate_emotion_dataset.py --input path/to/dataset.jsonl --output artifacts/emotion_calibration.json
```

Enable in `config.yaml`:

```yaml
emotion:
  data_calibration:
    enabled: true
    artifact_path: "artifacts/emotion_calibration.json"
```

Accepted dataset fields (JSON/JSONL parser supports aliases):

- True label keys: `label`, `true_emotion`, `ground_truth`, `target`
- Scores keys: `scores`, `raw_emotions`, `emotion_scores`, `predicted_scores`
- Predicted label keys: `predicted_emotion`, `dominant_emotion`, `prediction`, `pred`
- Confidence keys: `confidence`, `predicted_confidence`, `probability`

Minimal JSONL example:

```json
{"ground_truth":"happy","predicted_emotion":"disgust","confidence":0.62,"raw_emotions":{"happy":38.0,"disgust":42.0,"neutral":20.0}}
{"ground_truth":"happy","predicted_emotion":"happy","confidence":0.81,"raw_emotions":{"happy":81.0,"neutral":19.0}}
```

## Synthetic Performance Benchmark

```bash
python scripts/benchmark_streaming.py --frames 300 --warmup 30
```

This runs synthetic `process_frame_async` benchmarking with mocked inference.

## Testing

Run tests:

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

Optional syntax checks:

```bash
python -m py_compile app.py services/emotion_pipeline.py emotion_smoother.py services/runtime_calibration.py
node --check static/js/main.js
```

## Troubleshooting

### `ERR_CONNECTION_REFUSED` on `:5000`

Backend is not running or crashed. Start it:

```bash
python app.py
```

Verify listener on Windows:

```powershell
Get-NetTCPConnection -LocalPort 5000 -State Listen
```

### `/api/readiness` returns `503`

Server is up but detector is not ready in the active environment.

Check `/api/health` and inspect:

- `checks.detector_ready`
- `runtime.deepface_available`
- `runtime.gpu_available`

Common causes:

- Running with wrong venv
- Missing DeepFace or GPU stack in current interpreter

### Missing optional Flask extensions

If `flask_caching` or `flask_limiter` is missing, the app now falls back to no-op implementations and logs warnings.
Install the packages from `requirements.txt` if you want full caching/rate-limit behavior.

### WebSocket failures

If you see Socket.IO websocket errors with repeated reconnects:

- Confirm backend is listening on `127.0.0.1:5000`
- Confirm browser is loading frontend from allowed `security.cors_origins`

### Camera consent errors

If UI says consent is required:

- Enable `Consent to Camera Analysis` toggle in the UI
- Retry camera start

### Performance tuning

- Increase `emotion.analyze_every_n_frames` to `2` for lower CPU usage.
- Keep `emotion.worker_count` small (`1..4`) based on hardware.
- Use GPU path when available (`gpu_emotion.enabled: true`).

## Deployment Notes

This app relies on long-lived Socket.IO connections and real-time inference.

- Not suitable for Vercel Functions as a WebSocket backend.
- Use a persistent host (VM/container/bare metal) for backend runtime.
- If exposed publicly, harden `FLASK_SECRET_KEY`, CORS, and rate-limit backend storage.

## Security and Privacy Notes

- Camera frames are streamed to backend for inference.
- Treat output as assistive, not medical diagnosis.
- Configure secure cookies and production CORS before internet exposure.

## Current Branding

- App name: `Eternix`
- Footer: `(c) 2026 Eternix - Developed by Mohit Pandya`
