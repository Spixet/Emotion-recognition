import argparse
import atexit
import base64
import concurrent.futures
import logging
from logging.handlers import RotatingFileHandler
import os
import socket
import subprocess
import sys
import threading
import time


import cv2
import numpy as np
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from flask_socketio import SocketIO, emit
import yaml

from emotion_smoother import EmotionSmoother
from services.chat_service import ChatClientManager
from services.face_tracker import SessionFaceTracker
from services.emotion_pipeline import EmotionPipeline
from services.runtime_calibration import RuntimeCalibration


_HAS_FLASK_CACHING = True
try:
    from flask_caching import Cache
except ImportError:
    _HAS_FLASK_CACHING = False

    class Cache:  # type: ignore[override]
        """No-op fallback when flask_caching is unavailable."""

        def __init__(self, app=None, config=None):
            self.app = None
            self.config = {}
            if app is not None:
                self.init_app(app, config=config)

        def init_app(self, app, config=None):
            self.app = app
            self.config = dict(config or {})

        def get(self, _key):
            return None

        def set(self, _key, _value, timeout=None):
            _ = timeout
            return False

        def delete(self, _key):
            return False


_HAS_FLASK_LIMITER = True
try:
    from flask_limiter import Limiter
    from flask_limiter.util import get_remote_address
except ImportError:
    _HAS_FLASK_LIMITER = False

    def get_remote_address():  # type: ignore[override]
        return request.remote_addr if request else "local"

    class Limiter:  # type: ignore[override]
        """No-op fallback when flask_limiter is unavailable."""

        def __init__(self, *_args, **_kwargs):
            pass

        def limit(self, *_args, **_kwargs):
            def decorator(func):
                return func

            return decorator


# Ensure Unicode logging does not crash on Windows cp1252 consoles.
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def configure_logging(config):
    log_cfg = config.get("logging", {})
    level_name = str(log_cfg.get("level", "INFO")).upper()
    level = getattr(logging, level_name, logging.INFO)

    logger = logging.getLogger("eternix")
    logger.setLevel(level)
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    # Clear handlers to avoid duplication during tests/imports.
    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    log_file = log_cfg.get("file")
    if log_file:
        max_size = int(log_cfg.get("max_size", 10 * 1024 * 1024))
        backup_count = int(log_cfg.get("backup_count", 5))
        file_handler = RotatingFileHandler(
            log_file, maxBytes=max_size, backupCount=backup_count, encoding="utf-8"
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# Load environment variables
load_dotenv()

# Load configuration
CONFIG = {}
try:
    with open("config.yaml", "r", encoding="utf-8") as config_file:
        CONFIG = yaml.safe_load(config_file) or {}
except FileNotFoundError:
    print("[ERROR] config.yaml not found. Please ensure it exists in the root directory.")
    sys.exit(1)
except yaml.YAMLError as exc:
    print(f"[ERROR] Failed to parse config.yaml: {exc}")
    sys.exit(1)

logger = configure_logging(CONFIG)
logger.info("Configuration loaded successfully from config.yaml")
if not _HAS_FLASK_CACHING:
    logger.warning("flask_caching is not installed; using no-op cache backend.")
if not _HAS_FLASK_LIMITER:
    logger.warning("flask_limiter is not installed; rate limiting is disabled.")

# Parse command line arguments
parser = argparse.ArgumentParser(description="Eternix - Your AI Companion")
parser.add_argument("--no-server-camera", action="store_true", help="Disable server-side camera")
parser.add_argument(
    "--optimize-speed", action="store_true", help="Use optimized settings for faster performance"
)
parser.add_argument(
    "--model", type=str, default=None, help="LLM model to use (overrides config.yaml if set)"
)
args, _unknown_cli_args = parser.parse_known_args()

# Initialize Flask app
app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("FLASK_SECRET_KEY", "default_insecure_secret_change_me")
if app.config["SECRET_KEY"] == "default_insecure_secret_change_me":
    logger.warning("FLASK_SECRET_KEY is using the insecure fallback value.")
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_SECURE"] = bool(CONFIG.get("security", {}).get("cookie_secure", False))

# Restrict CORS to configured origins only.
cors_origins = CONFIG.get("security", {}).get("cors_origins", [])
if not cors_origins:
    cors_origins = ["http://localhost:5000", "http://127.0.0.1:5000"]

socketio = SocketIO(
    app,
    cors_allowed_origins=cors_origins,
    max_http_buffer_size=10 * 1024 * 1024,
    async_mode="threading",
)

# Thread Pool for heavy tasks
worker_count = int(CONFIG.get("emotion", {}).get("worker_count", 1) or 1)
worker_count = max(1, min(worker_count, 4))
executor = concurrent.futures.ThreadPoolExecutor(max_workers=worker_count)
atexit.register(executor.shutdown, wait=False)

processing_lock = threading.Lock()
session_processing_flags = {}

frame_skip_lock = threading.Lock()
session_frame_counters = {}
session_last_results = {}
session_last_smoothed = {}

# Initialize Cache (kept for non-realtime endpoints/extensions)
cache_config = {"CACHE_TYPE": "SimpleCache", "CACHE_DEFAULT_TIMEOUT": 300}
cache = Cache(app, config=cache_config)

# Initialize Limiter
rate_limit_storage_uri = CONFIG.get("security", {}).get("rate_limit_storage_uri", "memory://")
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["200 per day", "50 per hour"],
    storage_uri=rate_limit_storage_uri,
)
if str(rate_limit_storage_uri).startswith("memory://"):
    logger.warning(
        "Rate limiter storage is memory://; use Redis-backed storage in production for multi-process safety."
    )

# Runtime state
last_emotion = None
emotion_confidence = 0.0
emotion_confidence_raw = 0.0
last_emotion_time = 0.0
emotion_history = []
emotion_history_lock = threading.Lock()
session_emotion_state = {}
consent_lock = threading.Lock()
session_camera_consent = {}

session_smoothers = {}
smoother_lock = threading.Lock()

model_init_complete = False
deepseek_client = None

emotion_pipeline = EmotionPipeline(CONFIG, optimize_speed=args.optimize_speed, logger=logger)
chat_client_manager = ChatClientManager(CONFIG, cli_model_override=args.model, logger=logger)
runtime_calibration = RuntimeCalibration(CONFIG, logger=logger)

face_tracking_cfg = CONFIG.get("camera", {}).get("face_tracking", {})
face_tracker = SessionFaceTracker(
    enabled=face_tracking_cfg.get("enabled", True),
    smoothing_alpha=face_tracking_cfg.get("smoothing_alpha", 0.65),
    max_missing_frames=face_tracking_cfg.get("max_missing_frames", 3),
    reacquire_iou_threshold=face_tracking_cfg.get("reacquire_iou_threshold", 0.05),
    max_center_jump_ratio=face_tracking_cfg.get("max_center_jump_ratio", 0.45),
)

runtime_started_at = time.time()
runtime_metrics_lock = threading.Lock()
runtime_metrics = {
    "frames_received": 0,
    "frames_dropped_busy": 0,
    "frames_dropped_no_consent": 0,
    "frames_decode_error": 0,
    "frames_processed": 0,
    "frames_inferred": 0,
    "frames_skipped": 0,
    "frames_failed": 0,
    "frames_face_tracked": 0,
    "frames_calibrated": 0,
    "avg_processing_ms": 0.0,
}


def _increment_metric(metric_name, amount=1):
    with runtime_metrics_lock:
        runtime_metrics[metric_name] = int(runtime_metrics.get(metric_name, 0)) + int(amount)


def _record_processing_time(processing_ms):
    with runtime_metrics_lock:
        previous_avg = float(runtime_metrics.get("avg_processing_ms", 0.0))
        processed = int(runtime_metrics.get("frames_processed", 0))
        if processed <= 0:
            runtime_metrics["avg_processing_ms"] = float(processing_ms)
        else:
            runtime_metrics["avg_processing_ms"] = (
                ((previous_avg * (processed - 1)) + float(processing_ms)) / processed
            )


def _runtime_metrics_snapshot():
    with runtime_metrics_lock:
        snapshot = dict(runtime_metrics)
    snapshot["uptime_seconds"] = max(0.0, time.time() - runtime_started_at)
    return snapshot


def _clamp01(value):
    try:
        return float(np.clip(float(value), 0.0, 1.0))
    except Exception:
        return 0.0


def _normalized_top_scores(emotion_scores, dominant_emotion=None):
    if not emotion_scores:
        return 0.0, 0.0, 0.0
    safe_scores = {k: max(0.0, float(v)) for k, v in emotion_scores.items()}
    if not safe_scores:
        return 0.0, 0.0, 0.0

    total = sum(safe_scores.values())
    if total <= 0:
        return 0.0, 0.0, 0.0

    sorted_items = sorted(safe_scores.items(), key=lambda x: x[1], reverse=True)
    if dominant_emotion and dominant_emotion in safe_scores:
        top1 = safe_scores[dominant_emotion]
        second_candidates = [score for emotion, score in safe_scores.items() if emotion != dominant_emotion]
        top2 = max(second_candidates) if second_candidates else 0.0
    else:
        top1 = float(sorted_items[0][1])
        top2 = float(sorted_items[1][1]) if len(sorted_items) > 1 else 0.0
    return top1, top2, total


def compute_display_confidence(emotion_scores, dominant_emotion=None, fallback_confidence=0.0):
    """
    Confidence calibration for display:
    - relative certainty: top1 vs top2
    - absolute share: top1 vs total
    Keeps gating logic untouched (gating still uses raw model confidence).
    """
    calibration_cfg = CONFIG.get("emotion", {}).get("confidence_calibration", {})
    if not calibration_cfg.get("enabled", True):
        return _clamp01(fallback_confidence)

    top1, top2, total = _normalized_top_scores(emotion_scores, dominant_emotion=dominant_emotion)
    if total <= 0 or top1 <= 0:
        return _clamp01(fallback_confidence)

    relative = top1 / max(top1 + top2, 1e-6)   # 0.5..1.0
    absolute = top1 / total                     # 0..1.0
    abs_weight = float(np.clip(calibration_cfg.get("absolute_weight", 0.2), 0.0, 1.0))
    blended = ((1.0 - abs_weight) * relative) + (abs_weight * absolute)

    temperature = float(np.clip(calibration_cfg.get("temperature", 0.75), 0.3, 2.0))
    calibrated = np.power(max(blended, 1e-6), temperature)
    min_display_conf = float(np.clip(calibration_cfg.get("min_display_confidence", 0.0), 0.0, 1.0))
    return _clamp01(max(calibrated, min_display_conf))


def confidence_from_scores(scores, dominant_emotion):
    if not scores or not dominant_emotion or dominant_emotion not in scores:
        return 0.0
    top = max(0.0, float(scores.get(dominant_emotion, 0.0)))
    total = sum(max(0.0, float(v)) for v in scores.values())
    if total <= 0:
        return _clamp01(top / 100.0 if top > 1.0 else top)
    return _clamp01(top / total)


def runtime_health_snapshot():
    deepface_module_available = bool(get_deepface_module())
    gpu_config = CONFIG.get("gpu_emotion", {})
    gpu_cfg_enabled = bool(gpu_config.get("enabled", False))
    gpu_status = emotion_pipeline.get_gpu_emotion_status(try_initialize=False)
    gpu_available = bool(gpu_status.get("available", False))
    gpu_device = gpu_status.get("device", "unavailable")

    security_cfg = CONFIG.get("security", {})
    calibration_status = runtime_calibration.get_status()
    detector_ready = deepface_module_available or gpu_available
    limiter_ready = bool(rate_limit_storage_uri)

    checks = {
        "detector_ready": detector_ready,
        "limiter_ready": limiter_ready,
        "secret_configured": app.config.get("SECRET_KEY") != "default_insecure_secret_change_me",
    }

    return {
        "status": "ok" if all(checks.values()) else "degraded",
        "checks": checks,
        "runtime": {
            "uptime_seconds": max(0.0, time.time() - runtime_started_at),
            "deepface_available": deepface_module_available,
            "gpu_config_enabled": gpu_cfg_enabled,
            "gpu_available": gpu_available,
            "gpu_device": gpu_device,
            "gpu_init_reason": gpu_status.get("reason", ""),
            "rate_limit_storage_uri": rate_limit_storage_uri,
            "calibration": calibration_status,
            "cors_origins_count": len(cors_origins),
            "session_cookie_secure": bool(security_cfg.get("cookie_secure", False)),
        },
    }


# Compatibility wrappers (existing tests + callsites rely on these names)
def get_deepface_module():
    return emotion_pipeline.get_deepface_module()


def get_gpu_emotion_models():
    return emotion_pipeline.get_gpu_emotion_models()


def preprocess_frame(frame):
    return emotion_pipeline.preprocess_frame(frame)


def _hsemotion_scores_to_output(probabilities, idx_to_class, gpu_config):
    return emotion_pipeline.hsemotion_scores_to_output(probabilities, idx_to_class, gpu_config)


def _apply_gpu_disgust_guardrails(emotion_scores, gpu_config):
    return emotion_pipeline.apply_gpu_disgust_guardrails(emotion_scores, gpu_config)


def detect_emotion_gpu(frame, df_module):
    return emotion_pipeline.detect_emotion_gpu(frame, df_module)


def detect_emotion(frame):
    # Keep this wrapper so tests can patch app.get_deepface_module.
    df_module = get_deepface_module()
    return emotion_pipeline.detect_emotion(frame, df_module=df_module, optimize_speed=args.optimize_speed)


def get_deepseek_client():
    global model_init_complete, deepseek_client
    client = chat_client_manager.get_client()
    model_init_complete = chat_client_manager.model_init_complete
    deepseek_client = client
    return client


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/emotion")
def get_emotion():
    global last_emotion, emotion_confidence, emotion_confidence_raw
    return jsonify(
        {
            "emotion": last_emotion or "unknown",
            "confidence": emotion_confidence,
            "confidence_raw": emotion_confidence_raw,
        }
    )


@app.route("/api/emotion/history")
@limiter.limit("10 per minute")
def get_emotion_history():
    global emotion_history
    with emotion_history_lock:
        history_snapshot = emotion_history[-100:]

    emotion_labels = [entry.get("emotion") for entry in history_snapshot if entry.get("emotion")]
    most_common_emotion = max(set(emotion_labels), key=emotion_labels.count) if emotion_labels else "unknown"

    return jsonify({"history": history_snapshot, "summary": {"most_common": most_common_emotion}})


@app.route("/api/metrics")
@limiter.limit("30 per minute")
def get_runtime_metrics():
    return jsonify(_runtime_metrics_snapshot())


@app.route("/api/health")
@limiter.limit("30 per minute")
def get_health():
    return jsonify(runtime_health_snapshot())


@app.route("/api/readiness")
@limiter.limit("30 per minute")
def get_readiness():
    health = runtime_health_snapshot()
    checks = health.get("checks", {})
    critical_ready = bool(checks.get("detector_ready")) and bool(checks.get("limiter_ready"))
    status_code = 200 if critical_ready else 503
    return jsonify({"ready": critical_ready, "health": health}), status_code


@app.route("/api/calibration/status")
@limiter.limit("30 per minute")
def get_calibration_status():
    return jsonify(runtime_calibration.get_status())


@app.route("/api/calibration/reload", methods=["POST"])
@limiter.limit("10 per minute")
def post_calibration_reload():
    runtime_calibration.load_artifact()
    return jsonify(runtime_calibration.get_status())


def get_session_smoother(sid):
    with smoother_lock:
        if sid not in session_smoothers:
            smoothing_config = CONFIG.get("emotion", {}).get("smoothing", {})
            session_smoothers[sid] = EmotionSmoother(
                alpha=smoothing_config.get("alpha", 0.4),
                confidence_threshold=smoothing_config.get("confidence_threshold", 0.35),
                switch_in_frames=smoothing_config.get("switch_in_frames", 3),
                switch_out_frames=smoothing_config.get("switch_out_frames", 4),
                stable_hold_ratio=smoothing_config.get("stable_hold_ratio", 0.9),
                stale_reset_frames=smoothing_config.get("stale_reset_frames", 15),
                min_dominance_margin=smoothing_config.get("min_dominance_margin", 2.5),
            )
        return session_smoothers[sid]


def remove_session_smoother(sid):
    with smoother_lock:
        session_smoothers.pop(sid, None)
    face_tracker.clear(sid)
    with processing_lock:
        session_processing_flags.pop(sid, None)
    with frame_skip_lock:
        session_frame_counters.pop(sid, None)
        session_last_results.pop(sid, None)
        session_last_smoothed.pop(sid, None)
    with emotion_history_lock:
        session_emotion_state.pop(sid, None)


def normalize_detection_tuple(result):
    if isinstance(result, tuple):
        if len(result) == 5:
            return result
        if len(result) == 4:
            emotion, confidence, face_location, face_detected = result
            return emotion, confidence, face_location, face_detected, {}
    return "error", 0.0, None, False, {}


def process_frame_async(frame, sid):
    global last_emotion, emotion_confidence, emotion_confidence_raw, last_emotion_time

    started = time.perf_counter()
    try:
        frame_height, frame_width = frame.shape[:2]
        emotion_config = CONFIG.get("emotion", {})
        logging_cfg = CONFIG.get("logging", {})
        frame_debug_every_n = int(logging_cfg.get("frame_debug_every_n", 30) or 30)
        frame_debug_every_n = max(1, frame_debug_every_n)
        face_tracked = False

        try:
            analyze_every_n_frames = int(emotion_config.get("analyze_every_n_frames", 1))
        except (TypeError, ValueError):
            analyze_every_n_frames = 1
        analyze_every_n_frames = max(1, analyze_every_n_frames)

        with frame_skip_lock:
            frame_counter = session_frame_counters.get(sid, -1) + 1
            session_frame_counters[sid] = frame_counter
            should_skip = (
                analyze_every_n_frames > 1
                and (frame_counter % analyze_every_n_frames != 0)
                and sid in session_last_results
            )

        if should_skip:
            _increment_metric("frames_skipped")
            with frame_skip_lock:
                emotion, confidence, face_location, face_detected, raw_emotions = session_last_results[sid]
                smoothed_data = session_last_smoothed.get(sid)
            if frame_counter % frame_debug_every_n == 0:
                logger.debug(
                    "Frame skipped for sid=%s (counter=%s, n=%s)",
                    sid,
                    frame_counter,
                    analyze_every_n_frames,
                )
        else:
            _increment_metric("frames_inferred")
            detection_result = normalize_detection_tuple(detect_emotion(frame))
            emotion, confidence, face_location, face_detected, raw_emotions = detection_result

            if raw_emotions:
                adjusted_scores = runtime_calibration.apply_class_bias(raw_emotions)
                if adjusted_scores and adjusted_scores != raw_emotions:
                    _increment_metric("frames_calibrated")
                raw_emotions = adjusted_scores
                emotion = max(raw_emotions, key=raw_emotions.get)
                confidence = confidence_from_scores(raw_emotions, emotion)

            tracked_location, tracked_flag = face_tracker.update(
                sid,
                face_location if face_detected else None,
                frame_width=frame_width,
                frame_height=frame_height,
            )
            if tracked_location is not None:
                face_location = tracked_location
            if tracked_flag and not face_detected:
                face_detected = True
                face_tracked = True
                _increment_metric("frames_face_tracked")

            detection_result = (emotion, confidence, face_location, face_detected, raw_emotions or {})
            with frame_skip_lock:
                session_last_results[sid] = detection_result

            smoother = get_session_smoother(sid)
            did_update = smoother.update(
                raw_emotions or {}, face_detected=face_detected, detection_confidence=confidence
            )
            averaged_scores = smoother.get_averaged_scores()
            avg_emotion, avg_conf = smoother.get_dominant_emotion(averaged_scores)
            smoothed_raw_conf = avg_conf / 100.0 if avg_conf > 1.0 else avg_conf
            smoothed_display_conf = (
                compute_display_confidence(
                    averaged_scores, dominant_emotion=avg_emotion, fallback_confidence=smoothed_raw_conf
                )
                if averaged_scores
                else 0.0
            )
            smoothed_display_conf = runtime_calibration.calibrate_confidence(smoothed_display_conf)

            smoothed_data = {
                "emotion": avg_emotion if averaged_scores else "unknown",
                "confidence": _clamp01(smoothed_display_conf),
                "confidence_raw": _clamp01(smoothed_raw_conf if averaged_scores else 0.0),
                "scores": averaged_scores,
                "updated": bool(did_update),
            }
            if frame_counter % frame_debug_every_n == 0:
                logger.debug(
                    "Smoothed sid=%s emotion=%s conf=%.2f raw=%.2f updated=%s",
                    sid,
                    smoothed_data["emotion"],
                    smoothed_data["confidence"],
                    smoothed_data["confidence_raw"],
                    did_update,
                )

            with frame_skip_lock:
                session_last_smoothed[sid] = smoothed_data

        display_confidence = compute_display_confidence(
            raw_emotions, dominant_emotion=emotion, fallback_confidence=confidence
        )
        display_confidence = runtime_calibration.calibrate_confidence(display_confidence)
        low_conf_threshold = float(
            np.clip(CONFIG.get("emotion", {}).get("low_confidence_threshold", 0.55), 0.0, 1.0)
        )

        last_emotion = emotion
        emotion_confidence_raw = _clamp01(confidence)
        emotion_confidence = display_confidence
        last_emotion_time = time.time()
        with emotion_history_lock:
            session_emotion_state[sid] = {
                "emotion": emotion,
                "confidence": display_confidence,
                "confidence_raw": _clamp01(confidence),
                "updated_at": last_emotion_time,
            }
            if face_detected and emotion not in ("unknown", "error"):
                emotion_history.append(
                    {
                        "sid": sid,
                        "emotion": emotion,
                        "confidence": display_confidence,
                        "confidence_raw": _clamp01(confidence),
                        "timestamp": last_emotion_time,
                    }
                )
                history_limit = int(CONFIG.get("emotion", {}).get("history_size", 100))
                history_limit = max(100, history_limit)
                if len(emotion_history) > history_limit:
                    del emotion_history[:-history_limit]

        if frame_counter % frame_debug_every_n == 0:
            logger.debug(
                "Frame sid=%s emotion=%s conf=%.2f raw=%.2f",
                sid,
                emotion,
                display_confidence,
                _clamp01(confidence),
            )

        socketio.emit(
            "emotion_update",
            {
                "emotion": emotion,
                "confidence": display_confidence,
                "confidence_raw": _clamp01(confidence),
                "raw_emotions": raw_emotions or {},
                "face_detected": face_detected,
                "face_tracked": face_tracked,
                "face_location": face_location,
                "frame_width": frame_width,
                "frame_height": frame_height,
                "smoothed": smoothed_data,
                "quality": {
                    "low_confidence": (not face_detected) or (display_confidence < low_conf_threshold),
                    "low_confidence_threshold": low_conf_threshold,
                },
            },
            room=sid,
        )
    except Exception as exc:
        _increment_metric("frames_failed")
        logger.exception("Error in async frame processing: %s", exc)
    finally:
        _increment_metric("frames_processed")
        _record_processing_time((time.perf_counter() - started) * 1000.0)
        with processing_lock:
            session_processing_flags[sid] = False


@socketio.on("frame")
def handle_frame(data):
    with processing_lock:
        if session_processing_flags.get(request.sid, False):
            _increment_metric("frames_dropped_busy")
            return
        session_processing_flags[request.sid] = True

    try:
        privacy_cfg = CONFIG.get("privacy", {})
        if bool(privacy_cfg.get("require_camera_consent", True)):
            with consent_lock:
                has_consent = bool(session_camera_consent.get(request.sid, False))
            if not has_consent:
                _increment_metric("frames_dropped_no_consent")
                emit("camera_consent_required", {"required": True})
                with processing_lock:
                    session_processing_flags[request.sid] = False
                return

        _increment_metric("frames_received")
        if not isinstance(data, str):
            raise ValueError("Frame payload must be a base64 string.")

        if "," in data:
            img_data = base64.b64decode(data.split(",", 1)[1])
        else:
            img_data = base64.b64decode(data)

        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            _increment_metric("frames_decode_error")
            logger.error("Frame could not be decoded from base64 data.")
            with processing_lock:
                session_processing_flags[request.sid] = False
            return

        executor.submit(process_frame_async, frame, request.sid)
    except Exception as exc:
        _increment_metric("frames_decode_error")
        logger.error("Error queuing frame: %s", exc)
        with processing_lock:
            session_processing_flags[request.sid] = False


def build_emotion_context(sid=None):
    """Gather all available emotion data and format it into a context string for the AI."""
    # --- Current emotion and confidence ---
    with emotion_history_lock:
        if sid:
            state = session_emotion_state.get(sid) or {}
            current = state.get("emotion")
            conf = float(state.get("confidence", 0.0) or 0.0)
            conf_raw = float(state.get("confidence_raw", 0.0) or 0.0)
        else:
            current = last_emotion
            conf = emotion_confidence
            conf_raw = emotion_confidence_raw

        history_snapshot = list(emotion_history[-50:])

    if not current or current in ("unknown", "error"):
        current = last_emotion if last_emotion and last_emotion not in ("unknown", "error") else None

    # --- History analysis ---
    recent_labels = [e.get("emotion") for e in history_snapshot if e.get("emotion")]
    last_n = recent_labels[-10:] if recent_labels else []

    # Dominant emotion (most frequent across full history)
    dominant = None
    if recent_labels:
        dominant = max(set(recent_labels), key=recent_labels.count)

    # Trend: look at last 10 readings for direction of change
    trend_description = None
    if len(last_n) >= 4:
        first_half = last_n[: len(last_n) // 2]
        second_half = last_n[len(last_n) // 2 :]
        first_dom = max(set(first_half), key=first_half.count)
        second_dom = max(set(second_half), key=second_half.count)
        if first_dom != second_dom:
            trend_description = f"shifting from {first_dom} toward {second_dom}"
        else:
            trend_description = f"consistently {first_dom}"

    # Stability: how many unique emotions in last 10 readings
    stability = None
    if last_n:
        unique_count = len(set(last_n))
        if unique_count <= 1:
            stability = "very stable"
        elif unique_count <= 2:
            stability = "mostly stable"
        elif unique_count <= 3:
            stability = "somewhat fluctuating"
        else:
            stability = "highly fluctuating"

    # Smoother EMA scores (if available for this session)
    smoother_info = None
    if sid:
        with smoother_lock:
            smoother = session_smoothers.get(sid)
        if smoother:
            ema = smoother.get_averaged_scores()
            stable_em = smoother.stable_emotion
            if ema:
                top_3 = sorted(ema.items(), key=lambda x: x[1], reverse=True)[:3]
                scores_str = ", ".join(f"{e}: {s:.1f}" for e, s in top_3)
                smoother_info = f"Smoothed EMA scores (top 3): {scores_str}. Stable emotion: {stable_em}."

    # --- Build the context string ---
    parts = []
    if current:
        conf_pct = int(conf * 100) if conf <= 1.0 else int(conf)
        parts.append(f"Current detected emotion: {current} (confidence: {conf_pct}%).")
    else:
        parts.append("No emotion currently detected (camera may be off or face not visible).")

    if dominant and recent_labels:
        count = recent_labels.count(dominant)
        parts.append(f"Dominant emotion this session: {dominant} ({count}/{len(recent_labels)} readings).")

    if trend_description:
        parts.append(f"Recent trend: {trend_description}.")

    if stability:
        parts.append(f"Emotional stability: {stability}.")

    if smoother_info:
        parts.append(smoother_info)

    return "\n".join(parts)


SYSTEM_PROMPT_TEMPLATE = """\
You are Eternix, an empathetic and perceptive AI therapist. You provide a safe, \
non-judgmental space for users to explore their feelings.

EMOTION CONTEXT (from real-time facial analysis):
{emotion_context}

GUIDELINES:
- Weave awareness of the user's emotional state into your responses naturally. \
Do NOT robotically announce the detected emotion (avoid phrases like "I see you are sad"). \
Instead, let it subtly inform your tone and questions.
- If the emotional trend shows a shift (e.g., from calm to tense), gently acknowledge it \
when relevant: "I notice something may have shifted as we've been talking."
- Ask thoughtful, open-ended follow-up questions that invite the user to reflect deeper.
- Keep responses concise and conversational (2-4 sentences). Be warm, not clinical.
- If no emotion is detected, focus entirely on the user's words and respond empathetically.
- Never mention confidence scores, EMA scores, or technical details to the user.\
"""


def get_ai_response_threaded(user_message_content, emotion_context, sid):
    client = get_deepseek_client()
    if not model_init_complete or not client:
        socketio.emit(
            "ai_response",
            {
                "error": "AI model is not initialized or the client is not available. "
                "Please check configuration and logs."
            },
            room=sid,
        )
        return

    try:
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(emotion_context=emotion_context)
        model_to_use = chat_client_manager.get_model_name()
        messages_payload = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message_content},
        ]

        logger.debug("Attempting chat API call in thread. model=%s sid=%s", model_to_use, sid)
        start_time = time.time()
        max_retries = 3
        retry_delay = 1.0
        ai_response = None

        for attempt in range(max_retries):
            try:
                completion = client.chat.completions.create(
                    model=model_to_use, messages=messages_payload, timeout=15
                )
                ai_response = completion.choices[0].message.content
                break
            except Exception as exc:
                logger.warning("Chat API attempt %s/%s failed: %s", attempt + 1, max_retries, exc)
                error_str = str(exc).lower()
                if (
                    "rate limit" in error_str
                    or "429" in error_str
                    or "insufficient balance" in error_str
                    or "402" in error_str
                ):
                    raise
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    raise

        duration = time.time() - start_time
        logger.debug("Chat API success sid=%s duration=%.3fs", sid, duration)
        socketio.emit("ai_response", {"response": ai_response.strip() if ai_response else ""}, room=sid)
    except Exception as exc:
        duration = time.time() - start_time if "start_time" in locals() else 0
        logger.error(
            "Chat API error in thread: %s | duration=%.3fs | model=%s | sid=%s",
            exc,
            duration,
            chat_client_manager.get_model_name(),
            sid,
        )
        logger.exception("Chat API exception details:")
        error_str = str(exc).lower()
        if "insufficient balance" in error_str or "402" in error_str:
            fallback = (
                "I apologize, but I'm currently operating in offline mode due to an API quota issue.\n\n"
                "(System Note: Your AI account has insufficient balance.)\n\n"
                "However, I am still here to listen. How are you feeling right now?"
            )
            socketio.emit("ai_response", {"response": fallback}, room=sid)
        elif "rate limit" in error_str or "429" in error_str or "resource_exhausted" in error_str:
            socketio.emit(
                "ai_response",
                {"response": "I'm currently receiving too many requests. Please wait a moment and try again."},
                room=sid,
            )
        elif "401" in error_str or "unauthorized" in error_str:
            socketio.emit(
                "ai_response",
                {"response": "There seems to be an authentication issue with my AI service."},
                room=sid,
            )
        else:
            socketio.emit(
                "ai_response",
                {"error": "Sorry, I am having trouble connecting to the AI service. Please try again later."},
                room=sid,
            )


@socketio.on("user_message")
def handle_user_message(data):
    try:
        user_message_content = data.get("message", "").strip()
        raw_text = data.get("raw_text", user_message_content)
        if not user_message_content:
            emit("ai_response", {"error": "Empty message"})
            return

        emit("ai_typing", {"status": "typing"})

        emotion_context = build_emotion_context(sid=request.sid)

        thread = threading.Thread(
            target=get_ai_response_threaded,
            args=(raw_text, emotion_context, request.sid),
            daemon=True,
        )
        thread.start()
    except Exception as exc:
        logger.exception("Error handling user message: %s", exc)
        emit("ai_response", {"error": "An internal server error occurred. Please try again."})


@app.route("/api/chat", methods=["POST"])
@limiter.limit("30 per hour")
def chat():
    try:
        data = request.get_json() or {}
        user_message = data.get("message", "").strip()
        if not user_message:
            return jsonify({"error": "Empty message"}), 400

        client = get_deepseek_client()
        if not model_init_complete or not client:
            return jsonify({"error": "AI model not initialized or client not available"}), 503

        emotion_context = build_emotion_context()
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(emotion_context=emotion_context)
        model_to_use = chat_client_manager.get_model_name()
        payload = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
        logger.debug("Attempting chat API call (HTTP endpoint). model=%s", model_to_use)
        completion = client.chat.completions.create(model=model_to_use, messages=payload)
        ai_response = completion.choices[0].message.content
        return jsonify({"response": ai_response})
    except Exception as exc:
        logger.exception("Error in /api/chat endpoint: %s", exc)
        return jsonify({"error": "Internal server error"}), 500


@socketio.on("connect")
def handle_connect():
    logger.info("Client connected sid=%s", request.sid)
    require_consent = bool(CONFIG.get("privacy", {}).get("require_camera_consent", True))
    with consent_lock:
        session_camera_consent[request.sid] = not require_consent


@socketio.on("disconnect")
def handle_disconnect():
    logger.info("Client disconnected sid=%s", request.sid)
    remove_session_smoother(request.sid)
    with consent_lock:
        session_camera_consent.pop(request.sid, None)


@socketio.on("camera_consent")
def handle_camera_consent(data):
    consent = bool((data or {}).get("consent", False))
    with consent_lock:
        session_camera_consent[request.sid] = consent
    emit("camera_consent_ack", {"consent": consent})


@socketio.on("toggle_lock")
def handle_toggle_lock(data):
    locked = data.get("locked", False)
    logger.info("Toggle lock received: %s", "Locked" if locked else "Unlocked")
    emit("lock_status_updated", {"locked": locked}, broadcast=True)


@socketio.on("clear_emotion_state")
def handle_clear_emotion_state():
    sid = request.sid
    remove_session_smoother(sid)
    with emotion_history_lock:
        emotion_history[:] = [entry for entry in emotion_history if entry.get("sid") != sid]
        session_emotion_state.pop(sid, None)
    emit("emotion_memory_cleared", {"cleared": True})


def _find_listening_pids_on_port(port):
    pids = set()
    if os.name == "nt":
        result = subprocess.run(
            ["netstat", "-ano", "-p", "tcp"], capture_output=True, text=True, check=False
        )
        for line in result.stdout.splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            local_addr = parts[1]
            state = parts[3]
            pid_text = parts[4]
            if not local_addr.endswith(f":{port}") or state.upper() != "LISTENING":
                continue
            try:
                pid = int(pid_text)
                if pid != os.getpid():
                    pids.add(pid)
            except ValueError:
                continue
    else:
        # Best effort non-Windows fallback.
        try:
            result = subprocess.run(["lsof", "-ti", f"tcp:{port}"], capture_output=True, text=True, check=False)
            for line in result.stdout.splitlines():
                try:
                    pid = int(line.strip())
                    if pid != os.getpid():
                        pids.add(pid)
                except ValueError:
                    continue
        except Exception:
            pass
    return pids


def stop_existing_port_listener(port):
    pids = _find_listening_pids_on_port(port)
    if not pids:
        return
    for pid in pids:
        try:
            logger.warning("Stopping existing process on port %s: PID %s", port, pid)
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(pid), "/F"], check=False, capture_output=True, text=True)
            else:
                os.kill(pid, 9)
        except Exception as exc:
            logger.warning("Failed to stop PID %s on port %s: %s", pid, port, exc)


if __name__ == "__main__":
    threading.Thread(target=get_deepseek_client, daemon=True).start()

    app_config = CONFIG.get("app", {})
    host = app_config.get("host", "0.0.0.0")
    debug_mode = app_config.get("debug", False)
    port = 5000

    auto_kill = bool(app_config.get("auto_kill_existing_port_process", True))
    if auto_kill:
        stop_existing_port_listener(port)
        time.sleep(0.2)

    logger.info("[START] Starting Eternix...")
    logger.info("Running on http://%s:%s (Debug mode: %s)", host, port, debug_mode)
    if host == "0.0.0.0":
        logger.info("Also accessible on your local network at http://<your-local-ip>:%s", port)
    else:
        logger.info("Visit http://%s:%s in your browser", host, port)

    try:
        socketio.run(
            app,
            host=host,
            port=port,
            debug=debug_mode,
            allow_unsafe_werkzeug=debug_mode,
        )
    except OSError as exc:
        if "Address already in use" in str(exc):
            logger.error("Port 5000 is already in use.")
        else:
            logger.error("Server failed to start: %s", exc)
        sys.exit(1)
    except Exception as exc:
        logger.error("Server failed to start: %s", exc)
        sys.exit(1)
