import logging
import threading
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import cv2
import numpy as np


@dataclass
class EmotionDetectionResult:
    dominant_emotion: str = "unknown"
    confidence: float = 0.0
    face_location: Optional[Dict[str, int]] = None
    face_detected: bool = False
    emotion_scores: Dict[str, float] = field(default_factory=dict)

    def as_tuple(self) -> Tuple[str, float, Optional[Dict[str, int]], bool, Dict[str, float]]:
        return (
            self.dominant_emotion,
            self.confidence,
            self.face_location,
            self.face_detected,
            self.emotion_scores,
        )


class EmotionPipeline:
    """All preprocessing + emotion inference logic in one testable component."""

    HSEMOTION_TO_OUTPUT = {
        "anger": "angry",
        "angry": "angry",
        "disgust": "disgust",
        "fear": "fear",
        "happiness": "happy",
        "happy": "happy",
        "neutral": "neutral",
        "sadness": "sad",
        "sad": "sad",
        "surprise": "surprise",
    }
    OUTPUT_EMOTION_KEYS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

    def __init__(self, config, optimize_speed=False, logger=None):
        self.config = config
        self.optimize_speed = optimize_speed
        self.logger = logger

        self.deepface_module = None
        self.deepface_init_lock = threading.Lock()

        self.gamma_lut_input = np.arange(256, dtype=np.float32) / 255.0
        self.gamma_lut_cache = {}

        self.gpu_emotion_lock = threading.Lock()
        self.gpu_emotion_models = {
            "initialized": False,
            "available": False,
            "reason": "",
            "device": "cpu",
            "recognizer": None,
            "torch": None,
        }

    def set_optimize_speed(self, optimize_speed):
        self.optimize_speed = bool(optimize_speed)

    def _log_debug(self, message):
        if self.logger:
            self.logger.debug(message)

    def _log_info(self, message):
        if self.logger:
            self.logger.info(message)

    def _log_warning(self, message):
        if self.logger:
            self.logger.warning(message)

    def _log_error(self, message):
        if self.logger:
            self.logger.error(message)

    def normalize_emotion_scores(self, scores):
        normalized = {key: max(0.0, float(value)) for key, value in scores.items()}
        total = sum(normalized.values())
        if total <= 0:
            return normalized
        scale = 100.0 / total
        return {key: value * scale for key, value in normalized.items()}

    def resolve_hsemotion_label(self, label, gpu_config):
        if label == "contempt":
            contempt_target = str(gpu_config.get("contempt_maps_to", "neutral")).strip().lower()
            if contempt_target not in self.OUTPUT_EMOTION_KEYS:
                contempt_target = "neutral"
            return contempt_target
        return self.HSEMOTION_TO_OUTPUT.get(label)

    def resolve_disgust_guardrails(self, runtime_config=None):
        emotion_guardrails = self.config.get("emotion", {}).get("disgust_guardrails")
        if isinstance(emotion_guardrails, dict):
            return {"disgust_guardrails": emotion_guardrails}

        cfg = runtime_config if isinstance(runtime_config, dict) else {}
        runtime_guardrails = cfg.get("disgust_guardrails")
        if isinstance(runtime_guardrails, dict):
            return {"disgust_guardrails": runtime_guardrails}

        gpu_guardrails = self.config.get("gpu_emotion", {}).get("disgust_guardrails")
        if isinstance(gpu_guardrails, dict):
            return {"disgust_guardrails": gpu_guardrails}

        return {"disgust_guardrails": {}}

    def apply_gpu_disgust_guardrails(self, emotion_scores, gpu_config):
        guardrails = gpu_config.get("disgust_guardrails", {})
        if not guardrails.get("enabled", True):
            return emotion_scores
        if not emotion_scores:
            return emotion_scores

        dominant = max(emotion_scores, key=emotion_scores.get)
        if dominant != "disgust":
            return emotion_scores

        disgust_score = float(emotion_scores.get("disgust", 0.0))
        happy_score = float(emotion_scores.get("happy", 0.0))
        if disgust_score <= 0:
            return emotion_scores

        min_disgust_score = float(np.clip(guardrails.get("min_disgust_score", 45.0), 0.0, 100.0))
        happy_rescue_ratio = float(np.clip(guardrails.get("happy_rescue_ratio", 0.88), 0.0, 1.5))
        second_choice_ratio = float(np.clip(guardrails.get("second_choice_ratio", 0.92), 0.0, 1.5))

        target_emotion = None
        if happy_score > 0 and happy_score >= (disgust_score * happy_rescue_ratio):
            target_emotion = "happy"
        elif disgust_score < min_disgust_score:
            second_emotion, second_score = max(
                ((emotion, score) for emotion, score in emotion_scores.items() if emotion != "disgust"),
                key=lambda pair: pair[1],
                default=("neutral", 0.0),
            )
            if second_score >= (disgust_score * second_choice_ratio):
                target_emotion = second_emotion

        if not target_emotion:
            return emotion_scores

        adjusted_scores = dict(emotion_scores)
        epsilon = 0.01
        previous_target_score = float(adjusted_scores.get(target_emotion, 0.0))
        adjusted_scores[target_emotion] = disgust_score + epsilon
        adjusted_scores["disgust"] = max(previous_target_score - epsilon, 0.0)
        return self.normalize_emotion_scores(adjusted_scores)

    def hsemotion_scores_to_output(self, probabilities, idx_to_class, gpu_config):
        scores = {key: 0.0 for key in self.OUTPUT_EMOTION_KEYS}
        if probabilities is None:
            return scores

        for idx, prob in enumerate(probabilities):
            label = str(idx_to_class.get(idx, "")).strip().lower()
            mapped = self.resolve_hsemotion_label(label, gpu_config)
            if mapped:
                scores[mapped] += max(0.0, float(prob))
        return self.normalize_emotion_scores(scores)

    def to_uint8_rgb(self, face_image, fallback_bgr=None):
        if isinstance(face_image, np.ndarray) and face_image.size > 0:
            rgb = face_image
            if rgb.ndim == 2:
                rgb = cv2.cvtColor(rgb, cv2.COLOR_GRAY2RGB)
            elif rgb.ndim == 3 and rgb.shape[2] == 4:
                rgb = rgb[:, :, :3]
            if rgb.dtype != np.uint8:
                max_value = float(np.max(rgb)) if rgb.size else 0.0
                if max_value <= 1.0:
                    rgb = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
                else:
                    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
            return rgb

        if isinstance(fallback_bgr, np.ndarray) and fallback_bgr.size > 0:
            return cv2.cvtColor(fallback_bgr, cv2.COLOR_BGR2RGB)
        return None

    def get_deepface_module(self):
        if self.deepface_module is None:
            with self.deepface_init_lock:
                if self.deepface_module is None:
                    self._log_debug("Attempting to import DeepFace on demand...")
                    try:
                        from deepface import DeepFace as DF_Module

                        self.deepface_module = DF_Module
                        self._log_info("DeepFace module imported successfully on demand.")
                    except ImportError:
                        self._log_error("DeepFace library not found. Emotion detection will be disabled.")
                        self.deepface_module = False
                    except Exception as exc:
                        self._log_error(f"DeepFace import initialization failed: {exc}")
                        if self.logger:
                            self.logger.exception("DeepFace import traceback:")
                        self.deepface_module = False
        return self.deepface_module if self.deepface_module is not False else None

    def get_gpu_emotion_models(self):
        with self.gpu_emotion_lock:
            if self.gpu_emotion_models["initialized"]:
                return self.gpu_emotion_models if self.gpu_emotion_models["available"] else None

            self.gpu_emotion_models["initialized"] = True
            try:
                from hsemotion.facial_emotions import HSEmotionRecognizer
                import torch

                gpu_config = self.config.get("gpu_emotion", {})
                requested_device = str(gpu_config.get("device", "cuda")).lower()
                model_name = gpu_config.get("hsemotion_model", "enet_b2_8")

                device = "cuda" if requested_device == "cuda" and torch.cuda.is_available() else "cpu"

                original_torch_load = torch.load
                torch.load = lambda *args, **kwargs: original_torch_load(  # noqa: E731
                    *args, **({"weights_only": False} | kwargs)
                )
                try:
                    recognizer = HSEmotionRecognizer(model_name=model_name, device=device)
                finally:
                    torch.load = original_torch_load

                self.gpu_emotion_models.update(
                    {
                        "available": True,
                        "reason": "",
                        "device": device,
                        "recognizer": recognizer,
                        "torch": torch,
                    }
                )
                self._log_info(f"GPU emotion models initialized on device={device}, model={model_name}")
            except Exception as exc:
                self.gpu_emotion_models.update(
                    {
                        "available": False,
                        "reason": str(exc)[:200],
                        "device": "cpu",
                        "recognizer": None,
                        "torch": None,
                    }
                )
                self._log_warning(f"GPU emotion model initialization failed: {self.gpu_emotion_models['reason']}")

        return self.gpu_emotion_models if self.gpu_emotion_models["available"] else None

    def get_gpu_emotion_status(self, try_initialize=False):
        if try_initialize:
            self.get_gpu_emotion_models()
        with self.gpu_emotion_lock:
            return {
                "initialized": bool(self.gpu_emotion_models.get("initialized", False)),
                "available": bool(self.gpu_emotion_models.get("available", False)),
                "reason": str(self.gpu_emotion_models.get("reason", "")),
                "device": str(self.gpu_emotion_models.get("device", "cpu")),
            }

    def preprocess_frame(self, frame):
        preprocess_config = self.config.get("image_preprocessing", {})
        if not preprocess_config.get("enabled", False):
            return frame

        processed = frame.copy()

        gamma_config = preprocess_config.get("gamma_correction", {})
        clahe_config = preprocess_config.get("clahe", {})
        gamma_enabled = gamma_config.get("enabled", False)
        clahe_enabled = clahe_config.get("enabled", False)

        if gamma_enabled or clahe_enabled:
            try:
                lab = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB)
                l_channel, a_channel, b_channel = cv2.split(lab)

                if gamma_enabled:
                    target_brightness = float(gamma_config.get("target_brightness", 127))
                    target_brightness = float(np.clip(target_brightness, 1.0, 254.0))

                    mean_luminance = float(np.mean(l_channel))
                    mean_luminance = float(np.clip(mean_luminance, 1.0, 254.0))

                    gamma = np.log(target_brightness / 255.0) / np.log(mean_luminance / 255.0)

                    if np.isfinite(gamma) and gamma > 0:
                        gamma_key = round(float(gamma), 4)
                        gamma_lut = self.gamma_lut_cache.get(gamma_key)
                        if gamma_lut is None:
                            gamma_curve = np.power(self.gamma_lut_input, gamma_key)
                            gamma_lut = np.clip(gamma_curve * 255.0, 0, 255).astype(np.uint8)
                            if len(self.gamma_lut_cache) > 512:
                                self.gamma_lut_cache.clear()
                            self.gamma_lut_cache[gamma_key] = gamma_lut

                        l_channel = cv2.LUT(l_channel, gamma_lut)
                    else:
                        self._log_debug(f"Gamma correction skipped due to invalid gamma: {gamma}")

                if clahe_enabled:
                    clip_limit = clahe_config.get("clip_limit", 2.0)
                    tile_size = clahe_config.get("tile_grid_size", [8, 8])
                    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tuple(tile_size))
                    l_channel = clahe.apply(l_channel)

                lab = cv2.merge((l_channel, a_channel, b_channel))
                processed = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            except Exception as exc:
                self._log_debug(f"LAB preprocessing failed: {exc}")

        denoise_config = preprocess_config.get("denoise", {})
        if denoise_config.get("enabled", False):
            try:
                strength = denoise_config.get("strength", 10)
                processed = cv2.fastNlMeansDenoisingColored(processed, None, strength, strength, 7, 21)
            except Exception as exc:
                self._log_debug(f"Denoising preprocessing failed: {exc}")

        sharpen_config = preprocess_config.get("sharpen", {})
        if sharpen_config.get("enabled", False):
            try:
                kernel_strength = sharpen_config.get("kernel_strength", 0.5)
                kernel = np.array(
                    [
                        [0, -kernel_strength, 0],
                        [-kernel_strength, 1 + 4 * kernel_strength, -kernel_strength],
                        [0, -kernel_strength, 0],
                    ]
                )
                processed = cv2.filter2D(processed, -1, kernel)
            except Exception as exc:
                self._log_debug(f"Sharpening preprocessing failed: {exc}")

        return processed

    def detect_emotion_gpu(self, frame, df_module):
        gpu_models = self.get_gpu_emotion_models()
        if not gpu_models:
            return None

        try:
            gpu_config = self.config.get("gpu_emotion", {})
            deepface_config = self.config.get("deepface", {})
            align_faces = deepface_config.get("align_faces", True)
            detector_backend = gpu_config.get("detector_backend", "yolov8")
            expand_percentage = int(gpu_config.get("expand_percentage", 15))
            expand_percentage = int(np.clip(expand_percentage, 0, 60))

            processed_frame = self.preprocess_frame(frame)

            faces = df_module.extract_faces(
                img_path=processed_frame,
                detector_backend=detector_backend,
                enforce_detection=False,
                align=align_faces,
                expand_percentage=expand_percentage,
                color_face="rgb",
                normalize_face=False,
            )

            if not faces:
                return EmotionDetectionResult().as_tuple()

            min_face_confidence = float(gpu_config.get("min_face_confidence", 0.35))
            min_face_confidence = float(np.clip(min_face_confidence, 0.0, 1.0))

            best_face = None
            best_score = -1.0
            for detected in faces:
                facial_area = detected.get("facial_area") or {}
                width = int(facial_area.get("w", 0) or 0)
                height = int(facial_area.get("h", 0) or 0)
                if width <= 0 or height <= 0:
                    continue

                confidence = float(detected.get("confidence", 0.0) or 0.0)
                if confidence > 1.0:
                    confidence /= 100.0
                if confidence < min_face_confidence:
                    continue

                ranking = confidence * (width * height)
                if ranking > best_score:
                    best_score = ranking
                    best_face = detected

            if best_face is None:
                return EmotionDetectionResult().as_tuple()

            facial_area = best_face.get("facial_area") or {}
            x = int(facial_area.get("x", 0) or 0)
            y = int(facial_area.get("y", 0) or 0)
            width = int(facial_area.get("w", 0) or 0)
            height = int(facial_area.get("h", 0) or 0)
            face_location_data = {"x": x, "y": y, "w": width, "h": height}

            fallback_crop = None
            if width > 0 and height > 0:
                x2 = min(x + width, processed_frame.shape[1])
                y2 = min(y + height, processed_frame.shape[0])
                x = max(x, 0)
                y = max(y, 0)
                if x2 > x and y2 > y:
                    fallback_crop = processed_frame[y:y2, x:x2]

            rgb_face = self.to_uint8_rgb(best_face.get("face"), fallback_crop)
            if rgb_face is None:
                return EmotionDetectionResult(face_location=face_location_data).as_tuple()

            recognizer = gpu_models["recognizer"]
            _, probabilities = recognizer.predict_emotions(rgb_face, logits=False)
            emotion_scores = self.hsemotion_scores_to_output(probabilities, recognizer.idx_to_class, gpu_config)
            guardrail_config = self.resolve_disgust_guardrails(gpu_config)
            emotion_scores = self.apply_gpu_disgust_guardrails(emotion_scores, guardrail_config)

            dominant_emotion = max(emotion_scores, key=emotion_scores.get)
            confidence = float(emotion_scores.get(dominant_emotion, 0.0)) / 100.0
            return EmotionDetectionResult(
                dominant_emotion=dominant_emotion,
                confidence=confidence,
                face_location=face_location_data,
                face_detected=True,
                emotion_scores=emotion_scores,
            ).as_tuple()
        except Exception as exc:
            self._log_debug(f"GPU emotion path failed, falling back to DeepFace analyze: {str(exc)[:200]}")
            return None

    def detect_emotion(self, frame, df_module=None, optimize_speed=None):
        if df_module is None:
            df_module = self.get_deepface_module()

        if not df_module:
            self._log_debug("DeepFace module not available or failed to initialize. Emotion detection skipped.")
            return EmotionDetectionResult().as_tuple()

        try:
            if frame is None:
                self._log_debug("detect_emotion called with None frame.")
                return EmotionDetectionResult().as_tuple()

            gpu_config = self.config.get("gpu_emotion", {})
            if gpu_config.get("enabled", False):
                gpu_result = self.detect_emotion_gpu(frame, df_module)
                if gpu_result is not None:
                    return gpu_result

            processed_frame = self.preprocess_frame(frame)
            img_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)

            deepface_config = self.config.get("deepface", {})
            align_faces = deepface_config.get("align_faces", True)
            detection_strategy = deepface_config.get("detection_strategy", "balanced")
            optimize = self.optimize_speed if optimize_speed is None else bool(optimize_speed)

            if optimize or detection_strategy == "speed_priority":
                detector_backends = ["opencv"]
            elif detection_strategy == "accuracy_priority":
                detector_backends = deepface_config.get("detector_backends", ["retinaface", "mtcnn", "ssd", "opencv"])
            else:
                detector_backends = ["opencv", "ssd"]

            results = None
            face_detected_flag = False

            for backend in detector_backends:
                try:
                    current_results_list = df_module.analyze(
                        img_path=img_rgb,
                        actions=["emotion"],
                        enforce_detection=True,
                        detector_backend=backend,
                        align=align_faces,
                        silent=True,
                    )
                    if current_results_list and len(current_results_list) > 0:
                        first_result = current_results_list[0]
                        if "emotion" in first_result:
                            results = first_result
                            face_detected_flag = True
                            self._log_debug(
                                f"Face detected with {backend} (strict). Emotion: {results.get('dominant_emotion')}"
                            )
                            break
                except ValueError:
                    continue
                except Exception as backend_exc:
                    self._log_debug(
                        f"Backend '{backend}' error: {type(backend_exc).__name__}: {str(backend_exc)[:150]}"
                    )
                    continue

            if not face_detected_flag:
                for backend in detector_backends:
                    try:
                        current_results_list = df_module.analyze(
                            img_path=img_rgb,
                            actions=["emotion"],
                            enforce_detection=False,
                            detector_backend=backend,
                            align=align_faces,
                            silent=True,
                        )

                        if current_results_list and len(current_results_list) > 0:
                            first_result = current_results_list[0]
                            if "emotion" in first_result:
                                region = first_result.get("region", {})
                                frame_h, frame_w = img_rgb.shape[:2]
                                region_w = region.get("w", 0)
                                region_h = region.get("h", 0)
                                if region_w > 0 and region_h > 0 and region_w < frame_w * 0.9 and region_h < frame_h * 0.9:
                                    results = first_result
                                    face_detected_flag = True
                                    self._log_debug(
                                        f"Face detected with {backend} (relaxed). Emotion: {results.get('dominant_emotion')}"
                                    )
                                    break
                    except ValueError:
                        continue
                    except Exception as backend_exc:
                        self._log_debug(f"Backend {backend} (relaxed) error: {str(backend_exc)[:100]}")
                        continue

            if not results or not face_detected_flag:
                return EmotionDetectionResult().as_tuple()

            emotions = results.get("emotion")
            if not emotions:
                return EmotionDetectionResult().as_tuple()

            emotions = self.normalize_emotion_scores(emotions)
            guardrail_config = self.resolve_disgust_guardrails()
            emotions = self.apply_gpu_disgust_guardrails(emotions, guardrail_config)
            dominant_emotion = max(emotions, key=emotions.get)
            confidence = float(emotions.get(dominant_emotion, 0.0)) / 100.0

            face_region = results.get("region")
            face_location_data = None
            if face_region:
                x = face_region["x"]
                y = face_region["y"]
                width = face_region["w"]
                height = face_region["h"]
                face_location_data = {"x": x, "y": y, "w": width, "h": height}

            return EmotionDetectionResult(
                dominant_emotion=dominant_emotion,
                confidence=confidence,
                face_location=face_location_data,
                face_detected=face_detected_flag,
                emotion_scores=emotions,
            ).as_tuple()
        except Exception as exc:
            self._log_error(f"Unhandled exception in detect_emotion: {exc}")
            if self.logger:
                self.logger.exception("detect_emotion traceback:")
            return EmotionDetectionResult(dominant_emotion="error").as_tuple()
