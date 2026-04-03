import json
import math
import os
import threading

import numpy as np


class RuntimeCalibration:
    """Applies optional data-fitted confidence calibration + class bias scaling."""

    def __init__(self, config, logger=None):
        self.config = config
        self.logger = logger
        self.lock = threading.Lock()
        self.artifact_path = None
        self.confidence_enabled = False
        self.confidence_slope = 1.0
        self.confidence_intercept = 0.0
        self.class_bias_enabled = False
        self.class_multipliers = {}
        self.last_error = ""
        self.loaded = False
        self.load_artifact()

    def _log_info(self, msg):
        if self.logger:
            self.logger.info(msg)

    def _log_warning(self, msg):
        if self.logger:
            self.logger.warning(msg)

    def _log_debug(self, msg):
        if self.logger:
            self.logger.debug(msg)

    def _clamp01(self, value):
        try:
            return float(np.clip(float(value), 0.0, 1.0))
        except Exception:
            return 0.0

    def _safe_logit(self, probability):
        probability = float(np.clip(probability, 1e-6, 1.0 - 1e-6))
        return math.log(probability / (1.0 - probability))

    def _sigmoid(self, value):
        return 1.0 / (1.0 + math.exp(-value))

    def _config(self):
        emotion_cfg = self.config.get("emotion", {})
        return emotion_cfg.get("data_calibration", {})

    def load_artifact(self):
        with self.lock:
            cfg = self._config()
            enabled = bool(cfg.get("enabled", False))
            self.artifact_path = cfg.get("artifact_path", "artifacts/emotion_calibration.json")

            # Reset to safe defaults every reload.
            self.loaded = False
            self.last_error = ""
            self.confidence_enabled = False
            self.confidence_slope = 1.0
            self.confidence_intercept = 0.0
            self.class_bias_enabled = False
            self.class_multipliers = {}

            if not enabled:
                return

            if not os.path.exists(self.artifact_path):
                self.last_error = f"artifact_not_found:{self.artifact_path}"
                self._log_warning(f"Calibration artifact missing at {self.artifact_path}; using defaults.")
                return

            try:
                with open(self.artifact_path, "r", encoding="utf-8") as f:
                    artifact = json.load(f)

                confidence_cfg = artifact.get("confidence", {})
                if confidence_cfg.get("enabled", False):
                    self.confidence_enabled = True
                    self.confidence_slope = float(confidence_cfg.get("slope", 1.0))
                    self.confidence_intercept = float(confidence_cfg.get("intercept", 0.0))

                class_bias_cfg = artifact.get("class_bias", {})
                multipliers = class_bias_cfg.get("multipliers", {})
                if class_bias_cfg.get("enabled", False) and isinstance(multipliers, dict):
                    safe_multipliers = {}
                    for key, value in multipliers.items():
                        try:
                            safe_multipliers[str(key).strip().lower()] = float(
                                np.clip(float(value), 0.25, 4.0)
                            )
                        except Exception:
                            continue
                    if safe_multipliers:
                        self.class_bias_enabled = True
                        self.class_multipliers = safe_multipliers

                self.loaded = True
                self._log_info(f"Loaded runtime calibration artifact: {self.artifact_path}")
            except Exception as exc:
                self.last_error = f"artifact_parse_error:{str(exc)[:180]}"
                self._log_warning(
                    f"Failed loading calibration artifact {self.artifact_path}: {str(exc)[:180]}"
                )

    def calibrate_confidence(self, confidence):
        with self.lock:
            p = self._clamp01(confidence)
            if not self.confidence_enabled:
                return p
            transformed = self._sigmoid(
                (self.confidence_slope * self._safe_logit(p)) + self.confidence_intercept
            )
            return self._clamp01(transformed)

    def apply_class_bias(self, scores):
        with self.lock:
            if not isinstance(scores, dict):
                return {}
            if not self.class_bias_enabled or not self.class_multipliers:
                return {k: float(v) for k, v in scores.items()}

            adjusted = {}
            original_total = 0.0
            for emotion, value in scores.items():
                numeric = max(0.0, float(value))
                original_total += numeric
                multiplier = self.class_multipliers.get(str(emotion).strip().lower(), 1.0)
                adjusted[emotion] = numeric * multiplier

            adjusted_total = float(sum(adjusted.values()))
            if original_total > 0 and adjusted_total > 0:
                scale = original_total / adjusted_total
                adjusted = {k: max(0.0, v * scale) for k, v in adjusted.items()}
            return adjusted

    def get_status(self):
        with self.lock:
            return {
                "enabled": bool(self._config().get("enabled", False)),
                "artifact_path": self.artifact_path,
                "loaded": self.loaded,
                "last_error": self.last_error,
                "confidence_enabled": self.confidence_enabled,
                "class_bias_enabled": self.class_bias_enabled,
            }

