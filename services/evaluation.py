import json
import math
from datetime import datetime, timezone

import numpy as np


EMOTION_KEYS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
LABEL_ALIASES = {
    "anger": "angry",
    "angry": "angry",
    "disgust": "disgust",
    "fear": "fear",
    "happy": "happy",
    "happiness": "happy",
    "neutral": "neutral",
    "sad": "sad",
    "sadness": "sad",
    "surprise": "surprise",
}


def canonical_label(label):
    if not label:
        return None
    normalized = str(label).strip().lower()
    return LABEL_ALIASES.get(normalized)


def safe_distribution(scores, labels=None):
    labels = labels or EMOTION_KEYS
    if not isinstance(scores, dict):
        return {}
    cleaned = {label: max(0.0, float(scores.get(label, 0.0) or 0.0)) for label in labels}
    total = float(sum(cleaned.values()))
    if total <= 0:
        return {}
    return {label: value / total for label, value in cleaned.items()}


def _extract_label(record):
    for key in ("label", "true_emotion", "ground_truth", "target"):
        value = canonical_label(record.get(key))
        if value:
            return value
    return None


def _extract_scores(record):
    for key in ("scores", "raw_emotions", "emotion_scores", "predicted_scores"):
        scores = record.get(key)
        if isinstance(scores, dict):
            return scores
    return {}


def _extract_predicted_label(record, scores):
    for key in ("predicted_emotion", "dominant_emotion", "prediction", "pred"):
        value = canonical_label(record.get(key))
        if value:
            return value
    distribution = safe_distribution(scores)
    if not distribution:
        return None
    return max(distribution, key=distribution.get)


def _extract_confidence(record, predicted_label, scores):
    for key in ("confidence", "predicted_confidence", "probability"):
        if key in record:
            value = float(record.get(key) or 0.0)
            if value > 1.0:
                value /= 100.0
            return float(np.clip(value, 0.0, 1.0))

    distribution = safe_distribution(scores)
    if not distribution or not predicted_label:
        return 0.0
    return float(np.clip(distribution.get(predicted_label, 0.0), 0.0, 1.0))


def parse_dataset_records(raw_records):
    parsed = []
    for record in raw_records:
        if not isinstance(record, dict):
            continue
        true_label = _extract_label(record)
        if not true_label:
            continue
        scores = _extract_scores(record)
        predicted_label = _extract_predicted_label(record, scores)
        if not predicted_label:
            continue
        confidence = _extract_confidence(record, predicted_label, scores)
        parsed.append(
            {
                "true_label": true_label,
                "predicted_label": predicted_label,
                "predicted_confidence": confidence,
                "scores": scores,
            }
        )
    return parsed


def load_records_from_path(path):
    if path.endswith(".jsonl"):
        raw_records = []
        with open(path, "r", encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                raw_records.append(json.loads(line))
        return parse_dataset_records(raw_records)

    with open(path, "r", encoding="utf-8-sig") as f:
        payload = json.load(f)
    if isinstance(payload, dict):
        payload = payload.get("records", [])
    if not isinstance(payload, list):
        raise ValueError("Dataset must be a JSON list or JSONL file.")
    return parse_dataset_records(payload)


def _empty_confusion(labels):
    return {true_label: {pred_label: 0 for pred_label in labels} for true_label in labels}


def confusion_matrix(records, labels=None):
    labels = labels or EMOTION_KEYS
    matrix = _empty_confusion(labels)
    for sample in records:
        true_label = sample.get("true_label")
        pred_label = sample.get("predicted_label")
        if true_label not in matrix or pred_label not in matrix[true_label]:
            continue
        matrix[true_label][pred_label] += 1
    return matrix


def _precision_recall_f1(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0:
        return precision, recall, 0.0
    f1 = 2.0 * precision * recall / (precision + recall)
    return precision, recall, f1


def classification_report(records, labels=None):
    labels = labels or EMOTION_KEYS
    matrix = confusion_matrix(records, labels=labels)
    total = max(1, len(records))
    correct = sum(matrix[label][label] for label in labels)
    accuracy = correct / total

    per_class = {}
    macro_f1 = 0.0
    weighted_f1 = 0.0
    total_support = 0

    for label in labels:
        tp = matrix[label][label]
        fp = sum(matrix[other][label] for other in labels if other != label)
        fn = sum(matrix[label][other] for other in labels if other != label)
        support = tp + fn
        precision, recall, f1 = _precision_recall_f1(tp, fp, fn)
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
        macro_f1 += f1
        weighted_f1 += f1 * support
        total_support += support

    macro_f1 /= max(1, len(labels))
    weighted_f1 = weighted_f1 / total_support if total_support > 0 else 0.0

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class": per_class,
        "confusion_matrix": matrix,
        "samples": len(records),
    }


def _sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def _safe_logit(p):
    p = float(np.clip(p, 1e-6, 1.0 - 1e-6))
    return math.log(p / (1.0 - p))


def _calibrate_confidence(probability, slope, intercept):
    return _sigmoid((slope * _safe_logit(probability)) + intercept)


def _brier_score(confidences, correct_flags):
    if len(confidences) == 0:
        return 0.0
    conf = np.array(confidences, dtype=np.float32)
    corr = np.array(correct_flags, dtype=np.float32)
    return float(np.mean((conf - corr) ** 2))


def _expected_calibration_error(confidences, correct_flags, bins=10):
    if len(confidences) == 0:
        return 0.0
    conf = np.array(confidences, dtype=np.float32)
    corr = np.array(correct_flags, dtype=np.float32)
    ece = 0.0
    for idx in range(bins):
        left = idx / bins
        right = (idx + 1) / bins
        if idx == bins - 1:
            mask = (conf >= left) & (conf <= right)
        else:
            mask = (conf >= left) & (conf < right)
        if not np.any(mask):
            continue
        bin_conf = float(np.mean(conf[mask]))
        bin_acc = float(np.mean(corr[mask]))
        ece += abs(bin_conf - bin_acc) * (float(np.sum(mask)) / len(conf))
    return float(ece)


def fit_confidence_calibration(records):
    confidences = []
    correct_flags = []
    for sample in records:
        confidence = float(np.clip(sample.get("predicted_confidence", 0.0), 0.0, 1.0))
        correct = 1.0 if sample.get("predicted_label") == sample.get("true_label") else 0.0
        confidences.append(confidence)
        correct_flags.append(correct)

    if len(confidences) < 30:
        return {
            "enabled": False,
            "reason": "insufficient_samples",
            "samples": len(confidences),
            "slope": 1.0,
            "intercept": 0.0,
            "brier_before": _brier_score(confidences, correct_flags),
            "brier_after": _brier_score(confidences, correct_flags),
            "ece_before": _expected_calibration_error(confidences, correct_flags),
            "ece_after": _expected_calibration_error(confidences, correct_flags),
        }

    base_brier = _brier_score(confidences, correct_flags)
    base_ece = _expected_calibration_error(confidences, correct_flags)

    best = {"slope": 1.0, "intercept": 0.0, "brier": base_brier}
    for slope in np.linspace(0.6, 2.4, 37):
        for intercept in np.linspace(-1.2, 1.2, 49):
            calibrated = [_calibrate_confidence(c, float(slope), float(intercept)) for c in confidences]
            brier = _brier_score(calibrated, correct_flags)
            if brier < best["brier"]:
                best = {"slope": float(slope), "intercept": float(intercept), "brier": float(brier)}

    calibrated = [_calibrate_confidence(c, best["slope"], best["intercept"]) for c in confidences]
    return {
        "enabled": True,
        "method": "logit_affine_grid",
        "samples": len(confidences),
        "slope": best["slope"],
        "intercept": best["intercept"],
        "brier_before": base_brier,
        "brier_after": _brier_score(calibrated, correct_flags),
        "ece_before": base_ece,
        "ece_after": _expected_calibration_error(calibrated, correct_flags),
    }


def fit_class_bias(records, labels=None, min_multiplier=0.65, max_multiplier=1.35, smoothing=0.6):
    labels = labels or EMOTION_KEYS
    true_counts = {label: 0 for label in labels}
    pred_counts = {label: 0 for label in labels}

    for sample in records:
        true_label = sample.get("true_label")
        pred_label = sample.get("predicted_label")
        if true_label in true_counts:
            true_counts[true_label] += 1
        if pred_label in pred_counts:
            pred_counts[pred_label] += 1

    total_true = float(sum(true_counts.values()))
    total_pred = float(sum(pred_counts.values()))
    multipliers = {}
    for label in labels:
        true_freq = (true_counts[label] / total_true) if total_true > 0 else 0.0
        pred_freq = (pred_counts[label] / total_pred) if total_pred > 0 else 0.0
        if pred_freq <= 0:
            ratio = 1.0
        else:
            ratio = true_freq / pred_freq
        ratio = 1.0 + (smoothing * (ratio - 1.0))
        multipliers[label] = float(np.clip(ratio, min_multiplier, max_multiplier))

    return {
        "enabled": True,
        "method": "frequency_ratio",
        "samples": len(records),
        "true_counts": true_counts,
        "predicted_counts": pred_counts,
        "multipliers": multipliers,
    }


def build_calibration_artifact(records, labels=None):
    labels = labels or EMOTION_KEYS
    report = classification_report(records, labels=labels)
    confidence = fit_confidence_calibration(records)
    class_bias = fit_class_bias(records, labels=labels)

    return {
        "version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "labels": labels,
        "dataset_samples": len(records),
        "metrics": report,
        "confidence": confidence,
        "class_bias": class_bias,
    }
