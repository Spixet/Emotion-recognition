import argparse
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from services.evaluation import build_calibration_artifact, load_records_from_path  # noqa: E402


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate emotion predictions against labeled data and build calibration artifact."
    )
    parser.add_argument("--input", required=True, help="Path to JSON or JSONL dataset file.")
    parser.add_argument(
        "--output",
        default=os.path.join("artifacts", "emotion_calibration.json"),
        help="Output artifact JSON path.",
    )
    args = parser.parse_args()

    records = load_records_from_path(args.input)
    if not records:
        print("[ERROR] No valid records found in dataset. Check labels/scores format.")
        return 1

    artifact = build_calibration_artifact(records)

    output_dir = os.path.dirname(args.output) or "."
    os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2)

    metrics = artifact.get("metrics", {})
    confidence = artifact.get("confidence", {})
    print(f"[OK] Evaluated samples: {artifact.get('dataset_samples', 0)}")
    print(f"[OK] Accuracy: {metrics.get('accuracy', 0.0):.4f}")
    print(f"[OK] Macro F1: {metrics.get('macro_f1', 0.0):.4f}")
    print(f"[OK] Confidence Brier before/after: {confidence.get('brier_before', 0.0):.4f} -> {confidence.get('brier_after', 0.0):.4f}")
    print(f"[OK] Artifact saved to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

