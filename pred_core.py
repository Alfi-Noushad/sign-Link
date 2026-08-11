from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

from sign_utils import FEATURE_VERSION, normalize_landmarks


MODEL_PATH = Path("sign_model.pkl")
MIN_CONFIDENCE = 0.80
AS_THUMB_RADIUS_THRESHOLD = 0.94


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "sign_model.pkl was not found. Add the model file to enable sign-to-text."
        )

    artifact = joblib.load(MODEL_PATH)
    if not isinstance(artifact, dict) or "model" not in artifact:
        raise RuntimeError("sign_model.pkl is not in the expected format.")

    if artifact.get("version") != FEATURE_VERSION:
        raise RuntimeError(
            "sign_model.pkl does not match the current feature version."
        )

    model = artifact["model"]
    classes = getattr(model, "classes_", None)
    if classes is None:
        classes = model.named_steps["logisticregression"].classes_

    return model, np.asarray(classes)


def refine_a_vs_s(label: str, hand_landmarks) -> str:
    if label not in {"A", "S"}:
        return label

    coords = normalize_landmarks(hand_landmarks)
    thumb_radius = float(np.linalg.norm(coords[4, :2]))
    return "A" if thumb_radius >= AS_THUMB_RADIUS_THRESHOLD else "S"
