from __future__ import annotations

import itertools

import numpy as np


LANDMARK_COUNT = 21
COORDS_PER_LANDMARK = 3
RAW_FEATURE_COUNT = LANDMARK_COUNT * COORDS_PER_LANDMARK

FINGERTIP_INDICES = np.array([4, 8, 12, 16, 20], dtype=np.int32)
FINGERTIP_PAIR_INDICES = tuple(itertools.combinations(range(len(FINGERTIP_INDICES)), 2))

FEATURE_VERSION = 2


def _to_landmark_array(values) -> np.ndarray:
    if hasattr(values, "landmark"):
        coords = [[lm.x, lm.y, lm.z] for lm in values.landmark]
        array = np.asarray(coords, dtype=np.float32)
    else:
        array = np.asarray(values, dtype=np.float32)

    if array.size != RAW_FEATURE_COUNT:
        raise ValueError(
            f"Expected {RAW_FEATURE_COUNT} landmark values, got {array.size}."
        )

    return array.reshape(LANDMARK_COUNT, COORDS_PER_LANDMARK)


def normalize_landmarks(values) -> np.ndarray:
    coords = _to_landmark_array(values).copy()
    coords -= coords[0]

    scale = float(np.linalg.norm(coords[:, :2], axis=1).max())
    if not np.isfinite(scale) or scale < 1e-6:
        scale = 1.0

    return coords / scale


def feature_vector(values) -> np.ndarray:
    coords = normalize_landmarks(values)
    fingertips = coords[FINGERTIP_INDICES, :2]
    fingertip_radius = np.linalg.norm(fingertips, axis=1)
    fingertip_pair_distance = np.array(
        [
            np.linalg.norm(fingertips[first] - fingertips[second])
            for first, second in FINGERTIP_PAIR_INDICES
        ],
        dtype=np.float32,
    )

    return np.concatenate(
        [coords.reshape(-1), fingertip_radius, fingertip_pair_distance]
    ).astype(np.float32)


def feature_names() -> list[str]:
    names = [
        f"norm_{axis}{index}"
        for index in range(LANDMARK_COUNT)
        for axis in ("x", "y", "z")
    ]
    names.extend([f"tip_radius_{index}" for index in FINGERTIP_INDICES])
    names.extend(
        [
            f"tip_pair_{FINGERTIP_INDICES[first]}_{FINGERTIP_INDICES[second]}"
            for first, second in FINGERTIP_PAIR_INDICES
        ]
    )
    return names
