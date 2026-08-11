from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from pathlib import Path

import cv2


BASE_DIR = Path(__file__).resolve().parent
ENV_DATASET_KEY = "SIGN_VIDEO_DIR"
DEFAULT_DATASET_DIR = BASE_DIR / "assets" / "sign_videos"
LEGACY_DATASET_DIRS = (
    BASE_DIR / "videos",
    BASE_DIR / "vedios",
)


def resolve_dataset_folder() -> Path:
    env_value = os.environ.get(ENV_DATASET_KEY)
    if env_value:
        return Path(env_value).expanduser().resolve()

    for candidate in (DEFAULT_DATASET_DIR, *LEGACY_DATASET_DIRS):
        if candidate.exists():
            return candidate

    return DEFAULT_DATASET_DIR


DATASET_FOLDER = resolve_dataset_folder()
METADATA_PATH = DATASET_FOLDER / "WLASL_v0.3.json"
VIDEOS_FOLDER = DATASET_FOLDER / "videos"


def dataset_status() -> tuple[bool, str]:
    if not DATASET_FOLDER.exists():
        return False, "Local sign clips were not found."

    if not METADATA_PATH.exists():
        return False, "Sign clip metadata is missing."

    if not VIDEOS_FOLDER.exists():
        return False, "The sign clip video folder is missing."

    return True, f"Loaded sign clips from {DATASET_FOLDER.name}."


def clean_text(text: str) -> list[str]:
    cleaned = re.sub(r"[^a-zA-Z\s]", " ", text.lower())
    return [word for word in cleaned.split() if word]


def load_video_index() -> tuple[dict[str, tuple[Path, ...]], int]:
    ready, _ = dataset_status()
    if not ready:
        return {}, 1

    try:
        with METADATA_PATH.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except (OSError, json.JSONDecodeError):
        return {}, 1

    video_index: dict[str, tuple[Path, ...]] = {}
    max_phrase_words = 1

    for item in data:
        gloss = str(item.get("gloss", "")).strip().lower()
        if not gloss:
            continue

        candidate_paths = []
        for instance in item.get("instances", []):
            video_id = instance.get("video_id")
            if not video_id:
                continue

            video_path = VIDEOS_FOLDER / f"{video_id}.mp4"
            if video_path.exists():
                candidate_paths.append(video_path)

        if candidate_paths and gloss not in video_index:
            video_index[gloss] = tuple(candidate_paths)
            max_phrase_words = max(max_phrase_words, len(gloss.split()))

    return video_index, max_phrase_words


def find_video_sequence(
    text: str,
    video_index: dict[str, tuple[Path, ...]],
    max_phrase_words: int,
) -> tuple[list[tuple[str, tuple[Path, ...], str]], list[str]]:
    words = clean_text(text)
    clips: list[tuple[str, tuple[Path, ...], str]] = []
    finger_spelled_words: list[str] = []
    index = 0

    while index < len(words):
        matched = False
        longest_phrase = min(max_phrase_words, len(words) - index)

        for size in range(longest_phrase, 0, -1):
            phrase = " ".join(words[index:index + size])
            if phrase in video_index:
                clips.append((phrase, video_index[phrase], "video"))
                index += size
                matched = True
                break

        if matched:
            continue

        word = words[index]
        letter_clips = []

        for letter in word:
            if letter in video_index:
                letter_clips.append((letter, video_index[letter], "video"))
            else:
                letter_clips.append((letter, (), "generated"))

        if letter_clips:
            clips.extend(letter_clips)
            finger_spelled_words.append(word)

        index += 1

    return clips, finger_spelled_words


@lru_cache(maxsize=4096)
def score_video_quality(video_path_str: str) -> float:
    capture = cv2.VideoCapture(video_path_str)
    if not capture.isOpened():
        return float("-inf")

    width = capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0
    height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0
    frame_total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    sample_positions = [0]
    if frame_total > 1:
        sample_positions.extend(
            [
                max(frame_total // 2, 0),
                max(frame_total - 2, 0),
            ]
        )

    brightness_values = []
    sharpness_values = []
    white_ratios = []

    for position in sample_positions:
        capture.set(cv2.CAP_PROP_POS_FRAMES, position)
        success, frame = capture.read()
        if not success or frame is None:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness_values.append(float(gray.mean()))
        sharpness_values.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
        white_ratios.append(float((gray > 238).mean()))

    capture.release()

    resolution_score = (width * height) / 1000.0
    brightness_score = (
        sum(brightness_values) / len(brightness_values) if brightness_values else 128.0
    )
    sharpness_score = (
        sum(sharpness_values) / len(sharpness_values) if sharpness_values else 0.0
    )
    white_ratio = sum(white_ratios) / len(white_ratios) if white_ratios else 0.0

    return (
        resolution_score
        + (sharpness_score * 0.12)
        - (abs(brightness_score - 122.0) * 0.8)
        - (white_ratio * 180.0)
    )


def pick_best_video(video_candidates) -> Path | None:
    existing_candidates = [path for path in video_candidates if path.exists()]
    if not existing_candidates:
        return None

    return max(existing_candidates, key=lambda path: score_video_quality(str(path)))
