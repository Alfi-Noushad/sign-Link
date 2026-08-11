from __future__ import annotations

import base64
import json
import os
import threading
import webbrowser
from dataclasses import dataclass
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote, urlparse

import cv2
import joblib
import mediapipe as mp
import numpy as np

from pred_core import MIN_CONFIDENCE, load_model as load_sign_model, refine_a_vs_s
from sign_utils import feature_vector
from text_to_sign import (
    METADATA_PATH,
    VIDEOS_FOLDER,
    dataset_status,
    find_video_sequence,
    load_video_index,
    pick_best_video,
)


BASE_DIR = Path(__file__).resolve().parent
WEB_DIR = BASE_DIR / "web"
MODEL_PATH = BASE_DIR / "sign_model.pkl"
HOST = "127.0.0.1"
PORT = int(os.environ.get("PORT", "8000"))
AUTO_OPEN_BROWSER = os.environ.get("OPEN_BROWSER", "1").lower() not in {"0", "false", "no"}


@dataclass
class AppState:
    video_index: dict[str, tuple[Path, ...]]
    max_phrase_words: int
    stats: dict[str, Any]
    sign_model: Any | None
    sign_classes: Any | None
    hand_detector: Any | None
    hand_detector_lock: threading.Lock


def load_model_stats() -> dict[str, Any]:
    if not MODEL_PATH.exists():
        return {
            "model_accuracy": "Not loaded",
            "feature_count": "Not loaded",
        }

    try:
        artifact = joblib.load(MODEL_PATH)
    except Exception as error:  # pragma: no cover - defensive fallback
        return {
            "model_accuracy": "Not loaded",
            "feature_count": "Not loaded",
            "model_error": str(error),
        }

    if not isinstance(artifact, dict):
        return {
            "model_accuracy": "Not loaded",
            "feature_count": "Not loaded",
        }

    stats: dict[str, Any] = {}
    accuracy = artifact.get("accuracy")
    if isinstance(accuracy, (int, float)):
        stats["model_accuracy"] = round(float(accuracy) * 100, 1)
    else:
        stats["model_accuracy"] = "Not loaded"

    feature_names = artifact.get("feature_names")
    if isinstance(feature_names, (list, tuple)):
        stats["feature_count"] = len(feature_names)
    else:
        stats["feature_count"] = "Not loaded"

    return stats


def load_metadata_entry_count() -> int | None:
    if not METADATA_PATH.exists():
        return None

    try:
        with METADATA_PATH.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
    except Exception:  # pragma: no cover - defensive fallback
        return None

    if isinstance(metadata, list):
        return len(metadata)
    return None


def build_app_state() -> AppState:
    clips_ready, clips_message = dataset_status()
    video_index, max_phrase_words = load_video_index()

    stats: dict[str, Any] = {
        "sign_count": len(video_index),
        "phrase_depth": max_phrase_words,
    }

    if clips_ready:
        metadata_entries = load_metadata_entry_count()
        if metadata_entries is not None:
            stats["metadata_entries"] = metadata_entries
    else:
        stats["text_to_sign_notice"] = clips_message

    stats.update(load_model_stats())

    sign_model = None
    sign_classes = None
    hand_detector = None

    try:
        sign_model, sign_classes = load_sign_model()
        hand_detector = mp.solutions.hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            model_complexity=0,
            min_detection_confidence=0.4,
            min_tracking_confidence=0.4,
        )
    except Exception as error:
        stats["sign_to_text_error"] = str(error)

    return AppState(
        video_index=video_index,
        max_phrase_words=max_phrase_words,
        stats=stats,
        sign_model=sign_model,
        sign_classes=sign_classes,
        hand_detector=hand_detector,
        hand_detector_lock=threading.Lock(),
    )


APP_STATE = build_app_state()


def parse_json_body(raw_body: bytes) -> dict[str, Any]:
    try:
        payload = json.loads(raw_body.decode("utf-8") or "{}")
    except json.JSONDecodeError as error:
        raise ValueError("Send valid JSON.") from error

    if not isinstance(payload, dict):
        raise ValueError("JSON payload must be an object.")

    return payload


def serialize_clip(label: str, video_candidates, clip_type: str) -> dict[str, Any]:
    if clip_type == "generated":
        return {
            "label": label.upper(),
            "type": "generated",
            "duration_ms": 850,
        }

    video_path = pick_best_video(video_candidates)
    if video_path is None:
        return {
            "label": label.upper(),
            "type": "generated",
            "duration_ms": 850,
        }

    relative_path = video_path.relative_to(VIDEOS_FOLDER).as_posix()
    return {
        "label": label.upper(),
        "type": "video",
        "url": f"/dataset/videos/{quote(relative_path)}",
        "filename": video_path.name,
    }


def decode_image_frame(image_data: str) -> np.ndarray:
    encoded = image_data.split(",", 1)[-1]

    try:
        binary = base64.b64decode(encoded)
    except Exception as error:
        raise ValueError("Image data could not be decoded.") from error

    array = np.frombuffer(binary, dtype=np.uint8)
    frame = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Image frame could not be read.")

    return frame


def generate_detection_frames(frame: np.ndarray) -> list[np.ndarray]:
    bordered = cv2.copyMakeBorder(
        frame,
        120,
        120,
        120,
        120,
        cv2.BORDER_REPLICATE,
    )
    flipped = cv2.flip(bordered, 1)
    return [bordered, flipped]


def predict_hand_landmarks(hand_landmarks) -> dict[str, Any]:
    features = feature_vector(hand_landmarks)
    model = APP_STATE.sign_model
    classes = APP_STATE.sign_classes

    if model is None or classes is None:
        raise RuntimeError("Sign-to-text is unavailable because the model is not ready.")

    if hasattr(model, "predict_proba"):
        probabilities = np.asarray(model.predict_proba([features])[0], dtype=np.float32)
    else:
        probabilities = np.ones(len(classes), dtype=np.float32)

    best_index = int(np.argmax(probabilities))
    best_label = str(classes[best_index]) if len(classes) else str(model.predict([features])[0])
    best_label = refine_a_vs_s(best_label, hand_landmarks)
    best_confidence = float(probabilities[best_index])

    top_indices = np.argsort(probabilities)[::-1][:3]
    top_predictions: list[dict[str, Any]] = []
    for rank, index in enumerate(top_indices):
        label = str(classes[int(index)])
        if rank == 0:
            label = best_label
        top_predictions.append(
            {
                "label": label,
                "confidence": round(float(probabilities[int(index)]), 4),
            }
        )

    committed_label = best_label if best_confidence >= MIN_CONFIDENCE else None
    committed_confidence = best_confidence if committed_label is not None else 0.0

    return {
        "label": committed_label,
        "confidence": round(float(committed_confidence), 4),
        "raw_label": best_label,
        "raw_confidence": round(float(best_confidence), 4),
        "top_predictions": top_predictions,
    }


def predict_sign_from_image(image_data: str) -> dict[str, Any]:
    if APP_STATE.sign_model is None or APP_STATE.sign_classes is None or APP_STATE.hand_detector is None:
        raise RuntimeError("Sign-to-text is unavailable because the model is not ready.")

    frame = decode_image_frame(image_data)
    result = None
    for candidate_frame in generate_detection_frames(frame):
        rgb_frame = cv2.cvtColor(candidate_frame, cv2.COLOR_BGR2RGB)
        with APP_STATE.hand_detector_lock:
            result = APP_STATE.hand_detector.process(rgb_frame)
        if result.multi_hand_landmarks:
            break

    if result is None or not result.multi_hand_landmarks:
        return {
            "label": None,
            "confidence": 0.0,
            "hand_detected": False,
        }

    hand_landmarks = result.multi_hand_landmarks[0]
    prediction = predict_hand_landmarks(hand_landmarks)

    return {
        "hand_detected": True,
        **prediction,
    }


class GestureSiteHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_DIR), **kwargs)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)

        if parsed.path == "/api/project-stats":
            self.respond_json({"ok": True, "stats": APP_STATE.stats})
            return

        if parsed.path == "/":
            self.path = "/index.html"

        super().do_GET()

    def do_POST(self) -> None:
        parsed = urlparse(self.path)

        if parsed.path not in {"/api/translate", "/api/predict-sign"}:
            self.send_error(HTTPStatus.NOT_FOUND, "Unknown API route.")
            return

        content_length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(content_length)

        try:
            payload = parse_json_body(raw_body)
        except ValueError as error:
            self.respond_json(
                {"ok": False, "error": str(error)},
                status=HTTPStatus.BAD_REQUEST,
            )
            return

        if parsed.path == "/api/predict-sign":
            image_data = str(payload.get("image", "")).strip()
            if not image_data:
                self.respond_json(
                    {"ok": False, "error": "Send an image frame first."},
                    status=HTTPStatus.BAD_REQUEST,
                )
                return

            try:
                prediction = predict_sign_from_image(image_data)
            except ValueError as error:
                self.respond_json(
                    {"ok": False, "error": str(error)},
                    status=HTTPStatus.BAD_REQUEST,
                )
                return
            except RuntimeError as error:
                self.respond_json(
                    {"ok": False, "error": str(error)},
                    status=HTTPStatus.SERVICE_UNAVAILABLE,
                )
                return

            self.respond_json({"ok": True, **prediction})
            return

        text = str(payload.get("text", "")).strip()
        if not text:
            self.respond_json(
                {"ok": False, "error": "Enter some text first."},
                status=HTTPStatus.BAD_REQUEST,
            )
            return

        clips, finger_spelled_words = find_video_sequence(
            text,
            APP_STATE.video_index,
            APP_STATE.max_phrase_words,
        )

        serialized_clips = [
            serialize_clip(label, video_candidates, clip_type)
            for label, video_candidates, clip_type in clips
        ]

        if not serialized_clips:
            self.respond_json(
                {
                    "ok": False,
                    "error": "No matching signs were found for that text.",
                },
                status=HTTPStatus.NOT_FOUND,
            )
            return

        self.respond_json(
            {
                "ok": True,
                "text": text,
                "clip_count": len(serialized_clips),
                "clips": serialized_clips,
                "finger_spelled_words": sorted(set(finger_spelled_words)),
            }
        )

    def translate_path(self, path: str) -> str:
        parsed_path = urlparse(path).path
        clean_path = unquote(parsed_path)

        if clean_path.startswith("/dataset/videos/"):
            relative = clean_path.removeprefix("/dataset/videos/").strip("/")
            candidate = (VIDEOS_FOLDER / relative).resolve()
            if candidate.is_file() and candidate.is_relative_to(VIDEOS_FOLDER.resolve()):
                return str(candidate)

        return super().translate_path(path)

    def log_message(self, format: str, *args) -> None:
        print(f"[web] {self.address_string()} - {format % args}")

    def respond_json(
        self,
        payload: dict[str, Any],
        status: HTTPStatus = HTTPStatus.OK,
    ) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


def open_browser(url: str) -> None:
    try:
        webbrowser.open(url, new=2)
    except Exception:
        pass


def main() -> None:
    server = ThreadingHTTPServer((HOST, PORT), GestureSiteHandler)
    url = f"http://{HOST}:{PORT}"
    print(f"Serving the project website at {url}")
    print("Press Ctrl+C to stop the server.")

    if AUTO_OPEN_BROWSER:
        threading.Timer(0.8, open_browser, args=(url,)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
    finally:
        server.server_close()
        if APP_STATE.hand_detector is not None:
            APP_STATE.hand_detector.close()


if __name__ == "__main__":
    main()
