# Sign Language Demo

A compact browser demo that combines three related capabilities:

- Text → Sign clip playback (local dataset with letter fallback)
- Live Sign → Text prediction (webcam frames → MediaPipe → local classifier)
- Text → Speech (browser-native speech synthesis)

This repository runs a small Python web server that serves a single-page frontend and exposes a tiny API used by the page.

**Quick links**
- Backend server: [app.py](app.py)
- Dataset helpers: [text_to_sign.py](text_to_sign.py)
- Model utilities: [pred_core.py](pred_core.py)
- Landmark features: [sign_utils.py](sign_utils.py)
- Frontend: [web/index.html](web/index.html) and [web/app.js](web/app.js)

**Supported Python:** 3.10+

**Dependencies** — listed in `requirements.txt` and installable with:

```bash
pip install -r requirements.txt
```

## Features

- Text to sign: sends typed text to the backend which looks up matching local sign clips and returns a timed sequence; missing signs fall back to generated letter cards.
- Sign to text: the frontend captures webcam frames and POSTs them to `/api/predict-sign`; the backend uses MediaPipe landmarks, a feature extractor, and a local classifier (if present) to return predictions.
- Text to speech: uses the browser `speechSynthesis` API to speak any text in the UI.

## Sign Language Alphabet Reference

The demo currently supports fingerspelling based on the American Sign Language (ASL) manual alphabet shown below. This is used as the fallback when a word isn't found in the video dataset (see **Text to sign** in Features).

![ASL fingerspelling alphabet chart, A to Z](web/assets/docs/asl-alphabet-chart.png)



## Project layout

- [app.py](app.py) — main web server (serves `web/`, API endpoints, and dataset video files)
- [pred_core.py](pred_core.py) — model loader and small post-processing helpers
- [sign_utils.py](sign_utils.py) — landmark normalization and feature vector code
- [text_to_sign.py](text_to_sign.py) — dataset discovery, indexing, and video quality scoring
- [requirements.txt](requirements.txt) — Python dependencies
- [web/](web/) — frontend assets (`index.html`, `app.js`, `styles.css`)

## Dataset and model

- Default dataset folder: `assets/sign_videos` inside the project (or legacy `videos` / `vedios`). The code will also use a folder set via the `SIGN_VIDEO_DIR` environment variable.
- The dataset index file expected is `WLASL_v0.3.json` inside the dataset folder; videos should live under `<dataset>/videos/`.
- The live sign-to-text classifier expects a serialized model artifact named `sign_model.pkl` in the project root. The loader in `pred_core.py` expects a dict with keys including `model`, `version`, `feature_names`, and `accuracy` (the same structure produced by the project's training pipeline).

If either dataset or model is missing, the app still runs:

- Missing dataset → text-to-sign falls back to generated letter cards.
- Missing model → sign-to-text is disabled and the frontend shows a helpful notice.

## Environment variables

- `SIGN_VIDEO_DIR`: path to local dataset root (overrides default `assets/sign_videos`).
- `PORT`: server port (default `8000`).
- `OPEN_BROWSER`: set to `0`, `false`, or `no` to disable auto-opening the browser (default is enabled).

## Running locally

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. (Optional) Prepare dataset and/or model:

- Put your dataset at `assets/sign_videos` or set `SIGN_VIDEO_DIR` to a folder containing `WLASL_v0.3.json` and a `videos` subfolder.
- Place a compatible `sign_model.pkl` in the project root to enable live sign-to-text.

3. Start the server:

```bash
python app.py
```

By default the server binds to `127.0.0.1:8000` and attempts to open the demo page in your default browser.

## API (used by the frontend)

- `GET /api/project-stats` — returns JSON with availability and stats (`sign_count`, `model_accuracy`, `feature_count`, etc.).
- `POST /api/translate` — request body `{ "text": "..." }`; returns a sequence of clips with `label`, `type` (`video|generated`), and `url` for video clips.
- `POST /api/predict-sign` — request body `{ "image": "data:image/jpeg;base64,..." }`; returns `{ hand_detected, label, confidence, raw_label, top_predictions, ... }` when a model is present.
- Static video files are served under `/dataset/videos/<relative-path>` by the server when the dataset is available.

## Frontend usage

- Open the page served by `app.py` (or visit `http://127.0.0.1:8000`).
- Use the **Text to sign** panel to type a phrase and play the generated sequence.
- Use **Sign to text** to start the camera and let the app auto-add letters when a sign prediction is stable.
- Use the audio studio to play any text using your browser voices.

## Development notes

- The feature extraction lives in `sign_utils.feature_vector()` and must match the feature layout used when training `sign_model.pkl` (see `FEATURE_VERSION`).
- Model loading and compatibility checks are in `pred_core.load_model()`; mismatched `FEATURE_VERSION` or missing keys will raise a clear error reported to the UI.

## Troubleshooting

- If the page shows "Model unavailable": add `sign_model.pkl` to the project root and restart the server.
- If translations show only generated letters: add a local dataset folder with `WLASL_v0.3.json` and corresponding `videos/` or set `SIGN_VIDEO_DIR` to a valid dataset path.
- MediaPipe/OpenCV installation can be platform-sensitive; ensure `mediapipe` and `opencv-python` are installed for your OS and Python version.

## License & credits

This demo stitches together several public building blocks (MediaPipe landmarks, WLASL/MS-ASL-style clips, a lightweight classifier). Check each dataset and model license before re-distributing large video/model artifacts.

---

