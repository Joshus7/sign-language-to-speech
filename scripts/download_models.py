"""Download the MediaPipe HandLandmarker model into models/ (git-ignored).

Usage:
    python scripts/download_models.py
"""
from __future__ import annotations

from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"

MODEL_URLS = {
    "hand_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/latest/hand_landmarker.task"
    ),
}


def main() -> None:
    MODELS.mkdir(exist_ok=True)
    for name, url in MODEL_URLS.items():
        dest = MODELS / name
        if dest.exists() and dest.stat().st_size > 0:
            print(f"{name}: already present, skipped")
            continue
        print(f"{name}: downloading from {url}")
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        dest.write_bytes(r.content)
        print(f"{name}: saved ({len(r.content) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
