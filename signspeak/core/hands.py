"""Thin wrapper around the MediaPipe Tasks HandLandmarker (up to 2 hands).

Each detected hand is returned as a :class:`Hand` with landmarks in *normalised image
coordinates* (x, y in [0, 1], z relative to the wrist, scaled like x) - the same format
as the landmark datasets - plus the frame size so callers can convert to pixels.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = ROOT / "models" / "hand_landmarker.task"


class ModelMissingError(FileNotFoundError):
    pass


@dataclass
class Hand:
    landmarks: np.ndarray   # (21, 3) normalised x, y, z
    handedness: str         # "Left" or "Right" as reported by MediaPipe
    score: float

    @property
    def is_left(self) -> bool:
        return self.handedness == "Left"


@dataclass
class HandsResult:
    hands: list[Hand]
    width: int
    height: int
    timestamp_ms: int = 0

    def pixels(self, hand: Hand) -> np.ndarray:
        return hand.landmarks[:, :2] * np.array([self.width, self.height], dtype=np.float32)


class HandTracker:
    """HandLandmarker in VIDEO mode (tracking between frames) or IMAGE mode (single photos).

    In VIDEO mode timestamps passed to :meth:`process` must increase; duplicates are
    fixed up so callers can pass video-file timestamps directly.
    """

    def __init__(self, model_path: str | Path | None = None, num_hands: int = 2, video: bool = True,
                 min_detection_confidence: float = 0.5, min_tracking_confidence: float = 0.5):
        model_path = Path(model_path or DEFAULT_MODEL)
        if not model_path.exists():
            raise ModelMissingError(
                f"Hand model not found at {model_path}. Run: python scripts/download_models.py")
        self.video = video
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path), delegate=BaseOptions.Delegate.CPU),
            running_mode=vision.RunningMode.VIDEO if video else vision.RunningMode.IMAGE,
            num_hands=num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._last_ts = -1

    def process(self, frame_bgr: np.ndarray, timestamp_ms: int = 0) -> HandsResult:
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        if self.video:
            ts = int(timestamp_ms)
            if ts <= self._last_ts:
                ts = self._last_ts + 1
            self._last_ts = ts
            result = self._landmarker.detect_for_video(image, ts)
        else:
            ts = 0
            result = self._landmarker.detect(image)
        hands = []
        for lms, cats in zip(result.hand_landmarks, result.handedness):
            arr = np.array([[p.x, p.y, p.z] for p in lms], dtype=np.float32)
            cat = cats[0]
            hands.append(Hand(arr, cat.category_name, float(cat.score)))
        return HandsResult(hands, w, h, ts)

    def close(self) -> None:
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
