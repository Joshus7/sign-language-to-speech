"""Frame sources: webcam, video file, or clips/images from the downloaded datasets."""
from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"

VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
SPLITS_FILE = ROOT / "models" / "trained" / "splits.json"


class SourceError(RuntimeError):
    pass


def _split_ids(kind: str, split: str) -> set[str] | None:
    """Relative paths in a train/val/test split written by the training scripts, if any."""
    if not SPLITS_FILE.exists():
        return None
    data = json.loads(SPLITS_FILE.read_text())
    ids = data.get(kind, {}).get(split)
    return set(ids) if ids is not None else None


def word_clips(split: str = "test") -> list[tuple[str, Path]]:
    """(gloss, path) for downloaded WLASL clips; restricted to ``split`` once a model was trained."""
    index_file = DATA_RAW / "wlasl" / "index.json"
    if not index_file.exists():
        return []
    keep = _split_ids("words", split)
    out = []
    for gloss, files in json.loads(index_file.read_text()).items():
        for rel in files:
            if keep is None or f"wlasl/{rel}" in keep:
                p = DATA_RAW / "wlasl" / rel
                if p.exists():
                    out.append((gloss, p))
    return out


def letter_images(split: str = "test") -> list[tuple[str, Path]]:
    """(letter, path) for downloaded letter photos (test split once a model was trained)."""
    keep = _split_ids("letters", split)
    out = []
    for root in (DATA_RAW / "asl_letters" / "train_val", DATA_RAW / "kaggle_asl_alphabet"):
        if not root.is_dir():
            continue
        for d in sorted(p for p in root.iterdir() if p.is_dir()):
            for p in sorted(d.iterdir()):
                rel = str(p.relative_to(DATA_RAW))
                if p.suffix.lower() in IMAGE_EXT and (keep is None or rel in keep):
                    out.append((d.name, p))
    return out


def pick(items: list, seed: int | None = None):
    if not items:
        raise SourceError("No dataset files found. Run: python scripts/download_data.py")
    return random.Random(seed).choice(items)


@dataclass
class Frame:
    image: np.ndarray
    timestamp_ms: int
    index: int


class VideoSource:
    """Uniform reader. For files, timestamps come from the file (so analysis is
    deterministic); with ``realtime=True`` reading is paced to the file's FPS."""

    def __init__(self, source: int | str | Path, width: int | None = None, height: int | None = None,
                 realtime: bool = False, loop: bool = False):
        self.is_camera = isinstance(source, int)
        self.source = source
        self.realtime = realtime
        self.loop = loop
        if self.is_camera:
            self.cap = cv2.VideoCapture(source)
            if width:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            if height:
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        else:
            path = Path(source)
            if not path.exists():
                raise SourceError(f"Video file not found: {path}")
            self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise SourceError(f"Could not open camera {source}" if self.is_camera
                              else f"Could not open video {source}")
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = fps if fps and 1 <= fps <= 240 else 30.0
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not self.is_camera else 0
        self._index = 0
        self._t0 = time.monotonic()
        self._loop_offset_ms = 0
        self._last_ts = 0

    def read(self) -> Frame | None:
        """Next frame, or None at end of file. Raises SourceError if a camera disconnects."""
        ok, img = self.cap.read()
        if not ok or img is None:
            if self.is_camera:
                raise SourceError("Camera disconnected or stopped sending frames")
            if self.loop and self._index > 0:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self._loop_offset_ms = self._last_ts + int(1000 / self.fps)
                ok, img = self.cap.read()
            if not ok or img is None:
                return None
        if self.is_camera:
            ts = int((time.monotonic() - self._t0) * 1000)
        else:
            ts = self._loop_offset_ms + int(round(self._index_in_file() * 1000 / self.fps))
            if self.realtime:
                delay = ts / 1000 - (time.monotonic() - self._t0)
                if delay > 0:
                    time.sleep(delay)
        self._last_ts = ts
        frame = Frame(img, ts, self._index)
        self._index += 1
        return frame

    def shift_clock(self, seconds: float) -> None:
        """Delay real-time pacing by ``seconds`` (after a pause, so playback doesn't race to catch up)."""
        self._t0 += seconds

    def _index_in_file(self) -> int:
        pos = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
        return int(pos) - 1 if pos else self._index

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()

    def __iter__(self):
        while True:
            f = self.read()
            if f is None:
                return
            yield f


def list_cameras(max_index: int = 4) -> list[int]:
    found = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            found.append(i)
        cap.release()
    return found
