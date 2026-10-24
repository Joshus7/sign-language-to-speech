"""Turn the downloaded datasets into normalised landmark features in data/features/.

* Letter landmarks (asl_now JSON) are normalised directly.
* Letter photos (asl_letters, optional Kaggle) go through HandLandmarker (IMAGE mode);
  photos where no hand is found are skipped and counted.
* WLASL word clips go through HandLandmarker (VIDEO mode) frame by frame.

Outputs (git-ignored):
    data/features/letters.npz   X (N, 63), y, source, id
    data/features/words/<gloss>/<clip>.npy   (T, 134) per-frame features
    data/features/words_index.json

Usage:
    python scripts/extract_landmarks.py [--only letters|words] [--force]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from signspeak.core.features import (FRAME_DIM, dominant_hand, frame_features, hands_present,  # noqa: E402
                                     normalise_hand, static_features, to_aspect)
from signspeak.core.hands import HandTracker  # noqa: E402
from signspeak.core.video_source import DATA_RAW, IMAGE_EXT, VideoSource  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ROOT / "data" / "features"
LETTERS = [chr(c) for c in range(ord("A"), ord("Z") + 1)]
# Kaggle grassknoted class folders that are not letters -> sentence-builder actions.
KAGGLE_EXTRA = {"space": "SPACE", "del": "DEL", "nothing": "NOTHING"}
# asl-now landmarks come from a browser webcam; 4:3 (640x480) is the browser default.
ASL_NOW_ASPECT = (4, 3)


def letter_landmark_json() -> tuple[list, list, list]:
    root = DATA_RAW / "asl_now"
    X, y, ids = [], [], []
    bad = 0
    for d in sorted(p for p in root.iterdir() if p.is_dir() and p.name in LETTERS) if root.exists() else []:
        for f in sorted(d.glob("*.json")):
            try:
                pts = json.loads(f.read_text())
                if isinstance(pts, dict):
                    pts = pts.get("landmarks", pts)
                if pts and isinstance(pts[0], list):  # several hands -> first one
                    pts = pts[0]
                arr = np.array([[p["x"], p["y"], p.get("z", 0.0)] for p in pts], dtype=np.float32)
                X.append(normalise_hand(to_aspect(arr, *ASL_NOW_ASPECT)))
                y.append(d.name)
                ids.append(str(f.relative_to(DATA_RAW)))
            except (ValueError, KeyError, TypeError, IndexError):
                bad += 1
    print(f"   asl_now landmarks: {len(X)} samples ({bad} unreadable)")
    return X, y, ids


def letter_images(tracker: HandTracker) -> tuple[list, list, list, list]:
    X, y, ids, src = [], [], [], []
    sources = [("asl_letters", DATA_RAW / "asl_letters" / "train_val", None),
               ("kaggle", DATA_RAW / "kaggle_asl_alphabet", KAGGLE_EXTRA)]
    for name, root, extra in sources:
        if not root.is_dir():
            continue
        n_img, skipped, t0 = 0, 0, time.perf_counter()
        for d in sorted(p for p in root.iterdir() if p.is_dir()):
            label = d.name if d.name in LETTERS else (extra or {}).get(d.name)
            if label is None:
                continue
            for f in sorted(d.iterdir()):
                if f.suffix.lower() not in IMAGE_EXT:
                    continue
                img = cv2.imread(str(f))
                n_img += 1
                if img is None:
                    skipped += 1
                    continue
                res = tracker.process(img)
                hand = dominant_hand(res)
                if hand is None:
                    if label == "NOTHING":  # 'nothing' is supposed to have no hand
                        continue
                    skipped += 1
                    continue
                X.append(static_features(hand, res.width, res.height))
                y.append(label)
                ids.append(str(f.relative_to(DATA_RAW)))
                src.append(name)
            print(f"\r   {name}: {n_img} images, {skipped} skipped (no hand)   ", end="", flush=True)
        print(f"\r   {name}: {n_img} images, {skipped} skipped (no hand), "
              f"{time.perf_counter() - t0:.0f} s")
    return X, y, ids, src


def extract_letters() -> None:
    print("== letters")
    X, y, ids = letter_landmark_json()
    src = ["asl_now"] * len(X)
    with HandTracker(video=False, num_hands=1, min_detection_confidence=0.3) as tracker:
        X2, y2, ids2, src2 = letter_images(tracker)
    X, y, ids, src = X + X2, y + y2, ids + ids2, src + src2
    if not X:
        print("!! no letter data - run scripts/download_data.py first")
        return
    FEATURES.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(FEATURES / "letters.npz", X=np.stack(X), y=np.array(y), id=np.array(ids),
                        source=np.array(src))
    counts = {c: y.count(c) for c in sorted(set(y))}
    print(f"   saved {len(X)} letter samples -> data/features/letters.npz")
    print(f"   per class: min {min(counts.values())}, max {max(counts.values())}\n")


def extract_words(force: bool) -> None:
    print("== words")
    index_file = DATA_RAW / "wlasl" / "index.json"
    if not index_file.exists():
        print("!! no WLASL clips - run scripts/download_data.py first")
        return
    vocab = yaml.safe_load((ROOT / "config" / "vocabulary.yaml").read_text())["words"]
    glosses = [w["gloss"] for w in vocab]
    index = json.loads(index_file.read_text())
    out_index: dict[str, list[dict]] = {}
    total = sum(len(index.get(g, [])) for g in glosses)
    done, no_hands, t0 = 0, 0, time.perf_counter()
    for gloss in glosses:
        out_index[gloss] = []
        for rel in index.get(gloss, []):
            done += 1
            dest = FEATURES / "words" / gloss / (Path(rel).stem + ".npy")
            key = str(dest.relative_to(FEATURES))
            entry = {"feature": key, "clip": f"wlasl/{rel}"}
            if dest.exists() and not force:
                out_index[gloss].append(entry)
                continue
            seq = []
            with VideoSource(DATA_RAW / "wlasl" / rel) as vs, HandTracker() as tracker:
                for frame in vs:
                    seq.append(frame_features(tracker.process(frame.image, frame.timestamp_ms)))
            seq = np.stack(seq) if seq else np.zeros((0, FRAME_DIM), np.float32)
            if not seq.size or not hands_present(seq).any():
                no_hands += 1
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            np.save(dest, seq)
            out_index[gloss].append(entry)
            print(f"\r   {done}/{total} clips ({time.perf_counter() - t0:.0f} s)   ", end="", flush=True)
    (FEATURES / "words_index.json").write_text(json.dumps({"glosses": glosses, "clips": out_index}, indent=1))
    n = sum(map(len, out_index.values()))
    print(f"\r   saved {n} sequences ({no_hands} clips without hands skipped) -> data/features/words/\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["letters", "words"])
    ap.add_argument("--force", action="store_true", help="re-extract word clips that already exist")
    args = ap.parse_args()
    if args.only in (None, "letters"):
        extract_letters()
    if args.only in (None, "words"):
        extract_words(args.force)


if __name__ == "__main__":
    main()
