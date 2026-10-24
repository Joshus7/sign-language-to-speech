"""Show the hand-landmark overlay on dataset images, dataset videos or the webcam.

Usage:
    python scripts/preview.py --images            # random letter photos (any key = next, q = quit)
    python scripts/preview.py --words             # WLASL clips of the vocabulary
    python scripts/preview.py --webcam 0
    python scripts/preview.py --images --save out/ --limit 10 --headless   # write annotated frames
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from signspeak.core.hands import HandTracker  # noqa: E402
from signspeak.core.overlay import draw_hands, draw_label  # noqa: E402
from signspeak.core.video_source import VideoSource, letter_images, word_clips  # noqa: E402


def show(win: str, frame, headless: bool, wait: int) -> bool:
    if headless:
        return True
    cv2.imshow(win, frame)
    return (cv2.waitKey(wait) & 0xFF) != ord("q")


def preview_images(args) -> None:
    items = letter_images(split="all")
    random.Random(0).shuffle(items)
    found = 0
    with HandTracker(video=False) as tracker:
        for i, (label, path) in enumerate(items[:args.limit]):
            img = cv2.imread(str(path))
            res = tracker.process(img)
            found += bool(res.hands)
            draw_hands(img, res)
            draw_label(img, f"{label}  hands: {len(res.hands)}")
            if args.save:
                cv2.imwrite(str(Path(args.save) / f"letter_{i:03d}_{label}.jpg"), img)
            if not show("letters", img, args.headless, 0):
                break
    n = min(args.limit, len(items))
    print(f"hands found in {found}/{n} images")


def preview_video(source, label: str, args) -> None:
    t0, frames, with_hands = time.perf_counter(), 0, 0
    with VideoSource(source, realtime=not args.headless) as vs, HandTracker() as tracker:
        for frame in vs:
            img = frame.image
            if vs.is_camera:
                img = cv2.flip(img, 1)
            res = tracker.process(img, frame.timestamp_ms)
            frames += 1
            with_hands += bool(res.hands)
            draw_hands(img, res)
            fps = frames / max(1e-6, time.perf_counter() - t0)
            draw_label(img, f"{label}  hands: {len(res.hands)}  {fps:.0f} fps")
            if args.save and frames % 10 == 1:
                cv2.imwrite(str(Path(args.save) / f"{label.replace(' ', '_')}_{frames:04d}.jpg"), img)
            if not show("signspeak preview", img, args.headless, 1):
                raise KeyboardInterrupt
            if vs.is_camera and args.limit and frames >= args.limit * 30:
                break
    print(f"{label}: {frames} frames, hands in {with_hands} ({frames / max(1e-6, time.perf_counter() - t0):.0f} fps)")


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--images", action="store_true")
    g.add_argument("--words", action="store_true")
    g.add_argument("--webcam", type=int)
    ap.add_argument("--limit", type=int, default=20, help="images / clips to show")
    ap.add_argument("--save", help="folder for annotated frames")
    ap.add_argument("--headless", action="store_true", help="no window (for testing)")
    args = ap.parse_args()
    if args.save:
        Path(args.save).mkdir(parents=True, exist_ok=True)
    try:
        if args.images:
            preview_images(args)
        elif args.words:
            clips = word_clips(split="all")
            random.Random(0).shuffle(clips)
            for gloss, path in clips[:args.limit]:
                preview_video(path, gloss, args)
        else:
            preview_video(args.webcam, "webcam", args)
    except KeyboardInterrupt:
        pass
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
