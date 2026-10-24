"""Landmark-space data augmentation (the datasets are small, especially WLASL)."""
from __future__ import annotations

import numpy as np

from . import landmarks as L
from .features import HAND_DYN_DIM, STATIC_DIM, normalise_hand


def augment_hand(feat: np.ndarray, rng: np.random.Generator, mirror_p: float = 0.5,
                 max_rot_deg: float = 15.0, aspect: float = 0.12, noise: float = 0.02) -> np.ndarray:
    """Random in-plane rotation, aspect-ratio change, mirroring and jitter of one
    normalised hand (63,), re-normalised afterwards so the output is a valid feature."""
    pts = feat.reshape(L.NUM_LANDMARKS, 3).copy()
    a = np.deg2rad(rng.uniform(-max_rot_deg, max_rot_deg))
    c, s = np.cos(a), np.sin(a)
    pts[:, :2] = pts[:, :2] @ np.array([[c, -s], [s, c]], dtype=np.float32).T
    pts[:, 0] *= rng.uniform(1 - aspect, 1 + aspect)
    pts += rng.normal(0, noise, pts.shape).astype(np.float32)
    pts[L.WRIST] = 0
    return normalise_hand(pts, is_left=bool(rng.random() < mirror_p))


def augment_static(X: np.ndarray, y: np.ndarray, copies: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    Xa = [X] + [np.stack([augment_hand(x, rng) for x in X]) for _ in range(copies)]
    return np.concatenate(Xa), np.concatenate([y] * (copies + 1))


def augment_sequence(seq: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One random variant of a (T, FRAME_DIM) word sequence: the same rotation/aspect/mirror
    for every frame (a signer doesn't change camera mid-sign), a small shift of the hand
    positions and per-frame jitter."""
    out = seq.copy()
    mirror = rng.random() < 0.5
    rot = rng.uniform(-12, 12)
    asp = rng.uniform(0.9, 1.1)
    shift = rng.normal(0, 0.04, 2).astype(np.float32)
    scale = rng.uniform(0.85, 1.15)
    frame_rng_seed = int(rng.integers(1 << 31))
    for slot in range(2):
        base = slot * HAND_DYN_DIM
        present = out[:, base + HAND_DYN_DIM - 1] > 0.5
        for t in np.flatnonzero(present):
            hand_rng = np.random.default_rng(frame_rng_seed + slot * 10_000 + t)  # fresh jitter per frame
            feat = out[t, base:base + STATIC_DIM]
            pts = feat.reshape(L.NUM_LANDMARKS, 3).copy()
            a = np.deg2rad(rot)
            c, s = np.cos(a), np.sin(a)
            pts[:, :2] = pts[:, :2] @ np.array([[c, -s], [s, c]], dtype=np.float32).T
            pts[:, 0] *= asp
            pts += hand_rng.normal(0, 0.015, pts.shape).astype(np.float32)
            pts[L.WRIST] = 0
            out[t, base:base + STATIC_DIM] = normalise_hand(pts, is_left=mirror)
            pos = out[t, base + STATIC_DIM:base + STATIC_DIM + 2]
            if mirror:
                pos[0] = -pos[0]
            out[t, base + STATIC_DIM:base + STATIC_DIM + 2] = pos * scale + shift
            out[t, base + STATIC_DIM + 2] *= scale
    return out


def time_warp(seq: np.ndarray, rng: np.random.Generator, min_keep: float = 0.8) -> np.ndarray:
    """Random crop of 80-100 % of the sequence (start/end of a live window are imprecise)."""
    n = len(seq)
    keep = max(2, int(round(n * rng.uniform(min_keep, 1.0))))
    start = int(rng.integers(0, n - keep + 1))
    return seq[start:start + keep]
