"""Landmark normalisation shared by training, tests and the app.

Static (letter) features - one hand, 63 values:
    wrist at the origin, scaled so the wrist -> middle-finger-MCP distance is 1, and
    mirrored (x -> -x) for left hands so a letter signed with either hand looks the same.

Dynamic (word) features - one frame, 2 x 67 values:
    for the dominant and the other hand: the 63 static features, the wrist position in
    the frame (x, y centred on 0), the hand size relative to the frame height and a
    presence flag. A missing hand is all zeros. Sequences are resampled to SEQ_LEN frames.
"""
from __future__ import annotations

import numpy as np

from . import landmarks as L
from .hands import Hand, HandsResult

STATIC_DIM = L.NUM_LANDMARKS * 3
HAND_DYN_DIM = STATIC_DIM + 4
FRAME_DIM = 2 * HAND_DYN_DIM
SEQ_LEN = 30


def to_aspect(lms: np.ndarray, width: int, height: int) -> np.ndarray:
    """Normalised image coords -> units of frame height, so x and y share a scale."""
    out = np.asarray(lms, dtype=np.float32).copy()
    out[:, 0] *= width / height
    out[:, 2] *= width / height  # MediaPipe z is scaled like x
    return out


def hand_size(lms: np.ndarray) -> float:
    return float(np.linalg.norm(lms[L.MIDDLE_MCP, :2] - lms[L.WRIST, :2]))


def normalise_hand(lms: np.ndarray, is_left: bool = False) -> np.ndarray:
    """(21, 3) landmarks (any consistent units) -> (63,) translation/scale-invariant features."""
    lms = np.asarray(lms, dtype=np.float32)
    if lms.shape != (L.NUM_LANDMARKS, 3):
        raise ValueError(f"expected (21, 3) landmarks, got {lms.shape}")
    centred = lms - lms[L.WRIST]
    size = hand_size(lms)
    if size < 1e-6:
        size = float(np.abs(centred[:, :2]).max()) or 1.0
    feat = centred / size
    if is_left:
        feat[:, 0] *= -1.0
    return feat.reshape(-1)


def static_features(hand: Hand, width: int = 1, height: int = 1) -> np.ndarray:
    return normalise_hand(to_aspect(hand.landmarks, width, height), hand.is_left)


def dominant_hand(result: HandsResult) -> Hand | None:
    """The larger (closest) hand, which is almost always the signing hand."""
    if not result.hands:
        return None
    return max(result.hands, key=lambda h: hand_size(to_aspect(h.landmarks, result.width, result.height)))


def frame_features(result: HandsResult) -> np.ndarray:
    """(FRAME_DIM,) features for one video frame; the dominant hand goes in the first slot."""
    out = np.zeros(FRAME_DIM, dtype=np.float32)
    dom = dominant_hand(result)
    if dom is None:
        return out
    others = [h for h in result.hands if h is not dom]
    for slot, hand in enumerate([dom] + others[:1]):
        lms = to_aspect(hand.landmarks, result.width, result.height)
        base = slot * HAND_DYN_DIM
        out[base:base + STATIC_DIM] = normalise_hand(lms, hand.is_left)
        wrist = hand.landmarks[L.WRIST]
        out[base + STATIC_DIM:base + STATIC_DIM + 4] = (wrist[0] - 0.5, wrist[1] - 0.5, hand_size(lms), 1.0)
    return out


def hands_present(seq: np.ndarray) -> np.ndarray:
    """Boolean mask of frames in a (T, FRAME_DIM) sequence with at least one hand."""
    return seq[:, HAND_DYN_DIM - 1] > 0.5


def resample(seq: np.ndarray, length: int = SEQ_LEN) -> np.ndarray:
    """Resample a (T, D) sequence to (length, D) by nearest frame (interpolating towards an
    all-zero missing-hand frame would invent hand shapes)."""
    seq = np.asarray(seq, dtype=np.float32)
    if len(seq) == 0:
        return np.zeros((length, seq.shape[1] if seq.ndim == 2 else FRAME_DIM), dtype=np.float32)
    idx = np.rint(np.linspace(0, len(seq) - 1, length)).astype(int)
    return seq[idx]


def active_span(seq: np.ndarray, pad: int = 2) -> tuple[int, int]:
    """[start, end) of the frames between the first and last frame with a hand."""
    idx = np.flatnonzero(hands_present(seq))
    if len(idx) == 0:
        return 0, len(seq)
    return max(0, idx[0] - pad), min(len(seq), idx[-1] + 1 + pad)
