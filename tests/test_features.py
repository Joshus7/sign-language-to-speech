"""Normalisation must make letter features independent of position, size and hand side."""
import numpy as np
import pytest

from signspeak.core import landmarks as L
from signspeak.core.features import (FRAME_DIM, HAND_DYN_DIM, STATIC_DIM, active_span, frame_features,
                                     hands_present, normalise_hand, resample, to_aspect)
from signspeak.core.hands import Hand, HandsResult


def fake_hand(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pts = rng.uniform(0.3, 0.6, size=(21, 3)).astype(np.float32)
    pts[:, 2] *= 0.1
    pts[L.MIDDLE_MCP, :2] = pts[L.WRIST, :2] + [0.0, -0.1]  # non-zero hand size
    return pts


def test_shape_and_wrist_at_origin():
    f = normalise_hand(fake_hand())
    assert f.shape == (STATIC_DIM,)
    assert np.allclose(f[:3], 0)


def test_hand_size_is_one():
    f = normalise_hand(fake_hand()).reshape(21, 3)
    assert np.linalg.norm(f[L.MIDDLE_MCP, :2]) == pytest.approx(1.0, abs=1e-5)


def test_translation_and_scale_invariant():
    pts = fake_hand(1)
    moved = pts * 2.5 + np.array([0.2, -0.1, 0.05], dtype=np.float32)
    assert np.allclose(normalise_hand(pts), normalise_hand(moved), atol=1e-5)


def test_left_hand_is_mirrored():
    pts = fake_hand(2)
    mirrored = pts.copy()
    mirrored[:, 0] = 1.0 - mirrored[:, 0]
    assert np.allclose(normalise_hand(pts), normalise_hand(mirrored, is_left=True), atol=1e-5)


def test_aspect_correction():
    pts = fake_hand(3)
    wide = to_aspect(pts, 1280, 720)
    assert np.allclose(wide[:, 0], pts[:, 0] * 1280 / 720)
    assert np.allclose(wide[:, 1], pts[:, 1])


def test_bad_shape_raises():
    with pytest.raises(ValueError):
        normalise_hand(np.zeros((20, 3)))


def test_frame_features_slots():
    small, big = fake_hand(4), fake_hand(5)
    big = (big - big[0]) * 2 + big[0]  # bigger hand = dominant
    res = HandsResult([Hand(small, "Left", 0.9), Hand(big, "Right", 0.9)], 640, 480)
    f = frame_features(res)
    assert f.shape == (FRAME_DIM,)
    assert f[HAND_DYN_DIM - 1] == 1.0 and f[-1] == 1.0       # both present
    assert f[HAND_DYN_DIM - 2] > f[-2]                        # dominant slot holds the bigger hand
    assert np.allclose(frame_features(HandsResult([], 640, 480)), 0)


def test_resample_and_span():
    seq = np.zeros((50, FRAME_DIM), np.float32)
    seq[10:40, HAND_DYN_DIM - 1] = 1
    seq[:, 0] = np.arange(50)
    assert resample(seq).shape == (30, FRAME_DIM)
    assert resample(seq[:1]).shape == (30, FRAME_DIM)
    r = resample(seq[10:40], 30)
    assert np.all(np.diff(r[:, 0]) >= 0)                     # order kept
    assert set(r[:, 0]) <= set(seq[10:40, 0])                # nearest frames, no blending
    assert active_span(seq, pad=0) == (10, 40)
    assert hands_present(seq).sum() == 30
