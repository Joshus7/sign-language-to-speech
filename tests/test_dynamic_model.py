"""The committed word LSTM loads and recognises held-out WLASL test sequences (landmarks only)."""
import json
from pathlib import Path

import numpy as np
import pytest

from signspeak.core.augment import augment_sequence, time_warp
from signspeak.core.features import FRAME_DIM, SEQ_LEN
from signspeak.core.models import DYNAMIC_MODEL

FIXTURE = Path(__file__).parent / "fixtures" / "word_sequences.json"


@pytest.fixture(scope="module")
def clf():
    if not DYNAMIC_MODEL.exists():
        pytest.skip("word model not trained")
    from signspeak.core.models import DynamicClassifier
    return DynamicClassifier()


def test_vocabulary(clf):
    assert len(clf.classes) == 20
    assert clf.text("i") == "I" and clf.text("thank you") == "thank you"


def test_test_sequences(clf):
    data = json.loads(FIXTURE.read_text())
    top3 = correct = 0
    for gloss, seq in data.items():
        x = np.array(seq, dtype=np.float32)
        assert x.shape == (SEQ_LEN, FRAME_DIM)
        pred = clf.predict(x)
        correct += pred.label == gloss
        top3 += gloss in [g for g, _ in pred.top]
    # One unseen-signer clip per word: well above chance (5 %), most in the top 3.
    assert correct / len(data) >= 0.6
    assert top3 / len(data) >= 0.85


def test_sequence_augmentation_shapes():
    rng = np.random.default_rng(0)
    seq = np.array(next(iter(json.loads(FIXTURE.read_text()).values())), dtype=np.float32)
    out = augment_sequence(time_warp(seq, rng), rng)
    assert out.shape[1] == FRAME_DIM and 2 <= len(out) <= SEQ_LEN
    assert np.array_equal(out[:, 66] > 0.5, time_warp(seq, np.random.default_rng(0))[:, 66] > 0.5)
