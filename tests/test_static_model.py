"""The committed letter model loads and recognises held-out test samples (from the MIT asl-now set)."""
import json
from pathlib import Path

import numpy as np
import pytest

from signspeak.core.augment import augment_hand, augment_static
from signspeak.core.features import STATIC_DIM
from signspeak.core.models import STATIC_MODEL, StaticClassifier

FIXTURE = Path(__file__).parent / "fixtures" / "letter_features.json"


@pytest.fixture(scope="module")
def clf():
    if not STATIC_MODEL.exists():
        pytest.skip("letter model not trained")
    return StaticClassifier()


def test_all_letters_known(clf):
    assert clf.classes[:26] == [chr(c) for c in range(65, 91)]


def test_fixture_accuracy(clf):
    data = json.loads(FIXTURE.read_text())
    total = correct = 0
    for label, samples in data.items():
        for s in samples:
            pred = clf.predict(np.array(s, dtype=np.float32))
            assert 0 <= pred.confidence <= 1 and len(pred.top) == 3
            correct += pred.label == label
            total += 1
    assert total >= 60
    assert correct / total >= 0.85


def test_augmentation_keeps_normalisation():
    rng = np.random.default_rng(0)
    x = np.array(next(iter(json.loads(FIXTURE.read_text()).values()))[0], dtype=np.float32)
    a = augment_hand(x, rng).reshape(21, 3)
    assert np.allclose(a[0], 0)
    assert np.linalg.norm(a[9, :2]) == pytest.approx(1.0, abs=1e-4)
    X, y = augment_static(x[None].repeat(5, 0), np.array(["A"] * 5), copies=2)
    assert X.shape == (15, STATIC_DIM) and len(y) == 15
