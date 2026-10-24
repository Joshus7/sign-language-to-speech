"""Loading the trained letter / word models and running them on landmark features."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .features import SEQ_LEN

ROOT = Path(__file__).resolve().parents[2]
TRAINED = ROOT / "models" / "trained"
STATIC_MODEL = TRAINED / "static_letters.joblib"
DYNAMIC_MODEL = TRAINED / "dynamic_words.keras"
DYNAMIC_META = TRAINED / "dynamic_words.json"
SPLITS = TRAINED / "splits.json"


class ModelNotTrainedError(FileNotFoundError):
    pass


@dataclass
class Prediction:
    label: str
    confidence: float
    top: list[tuple[str, float]]


def _prediction(classes: list[str], probs: np.ndarray, k: int = 3) -> Prediction:
    order = np.argsort(probs)[::-1][:k]
    top = [(classes[i], float(probs[i])) for i in order]
    return Prediction(top[0][0], top[0][1], top)


def update_splits(kind: str, split: dict[str, list[str]]) -> None:
    data = json.loads(SPLITS.read_text()) if SPLITS.exists() else {}
    data[kind] = split
    SPLITS.parent.mkdir(parents=True, exist_ok=True)
    SPLITS.write_text(json.dumps(data, indent=0))


class StaticClassifier:
    """Letter classifier (scikit-learn pipeline saved with joblib)."""

    def __init__(self, path: str | Path = STATIC_MODEL):
        import joblib
        path = Path(path)
        if not path.exists():
            raise ModelNotTrainedError(f"Letter model not found at {path}. Run: python scripts/train_static.py")
        bundle = joblib.load(path)
        self.model = bundle["model"]
        self.classes = list(bundle["classes"])
        self.kind = bundle.get("kind", "")

    def predict(self, features: np.ndarray) -> Prediction:
        probs = self.model.predict_proba(np.asarray(features, dtype=np.float32).reshape(1, -1))[0]
        return _prediction(self.classes, probs)


class DynamicClassifier:
    """Word classifier: Keras LSTM over SEQ_LEN frames of features."""

    def __init__(self, path: str | Path = DYNAMIC_MODEL, meta: str | Path = DYNAMIC_META):
        path, meta = Path(path), Path(meta)
        if not path.exists() or not meta.exists():
            raise ModelNotTrainedError(f"Word model not found at {path}. Run: python scripts/train_dynamic.py")
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        import keras
        self.model = keras.models.load_model(path, compile=False)
        info = json.loads(meta.read_text())
        self.classes = info["classes"]
        self.texts = info.get("texts", {})
        self.seq_len = info.get("seq_len", SEQ_LEN)

    def predict(self, sequence: np.ndarray) -> Prediction:
        x = np.asarray(sequence, dtype=np.float32)[None]
        probs = np.asarray(self.model(x, training=False))[0]
        return _prediction(self.classes, probs)

    def text(self, gloss: str) -> str:
        return self.texts.get(gloss, gloss)
