"""Train the word (dynamic sign) LSTM on the WLASL landmark sequences in data/features/words/.

Per word, at least one clip goes to validation and one to test (there are only 4-14
clips per word). Each training clip is turned into many samples: random crops of the
signing span (time warp), one shared rotation / aspect / mirror / position change per
sample and per-frame jitter, then resampled to 30 frames.

Outputs: models/trained/dynamic_words.keras + dynamic_words.json,
         reports/dynamic_report.md, reports/dynamic_confusion.png, reports/dynamic_training.png

Usage:
    python scripts/train_dynamic.py [--variants 60] [--epochs 80]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train_static import plot_confusion  # noqa: E402
from signspeak.core.augment import augment_sequence, time_warp  # noqa: E402
from signspeak.core.features import FRAME_DIM, SEQ_LEN, active_span, resample  # noqa: E402
from signspeak.core.models import DYNAMIC_META, DYNAMIC_MODEL, update_splits  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ROOT / "data" / "features"
REPORTS = ROOT / "reports"


def load_sequences():
    index = json.loads((FEATURES / "words_index.json").read_text())
    signers_file = ROOT / "data" / "raw" / "wlasl" / "signers.json"
    signer_of = json.loads(signers_file.read_text()) if signers_file.exists() else {}
    seqs, labels, clips, signers = [], [], [], []
    for gloss in index["glosses"]:
        for entry in index["clips"].get(gloss, []):
            seq = np.load(FEATURES / entry["feature"])
            a, b = active_span(seq)
            seqs.append(seq[a:b])
            labels.append(gloss)
            clips.append(entry["clip"])
            signers.append(signer_of.get(entry["clip"].removeprefix("wlasl/"), f"unknown-{len(clips)}"))
    return seqs, np.array(labels), clips, np.array(signers, dtype=object), index["glosses"]


def split_per_class(labels: np.ndarray, signers: np.ndarray, rng: np.random.Generator):
    """Per word: >= 20 % of clips to test and >= 15 % to validation (at least one each), assigning whole
    signers so nobody's test clip of a word has a training twin by the same person."""
    tr, va, te = [], [], []
    for g in sorted(set(labels)):
        idx = np.flatnonzero(labels == g)
        groups: dict = {}
        for i in idx:
            groups.setdefault(signers[i], []).append(int(i))
        order = [groups[k] for k in rng.permutation(np.array(list(groups), dtype=object))]
        order.sort(key=len)  # small signer groups first, so test/val don't swallow a prolific signer
        need_te, need_va = max(1, int(round(0.2 * len(idx)))), max(1, int(round(0.15 * len(idx))))
        g_te, g_va, g_tr = [], [], []
        for grp in order:
            if len(g_te) < need_te and len(idx) - len(g_te) - len(grp) >= 2:
                g_te += grp
            elif len(g_va) < need_va and len(idx) - len(g_te) - len(g_va) - len(grp) >= 1:
                g_va += grp
            else:
                g_tr += grp
        te += g_te
        va += g_va
        tr += g_tr
    return np.array(tr), np.array(va), np.array(te)


def make_samples(seqs, y_idx, ids, variants: int, rng: np.random.Generator):
    X, Y = [], []
    for i in ids:
        X.append(resample(seqs[i]))
        Y.append(y_idx[i])
        for _ in range(variants):
            X.append(resample(augment_sequence(time_warp(seqs[i], rng), rng)))
            Y.append(y_idx[i])
    return np.stack(X).astype(np.float32), np.array(Y)


def build_model(n_classes: int):
    import keras
    from keras import layers
    model = keras.Sequential([
        keras.Input((SEQ_LEN, FRAME_DIM)),
        layers.Masking(mask_value=0.0),
        layers.LSTM(96, return_sequences=True),
        layers.Dropout(0.3),
        layers.LSTM(64),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.Dense(n_classes, activation="softmax"),
    ])
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def train(X, Y, Xv, Yv, n_classes, epochs, seed):
    import keras
    keras.utils.set_random_seed(seed)
    model = build_model(n_classes)
    cb = [keras.callbacks.EarlyStopping(monitor="val_accuracy" if Xv is not None else "accuracy", patience=15,
                                        restore_best_weights=True, mode="max")]
    hist = model.fit(X, Y, validation_data=(Xv, Yv) if Xv is not None else None, epochs=epochs,
                     batch_size=64, callbacks=cb, verbose=0, shuffle=True)
    return model, hist.history


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", type=int, default=60, help="augmented samples per training clip")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    if not (FEATURES / "words_index.json").exists():
        sys.exit("No word features - run scripts/extract_landmarks.py first")
    rng = np.random.default_rng(args.seed)
    seqs, labels, clips, signers, glosses = load_sequences()
    classes = [g for g in glosses if g in set(labels)]
    y_idx = np.array([classes.index(g) for g in labels])
    tr, va, te = split_per_class(labels, signers, rng)
    leak = sum((labels[i], signers[i]) in {(labels[j], signers[j]) for j in tr} for i in te)
    print(f"clips: {len(seqs)}  train {len(tr)} / val {len(va)} / test {len(te)}  words {len(classes)}  "
          f"signers {len(set(signers))}  test clips with same signer+word in train: {leak}")

    X, Y = make_samples(seqs, y_idx, tr, args.variants, rng)
    Xv = np.stack([resample(seqs[i]) for i in va])
    Xt = np.stack([resample(seqs[i]) for i in te])
    print(f"training samples after augmentation: {len(X)}")
    model, hist = train(X, Y, Xv, y_idx[va], len(classes), args.epochs, args.seed)
    val_acc = accuracy_score(y_idx[va], model.predict(Xv, verbose=0).argmax(1))
    probs = model.predict(Xt, verbose=0)
    pred = probs.argmax(1)
    test_acc = accuracy_score(y_idx[te], pred)
    top3 = float(np.mean([y in np.argsort(p)[-3:] for y, p in zip(y_idx[te], probs)]))
    print(f"val acc {val_acc:.3f}  test acc {test_acc:.3f}  test top-3 {top3:.3f}  ({len(hist['loss'])} epochs)")

    REPORTS.mkdir(exist_ok=True)
    cm = confusion_matrix(y_idx[te], pred, labels=range(len(classes)))
    plot_confusion(cm, classes, REPORTS / "dynamic_confusion.png", f"Words - LSTM, test acc {test_acc:.1%}")
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(hist["accuracy"], label="train (augmented)")
    ax.plot(hist["val_accuracy"], label="validation clips")
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORTS / "dynamic_training.png", dpi=120)
    plt.close(fig)
    report = classification_report(y_idx[te], pred, labels=range(len(classes)), target_names=classes,
                                   digits=3, zero_division=0)
    counts = {g: int((labels == g).sum()) for g in classes}
    md = ["# Dynamic word classifier (LSTM)", "",
          f"WLASL clips: {len(seqs)} for {len(classes)} words ({min(counts.values())}-{max(counts.values())} per "
          f"word, {len(set(signers))} signers). Per-word, signer-disjoint split train {len(tr)} / val {len(va)} / "
          f"test {len(te)} clips (test clips with the same signer+word in train: {leak}); each training clip "
          f"gives {args.variants + 1} samples (time-warp crops, rotation, aspect, mirror, position shift, jitter).",
          "", "Model: Masking -> LSTM(96) -> LSTM(64) -> Dense(64) -> softmax over 30 frames x 134 features.", "",
          "| metric | value |", "|---|---|",
          f"| validation accuracy | {val_acc:.1%} |", f"| **test accuracy** | **{test_acc:.1%}** |",
          f"| test top-3 accuracy | {top3:.1%} |", f"| chance level | {1 / len(classes):.1%} |", "",
          "With one or two test clips per word these numbers are noisy (each clip is "
          f"{1 / len(te):.1%} of the test set).", "",
          "Clips per word: " + ", ".join(f"{g} {n}" for g, n in counts.items()), "",
          "![confusion matrix](dynamic_confusion.png)", "", "![training curve](dynamic_training.png)", "",
          "```", report, "```"]
    (REPORTS / "dynamic_report.md").write_text("\n".join(md) + "\n")

    # Shipped model: retrain on train + val clips with the epoch count found above.
    final = np.concatenate([tr, va])
    Xf, Yf = make_samples(seqs, y_idx, final, args.variants, rng)
    best_epochs = int(np.argmax(hist["val_accuracy"])) + 1
    final_model, _ = train(Xf, Yf, None, None, len(classes), max(best_epochs, 20), args.seed)
    final_test = accuracy_score(y_idx[te], final_model.predict(Xt, verbose=0).argmax(1))
    print(f"final model (train+val, {max(best_epochs, 20)} epochs) test acc {final_test:.3f}")
    DYNAMIC_MODEL.parent.mkdir(parents=True, exist_ok=True)
    final_model.save(DYNAMIC_MODEL)
    vocab = yaml.safe_load((ROOT / "config" / "vocabulary.yaml").read_text())["words"]
    DYNAMIC_META.write_text(json.dumps({"classes": classes, "texts": {w["gloss"]: w["text"] for w in vocab},
                                        "seq_len": SEQ_LEN, "frame_dim": FRAME_DIM,
                                        "test_accuracy": round(test_acc, 4),
                                        "final_model_test_accuracy": round(final_test, 4)}, indent=1))
    update_splits("words", {"train": [clips[i] for i in tr], "val": [clips[i] for i in va],
                            "test": [clips[i] for i in te]})
    with open(REPORTS / "dynamic_report.md", "a") as fh:
        fh.write(f"\nShipped model (retrained on train + val clips): test accuracy {final_test:.1%}.\n")
    print(f"saved {DYNAMIC_MODEL.relative_to(ROOT)} ({DYNAMIC_MODEL.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
