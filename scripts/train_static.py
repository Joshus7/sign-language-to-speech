"""Train the letter (static sign) classifier on data/features/letters.npz.

Stratified 70/15/15 train/validation/test split. An MLP and a RandomForest are trained
on augmented training data; the one with the better validation accuracy is kept,
evaluated once on the test set and saved to models/trained/static_letters.joblib.

Reports: reports/static_report.md, reports/static_confusion.png

Usage:
    python scripts/train_static.py [--copies 4]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import matplotlib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from signspeak.core.augment import augment_static  # noqa: E402
from signspeak.core.models import STATIC_MODEL, update_splits  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ROOT / "data" / "features" / "letters.npz"
REPORTS = ROOT / "reports"


def plot_confusion(cm: np.ndarray, labels: list[str], path: Path, title: str) -> None:
    cmn = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(0.42 * len(labels) + 3, 0.42 * len(labels) + 2.5))
    ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=90 if max(map(len, labels)) > 2 else 0)
    ax.set_yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if cm[i, j]:
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=7,
                        color="white" if cmn[i, j] > 0.5 else "black")
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--copies", type=int, default=4, help="augmented copies per training sample")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if not FEATURES.exists():
        sys.exit("No features - run scripts/extract_landmarks.py first")
    d = np.load(FEATURES)
    X, y, ids, src = d["X"], d["y"], d["id"], d["source"]
    idx = np.arange(len(X))
    tr, rest = train_test_split(idx, test_size=0.30, stratify=y, random_state=args.seed)
    va, te = train_test_split(rest, test_size=0.50, stratify=y[rest], random_state=args.seed)
    print(f"samples: {len(X)}  train {len(tr)} / val {len(va)} / test {len(te)}  classes {len(set(y))}")
    Xtr, ytr = augment_static(X[tr], y[tr], args.copies, seed=args.seed)

    candidates = {
        "MLP": make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(256, 128), alpha=1e-3,
                                                             max_iter=400, early_stopping=True,
                                                             random_state=args.seed)),
        "RandomForest": RandomForestClassifier(n_estimators=400, n_jobs=-1, random_state=args.seed),
    }
    results = {}
    for name, model in candidates.items():
        t0 = time.perf_counter()
        model.fit(Xtr, ytr)
        acc = accuracy_score(y[va], model.predict(X[va]))
        results[name] = acc
        print(f"{name:13s} val acc {acc:.3f}  ({time.perf_counter() - t0:.0f} s)")
    best = max(results, key=results.get)
    model = candidates[best]
    pred = model.predict(X[te])
    test_acc = accuracy_score(y[te], pred)
    print(f"best: {best}  test acc {test_acc:.3f}")

    labels = sorted(set(y))
    REPORTS.mkdir(exist_ok=True)
    cm = confusion_matrix(y[te], pred, labels=labels)
    plot_confusion(cm, labels, REPORTS / "static_confusion.png", f"Letters - {best}, test acc {test_acc:.1%}")
    per_source = {s: accuracy_score(y[te][src[te] == s], pred[src[te] == s]) for s in sorted(set(src[te]))}
    report = classification_report(y[te], pred, labels=labels, digits=3, zero_division=0)
    top = sorted(((cm[i, j], labels[i], labels[j]) for i in range(len(labels)) for j in range(len(labels))
                  if i != j and cm[i, j]), reverse=True)[:8]
    md = ["# Static letter classifier", "",
          f"Samples: {len(X)} ({', '.join(f'{s}: {(src == s).sum()}' for s in sorted(set(src)))}), "
          f"{len(labels)} classes. Stratified split train {len(tr)} / val {len(va)} / test {len(te)}; "
          f"training set augmented x{args.copies + 1} (rotation, aspect, mirror, jitter).", "",
          "| model | validation accuracy |", "|---|---|",
          *[f"| {n} | {a:.1%} |" for n, a in results.items()], "",
          f"**Selected: {best} - test accuracy {test_acc:.1%}**", "",
          "Test accuracy per source: " + ", ".join(f"{s} {a:.1%}" for s, a in per_source.items()), "",
          "Most frequent confusions (true -> predicted): " +
          (", ".join(f"{a}->{b} ({n})" for n, a, b in top) or "none"), "",
          "![confusion matrix](static_confusion.png)", "", "```", report, "```"]
    (REPORTS / "static_report.md").write_text("\n".join(md) + "\n")

    # Refit on train + val (more data) for the shipped model.
    final_idx = np.concatenate([tr, va])
    Xf, yf = augment_static(X[final_idx], y[final_idx], args.copies, seed=args.seed + 1)
    model.fit(Xf, yf)
    STATIC_MODEL.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "classes": list(model.classes_), "kind": best, "test_accuracy": test_acc},
                STATIC_MODEL, compress=3)
    update_splits("letters", {"train": ids[tr].tolist(), "val": ids[va].tolist(), "test": ids[te].tolist()})
    print(f"saved {STATIC_MODEL.relative_to(ROOT)} ({STATIC_MODEL.stat().st_size / 1e6:.1f} MB), "
          f"reports/static_report.md, reports/static_confusion.png")


if __name__ == "__main__":
    main()
