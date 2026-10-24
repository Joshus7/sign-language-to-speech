"""Download the public sign-language datasets into data/raw/ (git-ignored).

Sources
-------
1. Hugging Face ``sid220/asl-now-fingerspelling`` (MIT) - ASL letters A-Z as
   MediaPipe hand landmarks (JSON, 21 points), signed by several participants.
2. Hugging Face ``EitanG98/asl_letters`` (Unlicense; a mirror of the Roboflow
   "American Sign Language Letters" set) - photos of the letters A-Z. HandLandmarker
   is run over them by scripts/extract_landmarks.py, and they are used in the demo.
3. Hugging Face ``Voxel51/WLASL`` (C-UDA, academic/computational use) - only the
   clips of the ~20 word glosses in config/vocabulary.yaml are downloaded.
4. Optional, Kaggle ``grassknoted/asl-alphabet`` (GPL-2.0) - a few hundred images
   per letter plus space/del/nothing. Needs a Kaggle token (~/.kaggle/access_token,
   ~/.kaggle/kaggle.json, KAGGLE_API_TOKEN or KAGGLE_USERNAME / KAGGLE_KEY); it is
   skipped with setup instructions when none is set up.

The script is re-runnable: files that are already there are skipped.

Usage:
    python scripts/download_data.py                  # Hugging Face sources
    python scripts/download_data.py --only wlasl
    python scripts/download_data.py --only kaggle --per-class 300
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import yaml
from huggingface_hub import hf_hub_download, snapshot_download

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

HF_LETTER_LANDMARKS = "sid220/asl-now-fingerspelling"
HF_LETTER_IMAGES = "EitanG98/asl_letters"
HF_WLASL = "Voxel51/WLASL"
KAGGLE_ALPHABET = "grassknoted/asl-alphabet"


def vocabulary() -> list[str]:
    cfg = yaml.safe_load((ROOT / "config" / "vocabulary.yaml").read_text())
    return [w["gloss"] for w in cfg["words"]]


def count_files(path: Path, pattern: str = "*") -> int:
    return sum(1 for p in path.rglob(pattern) if p.is_file()) if path.exists() else 0


def download_letter_landmarks() -> None:
    out = RAW / "asl_now"
    print(f"== {HF_LETTER_LANDMARKS} (letter landmarks) -> {out.relative_to(ROOT)}")
    snapshot_download(HF_LETTER_LANDMARKS, repo_type="dataset", local_dir=out,
                      allow_patterns=["*/*.json"], max_workers=16)
    print(f"   {count_files(out, '*.json')} landmark files\n")


def download_letter_images() -> None:
    out = RAW / "asl_letters"
    print(f"== {HF_LETTER_IMAGES} (letter photos) -> {out.relative_to(ROOT)}")
    snapshot_download(HF_LETTER_IMAGES, repo_type="dataset", local_dir=out,
                      allow_patterns=["train_val/*/*.jpg"], max_workers=8)
    print(f"   {count_files(out, '*.jpg')} images\n")


def download_wlasl(max_per_word: int | None) -> None:
    out = RAW / "wlasl"
    print(f"== {HF_WLASL} (word clips) -> {out.relative_to(ROOT)}")
    meta = Path(hf_hub_download(HF_WLASL, "samples.json", repo_type="dataset", local_dir=out))
    samples = json.loads(meta.read_text())["samples"]
    words = vocabulary()
    chosen: dict[str, list[str]] = {w: [] for w in words}
    for s in samples:
        gloss = s["gloss"]["label"]
        if gloss in chosen:
            chosen[gloss].append(s["filepath"])
    missing = [w for w, files in chosen.items() if not files]
    if missing:
        print(f"!! not in WLASL: {missing} - replace them in config/vocabulary.yaml")
    if max_per_word:
        chosen = {w: sorted(f)[:max_per_word] for w, f in chosen.items()}
    files = sorted(f for fs in chosen.values() for f in fs)
    print(f"   {len(files)} clips for {len(words) - len(missing)} words "
          f"({min(len(v) for v in chosen.values())}-{max(len(v) for v in chosen.values())} per word)")
    snapshot_download(HF_WLASL, repo_type="dataset", local_dir=out, allow_patterns=files, max_workers=8)
    index = {w: [f for f in fs if (out / f).exists()] for w, fs in chosen.items()}
    (out / "index.json").write_text(json.dumps(index, indent=1))
    print(f"   {sum(map(len, index.values()))} clips on disk, index written to wlasl/index.json\n")


def kaggle_credentials_present() -> bool:
    if os.environ.get("KAGGLE_API_TOKEN"):
        return True
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    home = Path.home() / ".kaggle"
    return (home / "access_token").exists() or (home / "kaggle.json").exists()


def download_kaggle_alphabet(per_class: int) -> None:
    out = RAW / "kaggle_asl_alphabet"
    print(f"== Kaggle {KAGGLE_ALPHABET} (optional, {per_class} images per class) -> {out.relative_to(ROOT)}")
    if (out / ".complete").exists():
        print("   already downloaded, skipped\n")
        return
    if not kaggle_credentials_present():
        print("!! Kaggle credentials not found - skipped (the Hugging Face letter data is enough to train).\n"
              "   1. Sign in at https://www.kaggle.com/settings -> 'API' -> 'Generate New Token' and copy it\n"
              "   2. mkdir -p ~/.kaggle && pbpaste > ~/.kaggle/access_token && chmod 600 ~/.kaggle/access_token\n"
              "      (Linux/Windows: save the token text to ~/.kaggle/access_token; a legacy kaggle.json also works)\n"
              "   3. Re-run: python scripts/download_data.py --only kaggle\n")
        return
    zip_dir = RAW / "_kaggle_zip"
    zip_dir.mkdir(parents=True, exist_ok=True)
    kaggle = shutil.which("kaggle") or str(Path(sys.executable).parent / "kaggle")
    if not list(zip_dir.glob("*.zip")):
        result = subprocess.run([kaggle, "datasets", "download", "-d", KAGGLE_ALPHABET, "-p", str(zip_dir)])
        if result.returncode != 0:
            print("!! Kaggle download failed - check the token and that you accepted the dataset terms\n")
            return
    archive = next(zip_dir.glob("*.zip"))
    rng = random.Random(0)
    with zipfile.ZipFile(archive) as zf:
        by_class: dict[str, list[str]] = {}
        for name in zf.namelist():
            parts = Path(name).parts
            # asl_alphabet_train/asl_alphabet_train/<CLASS>/<file>.jpg
            if len(parts) >= 3 and "train" in parts[0] and name.lower().endswith(".jpg"):
                by_class.setdefault(parts[-2], []).append(name)
        for cls, names in sorted(by_class.items()):
            dest = out / cls
            dest.mkdir(parents=True, exist_ok=True)
            for name in rng.sample(names, min(per_class, len(names))):
                target = dest / Path(name).name
                if not target.exists():
                    target.write_bytes(zf.read(name))
            print(f"   {cls}: {count_files(dest)} images")
    (out / ".complete").write_text("ok")
    shutil.rmtree(zip_dir)
    print()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=["letters", "images", "wlasl", "kaggle"], action="append",
                    help="download only these sources (repeatable)")
    ap.add_argument("--max-per-word", type=int, default=None, help="limit WLASL clips per word")
    ap.add_argument("--per-class", type=int, default=300, help="Kaggle images per letter")
    args = ap.parse_args()
    only = set(args.only or ["letters", "images", "wlasl", "kaggle"])
    RAW.mkdir(parents=True, exist_ok=True)
    if "letters" in only:
        download_letter_landmarks()
    if "images" in only:
        download_letter_images()
    if "wlasl" in only:
        download_wlasl(args.max_per_word)
    if "kaggle" in only:
        download_kaggle_alphabet(args.per_class)
    print("Done.")


if __name__ == "__main__":
    main()
