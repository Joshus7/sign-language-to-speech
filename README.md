# Sign Language to Speech

A person signs in front of the webcam; the app recognises ASL letters and words, builds a sentence and speaks it
aloud - fully offline (MediaPipe HandLandmarker + small scikit-learn / Keras models + pyttsx3).

> Work in progress - see the phases in the commit history. The full README is written in phase 5.

## Quick start

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_models.py     # hand_landmarker.task -> models/
python scripts/download_data.py       # Hugging Face datasets -> data/raw/
python scripts/preview.py --images    # hand overlay on dataset photos (also --words, --webcam 0)
```

## Datasets and credits

Used for educational purposes only; nothing from the datasets is committed.

- ASL letters (landmarks): [sid220/asl-now-fingerspelling](https://huggingface.co/datasets/sid220/asl-now-fingerspelling) - MIT
- ASL letters (photos): [EitanG98/asl_letters](https://huggingface.co/datasets/EitanG98/asl_letters) - Unlicense
- Words: [Voxel51/WLASL](https://huggingface.co/datasets/Voxel51/WLASL) - C-UDA (academic / computational use)
