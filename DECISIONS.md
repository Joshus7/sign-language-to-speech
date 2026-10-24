# Decisions

Ambiguities in the brief and how they were resolved.

## Environment
- Python 3.12 (Homebrew) in `.venv`; TensorFlow 2.21 + MediaPipe 1.1.1 install side by side (protobuf 7.36).
- The HandLandmarker model (`hand_landmarker.task`, 7.8 MB) is downloaded to the git-ignored `models/`.
  Trained models go in `models/trained/`, which *is* committed (all far below 50 MB) so the app works right after
  cloning - the brief allows this for small models.

## Data
- **Letters from Hugging Face instead of Kaggle.** The brief names Kaggle `grassknoted/asl-alphabet`, but the data
  rule prefers Hugging Face, and no Kaggle token was set up. Two permissively licensed HF sets are used:
  `sid220/asl-now-fingerspelling` (2,123 MediaPipe hand-landmark samples from several webcam users, MIT) and
  `EitanG98/asl_letters` (720 phone photos, a mirror of the Roboflow "American Sign Language Letters" set,
  Unlicense). The Kaggle set stays available as an optional extra (`--only kaggle`, 300 images per class); it
  also adds `space` / `del` / `nothing` classes, which become sentence-builder actions.
- **WLASL from `Voxel51/WLASL`** (the original WLASL clips with gloss labels). All 20 words of the brief's list exist,
  so none were replaced - but there are only 4-14 clips per word (158 in total). Only those clips are downloaded
  (~70 MB). The small amount of data is the main limit on word accuracy; augmentation is used to compensate.
- The asl-now landmarks are stored in normalised image coordinates without the image size; a 4:3 frame (the
  browser webcam default, 640x480) is assumed when correcting the aspect ratio.
