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

## Features
- Letters: 21 landmarks x (x, y, z) of the largest hand, x/z converted to frame-height units (so 16:9 and 4:3
  frames give the same shape), wrist at the origin, scaled by the wrist -> middle-MCP distance, mirrored for left
  hands. No rotation normalisation - orientation separates letters such as K/P and H/U.
- Words: per frame, both hands (dominant = larger one first) with the 63 shape values + wrist position in the frame +
  hand size + presence flag = 134 values. Position matters for words signed near the face or chest (hello, food,
  I), which hand shape alone cannot tell apart.
- Sequences are resampled to 30 frames by nearest frame, not interpolation: blending a hand frame with an all-zero
  "no hand" frame would invent shapes.

## Letter model
- MLP (256-128) vs RandomForest, picked on validation accuracy -> MLP (95.5 % vs 92.6 % val, 96.0 % test).
  Final model refit on train + val. Saved with joblib (0.6 MB, committed).
- Augmentation x5 in landmark space: +-15 deg rotation, +-12 % aspect, random mirroring, 0.02 jitter.
- The split is stratified-random, not by signer: asl-now has no participant IDs, so test samples may come from
  people also seen in training. The photo subset (different people, different camera) scores a little lower
  (93.4 %), which is a better guess of accuracy for a new user.
- J and Z are motion letters; the datasets capture them as one frame (the end pose), and so does the app.
