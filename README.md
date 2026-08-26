# Clipper

Clipper is a local web app for trimming long, mostly-static-camera footage into the parts an operator wants to keep.

The first target workflow is volleyball: label short 1-second samples as `keep` or `cut`, train an adaptive classifier on those decisions, analyze the full timeline, preview the edited playback without rewriting the source video, then export a real edited MP4/MOV with ffmpeg.

## Requirements

- Node 22+
- Python 3.12 and [uv](https://docs.astral.sh/uv/)
- ffmpeg and ffprobe on `PATH`

## Install

```bash
npm install
npm run install:vision
```

## Run

```bash
npm run dev
```

Open `http://127.0.0.1:5173`.

## Workflow

1. Upload a source video.
2. Clipper automatically generates 200 one-second training samples.
3. Review the randomized samples as `keep`, `cut`, or `unsure`. Clipper adds more samples as needed until you have at least 40 usable examples in both the `keep` and `cut` classes.
4. Train the classifier, then audit its predictions on 10 held-out clips in Model Check.
5. Analyze the full timeline.
6. Preview the virtual edit with the custom scrubber.
7. Export the approved cut.

## How The Classifier Works

The active implementation uses a fully local VideoMAE encoder with a lightweight classifier:

- ffmpeg samples 16 frames across a five-second context window centered on each labeled second.
- The frozen `MCG-NJU/videomae-base` encoder runs through PyTorch on Apple Metal when available and produces a 768-dimensional semantic video embedding.
- Model weights and embeddings are cached locally under `data/models/` and `data/jobs/<job-id>/embeddings/`; videos and labels are never uploaded.
- `Unsure` clips count as reviewed but are excluded from classifier training.
- A class-balanced regularized logistic layer searches for the best checkpoint on a deterministic 20% validation split, reserves five `keep` and five `cut` clips as an unseen audit set, then refits on every remaining usable label.
- The full video is sampled once per second, scored, smoothed, merged, padded, and converted into keep segments.
- Every label change is persisted to `data/jobs/<job-id>/labels.json` for reuse in later training experiments.

The first training run downloads approximately 377 MB of model weights. After that, set `CLIPPER_VIDEOMAE_OFFLINE=1` to force cached-only operation. The default MCG-NJU weights are CC BY-NC 4.0 and are intended for noncommercial use.

## Offline Quality Experiments

The legacy research pipeline uses OpenCV and does not change the active web
app or its saved labels. It extracts dense, color-independent Farneback flow
over longer windows, caches the features under
`data/jobs/<job-id>/experiments/`, and compares linear SVM, RBF SVM, and random
forest classifiers.

```bash
node server/scripts/run-motion-experiment.js <job-id> farneback-160-context5

python3 server/scripts/evaluate-motion-classifiers.py \
  --features data/jobs/<job-id>/experiments/farneback-160-context5-features.json \
  --job data/jobs/<job-id>/job.json \
  --output data/jobs/<job-id>/experiments/farneback-160-context5-final-result.json \
  --model-output data/jobs/<job-id>/experiments/farneback-160-context5-model.xml
```

Evaluation uses five time-blocked folds and removes training windows that
overlap a validation window. The optional model export fits the selected
classifier on all usable `keep` and `cut` labels after evaluation. This
pipeline requires Python 3 with OpenCV (`cv2`); feature extraction and model
search can take several minutes by design.
