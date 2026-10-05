# Animal Mind Reader — MVP

A research prototype for decoding mouse perceptual state from neural recordings.

## Dataset

Designed for DANDI 000231, *A detailed behavioral, videographic, and neural dataset on object recognition in mice*.

The dataset contains synchronized behavioral events, whisker tracking, and extracellular neural recordings in NWB format.

**Important:** this project does not claim to read arbitrary thoughts. The first scientifically testable task is decoding the stimulus/behavioral class associated with a trial from neural activity.

## Pipeline

DANDI/NWB
  -> spike times + trial labels
  -> binned neural activity
  -> train/validation/test split by trial
  -> Logistic Regression baseline
  -> Transformer sequence model
  -> evaluation + confusion matrix
  -> optional Streamlit dashboard

## Quick start

### 1. Environment

```bash
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows:
# .venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Inspect a DANDI session

```bash
python scripts/inspect_dandi.py --dandiset 000231
```

### 3. Download one NWB asset

```bash
python scripts/download_session.py \
  --dandiset 000231 \
  --asset "sub-231CR/sub-231CR_ses-20190921T144923_behavior+ecephys+image.nwb" \
  --output data/session.nwb
```

### 4. Build a trial dataset

```bash
python -m src.make_dataset \
  --nwb data/session.nwb \
  --output data/processed/trials.npz
```

### 5. Train baseline

```bash
python -m src.train_baseline \
  --data data/processed/trials.npz \
  --output models/baseline.joblib
```

### 6. Evaluate

```bash
python -m src.evaluate \
  --data data/processed/trials.npz \
  --model models/baseline.joblib
```

### 7. Transformer

```bash
python -m src.train_transformer \
  --data data/processed/trials.npz \
  --output models/transformer.pt
```

### 8. Dashboard

```bash
streamlit run app/dashboard.py
```

## Scientific hygiene

- Split by trial, not individual time bins.
- For stronger generalization, split by session or mouse.
- Never report training accuracy as evidence of decoding.
- Compare against shuffled-label and majority-class baselines.
- Keep a held-out test set untouched until model selection is complete.
