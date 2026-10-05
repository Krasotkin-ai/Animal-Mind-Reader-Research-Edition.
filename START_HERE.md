# START HERE

1. Install dependencies:
   `pip install -r requirements.txt`

2. Inspect a downloaded session:
   `python scripts/inspect_nwb.py --nwb data/session.nwb`

3. Build the trial dataset:
   `python -m src.make_dataset --nwb data/session.nwb --output data/processed/trials.npz`

4. Train:
   `python -m src.train_baseline --data data/processed/trials.npz --output models/baseline.joblib`

5. Evaluate:
   `python -m src.evaluate --data data/processed/trials.npz --model models/baseline.joblib`

6. Launch UI:
   `streamlit run app/dashboard.py`

7. Browser-only preview:
   open `app/demo.html`
