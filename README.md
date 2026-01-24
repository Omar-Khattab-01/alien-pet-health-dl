# Alien Pet Health Deep Learning Classifier

A Jupyter-driven workflow that builds and evaluates deep neural networks to predict alien pet health outcomes. The notebook downloads and caches the dataset, applies structured preprocessing, tunes regularization strategies, and reports production-ready metrics and plots.

## Features
- One-click dataset caching with validation of required columns and target labels.
- Reproducible train/validation/test split with scikit-learn pipelines for scaling/encoding.
- Baseline dense network plus experiments on overfitting checks, early stopping, dropout, and L2 regularization.
- Rich evaluation: F1, precision/recall, ROC-AUC, and saved learning-curve/ROC plots.
- Self-contained notebook with helper utilities; figures land in `figs/` for sharing.

## Tech Stack
- **Backend/ML:** Python 3.10+, TensorFlow/Keras, scikit-learn, NumPy, pandas
- **Visualization:** Matplotlib
- **Tooling:** Jupyter Notebook

## Project Structure
- `src/alien-pet-health-deep-learning.ipynb` — main notebook with data loading, modeling experiments, and evaluation. Symlinks in `src/data` and `src/figs` point to the shared folders below so paths stay consistent.
- `data/` — cached dataset (`alien_pet_health-realism-clean.csv`) downloaded automatically if missing.
- `figs/` — generated training and evaluation plots.
- `docs/` — placeholder for screenshots and documentation assets.
- `requirements.txt` — Python dependencies to reproduce the environment.
- `LICENSE` — MIT license.

## How to Run
1. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
   ```
2. Install dependencies:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
3. Launch the notebook from the repository root:
   ```bash
   jupyter notebook src/alien-pet-health-deep-learning.ipynb
   ```
4. Run cells sequentially. The dataset downloads to `data/` on first run; plots are written to `figs/`.

## What This Project Demonstrates
- End-to-end deep learning experimentation with clear separation of data prep, training, and evaluation steps.
- Regularization and generalization techniques (early stopping, dropout, L2) applied to tabular classification.
- Metric-driven reporting (F1/ROC-AUC) suitable for stakeholder-ready summaries.
- Reusable helper utilities for dataset caching and preprocessing inside a single notebook workflow.

## Future Improvements
- Extract reusable preprocessing/model code into a small Python package under `src/` for easier testing and reuse.
- Add automated checks (linting, formatting, unit tests) and CI to guard experiment code.
- Parameterize training runs (Hydra/Argparse) and log experiments to MLFlow or Weights & Biases.
- Provide a lightweight Streamlit demo for quick model exploration once the model is frozen.

## Author
Omar Khattab
