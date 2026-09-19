# 🩺 AI for Predictive Healthcare Diagnosis

An end-to-end machine-learning project that predicts disease risk from patient history and
symptoms. It trains and compares **classical ML** (Logistic Regression, Random Forest,
XGBoost) against **TensorFlow neural networks** across five healthcare datasets, explains
every prediction, and serves it all through an interactive **Streamlit** app.

> ⚠️ **Educational portfolio project — not a medical device.** Every prediction is a
> statistical estimate from public datasets and must never be used for real diagnosis.

---

## How it works

<p align="center">
  <img src="docs/pipeline.svg" alt="Animated overview of the diagnosis pipeline: patient input, preprocessing, model comparison, best-model selection, and explainable results" width="100%" />
</p>

<p align="center"><sub>The pipeline lights up stage by stage — this animation renders live on GitHub.</sub></p>

---

## Highlights

- **Five models in one app** — a symptom-based multi-disease checker plus dedicated risk
  models for diabetes, heart disease, cardiovascular disease, and stroke.
- **Classical vs. deep learning** — a fair, reproducible comparison of four algorithms per
  task, with the winner selected automatically (ROC-AUC for binary, macro-F1 for multiclass).
- **Explainable by design** — SHAP for tree models, coefficients for linear models, and
  permutation importance for the neural net, surfaced both globally and per prediction.
- **Conversational symptom chatbot** — describe symptoms in plain English; a Groq-hosted LLM
  maps them to the model's vocabulary, runs the symptom checker, and explains the results
  with an appropriate medical disclaimer.
- **Honest engineering** — proper handling of missing values, class imbalance, data-entry
  outliers, and leakage, with a clear-eyed discussion of limitations.
- **Production-shaped code** — modular `src/` package, saved artifacts, and a `pytest` suite.

## Results (held-out test set)

| Task | Best model | Accuracy | ROC-AUC | Notes |
|------|-----------|:--------:|:-------:|-------|
| Symptom checker (41 diseases) | Logistic Regression | 1.000 | — (top-3 = 1.000) | Small, highly separable — a pattern-recognition demo |
| Diabetes risk | Random Forest | 0.747 | 0.831 | Pima dataset, zero-coded missing values handled |
| Heart disease | Random Forest | 0.853 | 0.922 | Strongest tabular model |
| Cardiovascular disease | **Neural Network (TensorFlow)** | 0.737 | 0.803 | NN wins on the largest (~70k) dataset |
| Stroke risk | Logistic Regression | 0.747 | 0.844 | ~5% positive — accuracy is misleading, judged by ROC-AUC |

**Key insight:** deep learning is not automatically better. On small clinical tables, Random
Forest matches or beats the neural network; TensorFlow only pulls ahead on the large
cardiovascular dataset where it has enough data to learn richer structure.

## The app

Run it and you get, per disease:

1. **Predict** — enter symptoms (searchable multiselect) or patient measurements (auto-generated
   form), pick any trained model, and get a probability gauge (binary) or ranked top-3 diseases
   with descriptions and precautions (symptom checker).
2. **Why this prediction?** — a per-prediction feature-attribution chart, plus (for the four
   disease-risk models) an **AI explanation** that turns the probability and top factors into
   a plain-English, disclaimer-ended summary via a Groq LLM.
3. **Model performance** — side-by-side metrics table, confusion matrix, and feature importances.
4. **Data insights** — class distribution and feature correlations.

The symptom checker adds a **💬 Symptom Chat** tab: a natural-language conversation that
extracts symptoms, predicts, and explains — powered by a Groq LLM.

## Project structure

```
ai-healthcare-diagnosis/
├── .env.example              # Groq API key template (copy to .env)
├── data/raw/                 # Kaggle CSVs (git-ignored; see "Data" below)
├── docs/
│   └── pipeline.svg          # animated pipeline diagram for the README
├── src/
│   ├── config.py             # paths + dataset registry
│   ├── datasets/             # one cleaning/prep module per dataset
│   ├── models/               # classical + Keras model factories
│   ├── train.py              # trains & evaluates all models, saves artifacts
│   ├── evaluate.py           # metrics + confusion matrices
│   ├── explain.py            # global + local explainability
│   ├── inference.py          # single-row prediction (used by app & tests)
│   ├── eda.py                # exploratory figures
│   └── llm.py                # Groq LLM: symptom extraction + narration
├── backend/app/              # FastAPI service (main.py, engine.py, schemas.py)
├── frontend/                 # React + Vite single-page app
│   └── src/components/       # UI: form, result, charts, chat, performance
├── app/streamlit_app.py      # alternative all-in-one Streamlit UI
├── notebooks/exploration.ipynb
├── tests/                    # pytest: data contracts + inference smoke tests
├── models/                   # trained artifacts (git-ignored, regenerated)
├── reports/figures/          # generated charts
└── requirements.txt
```

## Architecture

There are two ways to use the models:

- **React + FastAPI (primary):** a `FastAPI` backend (`backend/app`) exposes the trained
  models over a REST API; a `React` + `Vite` frontend (`frontend/`) consumes it. The backend
  reuses the exact same `src/` engine — no ML logic is duplicated.
- **Streamlit (alternative):** `app/streamlit_app.py` is a single-process UI that calls `src/`
  directly. Handy for quick local demos.

### API endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/api/health` | Liveness check plus list of trained model slugs |
| GET | `/api/models` | List trained models |
| GET | `/api/models/{slug}` | Schema, metrics, and figure URLs |
| POST | `/api/models/{slug}/predict` | Probabilities + feature contributions |
| POST | `/api/models/{slug}/ai-explanation` | Groq plain-English risk narration (binary) |
| POST | `/api/chat` | Symptom chatbot (extract → predict → explain) |
| GET | `/figures/*` | Static confusion-matrix / importance / EDA images |

## Data

Download these Kaggle datasets and place the CSVs in `data/raw/`:

| File | Source |
|------|--------|
| `dataset.csv`, `Symptom-severity.csv`, `symptom_Description.csv`, `symptom_precaution.csv` | [Disease Symptom Prediction](https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset) |
| `diabetes.csv` | [Pima Indians Diabetes](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) |
| `heart_disease_uci.csv` | [Heart Disease UCI](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data) |
| `cardio_train.csv` | [Cardiovascular Disease](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset) |
| `healthcare-dataset-stroke-data.csv` | [Stroke Prediction](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset) |

## Setup & run

```powershell
# 1. Create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1        # Windows PowerShell
# source .venv/bin/activate         # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Put the CSVs in data/raw/  (see "Data" above)

# 3b. (Optional) enable the symptom chatbot: copy .env.example to .env and add your
#     Groq API key from https://console.groq.com/keys
copy .env.example .env

# 4. Train all models (writes to models/ and reports/figures/)
python -m src.train

# 5. Generate EDA + explainability figures (optional)
python -m src.eda
python -m src.explain
```

Train a single dataset with `python -m src.train --dataset heart`.

### Run the React + FastAPI app

Two terminals:

```powershell
# Terminal 1 — backend (from project root, venv active)
python -m uvicorn backend.app.main:app --reload --port 8000

# Terminal 2 — frontend
cd frontend
npm install        # first time only
npm run dev        # http://localhost:5173
```

Open http://localhost:5173. The Vite dev server proxies `/api` and `/figures` to the backend,
so no extra configuration is needed. API docs are at http://localhost:8000/docs.

### Or run the Streamlit app

```powershell
streamlit run app/streamlit_app.py
```

## Training pipeline

Each dataset module returns a standardised `PreparedData` bundle (cleaned features, encoded
target, column roles, and app metadata). A shared scikit-learn `ColumnTransformer` handles
median/most-frequent imputation, scaling, and one-hot encoding, so **training and the app use
the exact same preprocessing**. Four models are trained per task; the best is chosen by the
task-appropriate metric and all artifacts (preprocessor, models, metadata, metrics) are saved
under `models/<dataset>/`.

## Testing

```powershell
pytest -q
```

Covers dataset contracts (shapes, target consistency, column roles, symptom encoding) and an
end-to-end inference smoke test that every model returns a valid probability distribution.

## Limitations & ethics

- Public, simplified datasets — not representative of real clinical populations.
- The symptom checker's perfect score reflects an easy dataset, not diagnostic ability.
- No calibration, external validation, or fairness auditing has been performed.
- **This project must not be used for actual medical decisions.**

## Tech stack

**ML/Backend:** Python · TensorFlow/Keras · scikit-learn · XGBoost · SHAP · pandas ·
FastAPI · Groq (LLM) · pytest
**Frontend:** React · Vite · Recharts
**Alt UI:** Streamlit · Plotly
