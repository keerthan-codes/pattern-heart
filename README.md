# Comparative PR-Based Disease Prediction

A local, educational Pattern Recognition application comparing K-Nearest Neighbours, Gaussian Naive Bayes, and Decision Tree classifiers on the UCI Cleveland Heart Disease dataset. React presents clinical inputs and evaluation evidence; Flask validates inputs and runs saved Scikit-learn pipelines.

## Problem and objectives

Explore whether historical clinical measurements can distinguish records labeled with heart disease, compare three classic algorithms under a common evaluation protocol, and explain model selection transparently. This is a learning prototype, not a clinical diagnostic system.

## Quick start

Use Node.js 22.12+ (or a supported newer release) and Python 3.14, matching the supplied artifact environment. Existing dependencies and the backend virtual environment have been preserved. On a new checkout:

```sh
cd backend
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python train.py
python app.py
```

The backend listens on http://127.0.0.1:5001. In another terminal:

```sh
cd frontend
npm install
npm run dev
```

Open the URL Vite prints, normally http://localhost:5173. Vite proxies `/api` to Flask. Do not use Vite preview alone to exercise the API: the development proxy is configured on the dev server. A production deployment must route `/api` to Flask or build with `VITE_API_BASE_URL=https://your-backend/api`. Flask's built-in server is for local development.

With the current installed environment and supplied trained artifacts, startup only requires:

```sh
# Terminal 1, project root
backend/.venv/bin/python backend/app.py
# Terminal 2, project root
cd frontend
npm run dev
```

## Dataset and attribution

Source: [UCI Heart Disease](https://archive.ics.uci.edu/dataset/45/heart+disease), Janosi, Steinbrunn, Pfisterer & Detrano (1989), DOI [10.24432/C52P4X](https://doi.org/10.24432/C52P4X), CC BY 4.0. Only `processed.cleveland.data` is used: 303 records, 13 features, six missing feature cells represented by `?`. Original targets 1–4 become positive label 1; original target 0 remains negative label 0.

`train.py` downloads the [official archive](https://archive.ics.uci.edu/static/public/45/heart+disease.zip) when the local file is absent. If the download fails, manually extract `processed.cleveland.data` into `backend/data/` and rerun training. No synthetic or substitute dataset is used. The raw file is included, with its SHA-256 stored in the evaluation report.

| Feature | Meaning / original encoding |
| --- | --- |
| age | Age in years |
| sex | Dataset sex: 0 female, 1 male |
| cp | Chest pain: 1 typical angina, 2 atypical angina, 3 non-anginal pain, 4 asymptomatic |
| trestbps | Resting blood pressure on admission, mm Hg |
| chol | Serum cholesterol, mg/dL |
| fbs | Fasting blood sugar >120 mg/dL: 0 false, 1 true |
| restecg | 0 normal, 1 ST–T abnormality, 2 probable/definite left ventricular hypertrophy |
| thalach | Maximum heart rate achieved, bpm |
| exang | Exercise-induced angina: 0 no, 1 yes |
| oldpeak | Exercise-induced ST depression relative to rest |
| slope | Peak exercise ST slope: 1 upsloping, 2 flat, 3 downsloping |
| ca | Number of major vessels colored by fluoroscopy: 0–3 |
| thal | Thallium test: 3 normal, 6 fixed defect, 7 reversible defect |

The API's numeric ranges are broad application input constraints, not diagnostic thresholds or guarantees of model validity. They appear in `/api/features`; all fields are required at inference. Missing historical training values are imputed. Original categorical codes are preserved, especially `cp` and `thal`, which differ from several repackaged datasets.

## Architecture

```text
React / React Router / Recharts
           ↓ JSON via Vite proxy
Flask validation ← shared schema.py → training
           ↓                         ↓
Saved sklearn pipelines ← stratified CV + train-only preprocessing
           ↓
Three labels + CV-selected preferred classifier
```

- `frontend/src/pages/`: Overview, Predict, Compare Models.
- `frontend/src/components/Shared.jsx`: reusable status and disclaimer components.
- `frontend/src/services/`: JSON API client with timeout/error handling, reusable data-loading hook, and classifier descriptions.
- `backend/schema.py`: single source for feature order, encodings, labels, and validation constraints.
- `backend/train.py`: official dataset retrieval, preprocessing, tuning, selection, evaluation, artifact export.
- `backend/app.py`: API factory, artifact validation, request validation, and inference.
- `backend/models/`: Joblib pipeline bundle and actual JSON evaluation report.
- `backend/tests/test_api.py`: automated API and artifact tests.

## Preprocessing and algorithms

An 80/20 stratified split uses `random_state=42`: 242 training records (131 negative, 111 positive) and 61 test records (33 negative, 28 positive). Each model uses a Scikit-learn Pipeline and ColumnTransformer. Numeric features receive median imputation; categorical features receive most-frequent imputation and dense one-hot encoding with original category vocabularies. KNN additionally standardizes numerical features. All learned preprocessing is fitted inside training folds only.

Five-fold shuffled stratified cross-validation, seed 42, searches:

- KNN: neighbors 3, 5, 7, 9, 11, 15; uniform or distance weighting.
- GaussianNB: variance smoothing 1e-11, 1e-9, 1e-7, 1e-5, .001, .1.
- Decision Tree: depth 2, 3, 4, 5, unlimited; minimum leaf size 1, 5, 10; no class weighting or balanced weighting; seed 42.

GaussianNB on one-hot features is an educational baseline: binary indicators do not satisfy its Gaussian assumption, and indicators from the same category are dependent. No probability estimates are exposed.

## Evaluation and selection

The criterion is defined before test evaluation: maximize mean disease-positive recall in training cross-validation, then mean F1. Exact ties use fixed candidate/model order. Each selected pipeline is refitted on all training records. The preferred algorithm is selected from CV scores, then all three pipelines are evaluated on the same untouched test records. Models are not refitted on the full dataset afterward, so the saved pipelines correspond to the reported evaluation.

Actual results from the supplied run:

| Model | Accuracy | Precision | Recall | F1 | CV recall | CV F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| KNN | 86.89% | 81.25% | 92.86% | 86.67% | 80.08% | 80.42% |
| Gaussian Naive Bayes | 77.05% | 69.44% | 89.29% | 78.13% | 85.49% | 78.32% |
| Decision Tree | 72.13% | 64.10% | 89.29% | 74.63% | 81.90% | 74.00% |

Preferred: **Gaussian Naive Bayes**, selected by CV recall, even though KNN scored better on the test set. Changing the selection after seeing test results would compromise the evaluation protocol.

Confusion matrices (actual rows, predicted columns; labels `[0,1]`): KNN `[[27,6],[2,26]]`; Naive Bayes `[[22,11],[3,25]]`; Decision Tree `[[19,14],[3,25]]`. Accuracy measures overall correctness; precision measures correctness among positive predictions; recall measures detected positive records; F1 is their harmonic mean. The web application reads all results from the backend; it never hardcodes evaluation metrics.

Exact parameters, fold standard deviations, checksum, class counts, and package versions are in `backend/models/metrics.json`. Artifacts were generated with Python 3.14.2 / Scikit-learn 1.9.1. Retrain after changing library versions. Joblib uses pickle; only load trusted artifacts.

## REST API

Successful responses are JSON objects. Errors use `{"error":"message"}` with optional `fields` validation errors. CORS allows `http://localhost:5173` and `http://127.0.0.1:5173`; override with comma-separated `CORS_ORIGINS` if needed.

| Method / endpoint | Result |
| --- | --- |
| GET `/api/health` | `status`, `models_ready`; 503 if artifacts unavailable |
| GET `/api/features` | `features` metadata and `feature_order`; available without artifacts |
| GET `/api/models` | Classifier IDs, names, preferred flags, preferred ID |
| GET `/api/metrics` | Actual evaluation report, CV scores, methodology, provenance |
| POST `/api/predict` | Three classifier names and binary labels, preferred ID, disclaimer |

Example:

```sh
curl http://127.0.0.1:5001/api/predict \
  -H 'Content-Type: application/json' \
  -d '{"age":63,"sex":1,"cp":1,"trestbps":145,"chol":233,"fbs":1,"restecg":2,"thalach":150,"exang":0,"oldpeak":2.3,"slope":3,"ca":0,"thal":6}'
```

Predictions are under `predictions.knn`, `predictions.naive_bayes`, and `predictions.decision_tree`, each with `name` and `label`. Label 0 means no heart disease indicated **by the model**, not proof of health. Label 1 means heart disease indicated **by the model**, not a diagnosis. Responses contain no confidence percentages.

Missing/unknown fields, unsupported categories, booleans, non-numbers, non-finite values, out-of-range values, and fractional values for integer measurements return 400. Invalid JSON returns 400; unavailable model artifacts return 503; unknown routes return 404; unsupported methods return 405. Requests are limited to 16 KiB. The application does not persist submitted measurements.

## Verification

```sh
cd backend
.venv/bin/python -m unittest discover -s tests -v
cd ../frontend
npm run lint
npm run build
```

Tests cover health, metadata, valid inference, feature order independence, missing fields, categorical/numeric validation, malformed requests, model/report loading, incompatible feature schemas, and missing artifacts.

## Limitations and educational-use disclaimer

This small, historical Cleveland cohort is not representative of all populations. Its binary sex coding is limited. Measurements require clinical testing; this is not a symptom checker. Cross-validation estimates used for tuning have selection optimism, and the small held-out set gives uncertain performance estimates. There is no external validation, probability calibration, fairness assessment, clinical trial, or clinical safety evaluation. Maximizing recall can increase false positives. A negative classification cannot establish health, and a positive classification cannot diagnose disease. Use this application only to study pattern recognition; seek qualified clinical assessment for health concerns.

## Verification record (22 October 2026)

- Backend: all 11 unittest tests passed.
- Frontend ESLint: passed.
- Vite production build: passed; a non-blocking warning reports a 635 kB JavaScript bundle, including Recharts.
- Live HTTP smoke checks: prediction route served successfully; Vite-proxied `/api/health` returned `models_ready: true`.
- Browser visual and interaction QA was unavailable because no in-app browser connection was exposed. Responsive styling and accessibility controls are implemented but have not been visually verified.
- Joblib emitted a non-blocking macOS physical-core detection warning; training and inference completed successfully.
