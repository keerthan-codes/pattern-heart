# Prediction consistency investigation

## Finding

The reported desktop/mobile Gaussian Naive Bayes discrepancy was not reproduced with identical API payloads and the current model bundle. A separate frontend stale-response bug was found by tracing the asynchronous submit handler: a user could submit profile A, edit inputs to profile B while A was pending, and then see A's late response displayed alongside B's inputs. The edit cleared existing results but did not invalidate the pending request. This establishes a real timing gap, but does not prove it caused the reported discrepancy.

`Predict.jsx` now increments a request version on edits, submissions, and unmount. Only the current version can publish results, errors, validation messages, or loading-state updates. Edits still clear results, and users can submit their new profile without waiting for an obsolete request. Obsolete requests may finish on the network, but their responses are ignored. Automatic scrolling and the ready announcement occur only for accepted results. No API or model changes were made.

## Form and inference audit

- All 13 features are rendered from `/api/features`, using original categorical values. Feature names: `age`, `sex`, `cp`, `trestbps`, `chol`, `fbs`, `restecg`, `thalach`, `exang`, `oldpeak`, `slope`, `ca`, `thal`.
- State starts empty. Controlled inputs display blank until entered; there are no hidden clinical defaults, reset button, stored browser profile, or device-specific values. Input edits clear prior results; submission clears prior results and validation errors without resetting clinical values.
- Required browser controls block blank fields in ordinary form submission. Numeric values are converted with `Number()` before JSON serialization, including category strings. Zero remains zero; fractional `oldpeak` values are preserved. The backend independently rejects missing, invalid, or unexpected features. There is no device-specific numeric conversion.
- The backend constructs its DataFrame in the shared schema order, irrespective of JSON key order, and calls only `predict()`. Imputation, one-hot encoding, and KNN scaling use the saved fitted pipelines. No request fits a model or preprocessing step.
- A Flask app instance loads one fixed Joblib bundle at creation and reuses it. Tests confirm one artifact load across ten predictions and matching responses from a new app instance using the same file. Different deployments or processes started before/after an artifact replacement could still use different bundles; these cannot be verified from this repository alone.
- Viewport width affects CSS only. The only JavaScript media query on Predict controls scrolling motion. Device headers are not used in inference.
- The preferred banner and classifier cards read the same response object. The banner uses `result.predictions[result.preferred]`; current artifacts select `naive_bayes`, so its label matches the Gaussian Naive Bayes card. There is no independently calculated banner prediction.
- The API client sends JSON to `VITE_API_BASE_URL` or `/api` with a 20-second timeout. Different frontend builds or origins could target different servers; the original device URLs were not supplied.

## Reproduction payloads and results

These are manually chosen valid test inputs, not additional training data. No training was run.

```json
{
  "baseline": {"age":63,"sex":1,"cp":1,"trestbps":145,"chol":233,"fbs":1,"restecg":2,"thalach":150,"exang":0,"oldpeak":2.3,"slope":3,"ca":0,"thal":6},
  "profile_b": {"age":40,"sex":0,"cp":2,"trestbps":120,"chol":180,"fbs":0,"restecg":0,"thalach":175,"exang":0,"oldpeak":0,"slope":1,"ca":0,"thal":3},
  "profile_c": {"age":67,"sex":1,"cp":4,"trestbps":160,"chol":286,"fbs":0,"restecg":2,"thalach":108,"exang":1,"oldpeak":3,"slope":2,"ca":3,"thal":7}
}
```

Each payload was submitted to `POST /api/predict` ten times via Flask's in-process test client. For each payload, all ten full parsed JSON responses and serialized response bodies matched. The automated regression uses an identical serialized request body for each of ten repetitions. This exercises validation, saved preprocessing, inference, and response serialization without a network server.

| Payload | KNN | Gaussian Naive Bayes | Decision Tree | Full responses |
| --- | --- | --- | --- | --- |
| baseline | 0 | 1 | 1 | 10/10 identical |
| profile_b | 0 | 0 | 0 | 10/10 identical |
| profile_c | 1 | 1 | 1 | 10/10 identical |

Label 0: no heart disease indicated by the model. Label 1: heart disease indicated by the model. These are educational classifications, not diagnoses.

Additional regression checks verify matching complete JSON with desktop/mobile user-agent and viewport headers, reversed feature key order, equivalent integer/float representations, and a fresh Flask app loading the same artifacts. Every classifier produces both labels across the three tested profiles.

## Verification

```sh
cd backend
.venv/bin/python -m pytest tests -v
cd ../frontend
npm run lint
npm run build
```

Results: 14 pytest tests passed (19 subtests passed); ESLint passed; production build passed. Non-blocking warnings: Joblib's existing macOS physical-core detection warning and Vite's existing JavaScript chunk-size warning.

Files changed for the investigation: `frontend/src/pages/Predict.jsx`, `backend/tests/test_api.py`, and this report. Backend implementation, model algorithms, training code, API contracts, dataset, saved models, and evaluation metrics were not changed.

Artifact SHA-256 values, verified unchanged after the investigation:

- `backend/models/classifiers.joblib`: `912b13ba94bb49f0a082e3616cf74c31398ff6f61ce90f03a8666610cd8c93b7`
- `backend/models/metrics.json`: `9579f0608fafd30708cf5a68d4f00b28515892dba8a1791d9ce2af87ce76c7fb`
- `backend/data/processed.cleveland.data`: `a74b7efa387bc9d108d7d0115d831fe9b414b29ae7124f331b622b4efa0427c8`

## Final verification still needed

Ready for final desktop/mobile verification. No in-app browser connection was available, so browser interaction testing of the timing fix remains pending. Use network throttling: submit profile A, edit to B before A completes, and verify A never reappears; submit B and verify only B's response appears. Also check overlapping requests in reverse completion order and a failed obsolete request. Existing results scrolling, reduced-motion behavior, labels, and banner should remain unchanged.

To attribute the original discrepancy, capture both exact submitted JSON payloads and responses, confirm the frontend/backend URLs, and compare model bundle hashes on the serving instances. The original 13 values and serving URLs were requested but have not been supplied. Identical API payloads with the same loaded artifacts were deterministic in all tests; this does not establish that the two reported device sessions actually sent identical payloads to identical artifacts.
