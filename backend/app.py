"""Local educational API. Never load untrusted Joblib artifacts."""
import json
import math
import os
from pathlib import Path
import joblib
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS
from werkzeug.exceptions import HTTPException
from schema import FEATURES, NAMES, MODEL_NAMES

ROOT = Path(__file__).resolve().parent

def load_artifacts(directory):
    bundle = joblib.load(directory/'classifiers.joblib')
    report = json.loads((directory/'metrics.json').read_text())
    if bundle['features'] != NAMES or set(bundle['models']) != set(MODEL_NAMES):
        raise ValueError('Model feature schema or classifier set does not match the API.')
    if bundle['preferred'] != report['preferred'] or bundle['dataset_sha256'] != report['dataset']['sha256']:
        raise ValueError('Model bundle and evaluation report do not match.')
    for model in bundle['models'].values():
        if list(model.feature_names_in_) != NAMES:
            raise ValueError('Trained feature order does not match inference schema.')
    return bundle, report

def create_app(model_dir=None):
    app = Flask(__name__)
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024
    CORS(app, resources={r'/api/*':{'origins':os.getenv('CORS_ORIGINS','http://localhost:5173,http://127.0.0.1:5173').split(',')}})
    try:
        bundle, report = load_artifacts(Path(model_dir) if model_dir else ROOT/'models')
        artifact_error = None
    except (OSError, ValueError, KeyError, EOFError) as exc:
        bundle, report, artifact_error = None, None, str(exc)
        app.logger.error('Artifacts unavailable: %s. Run python train.py.', artifact_error)

    @app.errorhandler(HTTPException)
    def http_error(exc):
        return jsonify(error=exc.description), exc.code

    @app.get('/api/health')
    def health():
        return jsonify(status='ok' if bundle else 'degraded',models_ready=bundle is not None), 200 if bundle else 503

    @app.get('/api/features')
    def features():
        return jsonify(features=FEATURES,feature_order=NAMES)

    @app.before_request
    def require_models():
        if request.path in ['/api/models','/api/metrics','/api/predict'] and bundle is None:
            return jsonify(error='Model artifacts unavailable. Run python train.py in backend/.'),503

    @app.get('/api/models')
    def models():
        return jsonify(models=[dict(id=k,name=v,preferred=k==bundle['preferred']) for k,v in MODEL_NAMES.items()],preferred=bundle['preferred'])

    @app.get('/api/metrics')
    def metrics():
        return jsonify(report)

    @app.post('/api/predict')
    def predict():
        data = request.get_json(silent=True)
        if not isinstance(data,dict):
            return jsonify(error='Provide a JSON object containing the 13 clinical features.'),400
        errors = {}
        for key in set(data)-set(NAMES):
            errors[key] = 'Unknown feature.'
        for f in FEATURES:
            key = f['name']
            if key not in data:
                errors[key] = 'This field is required.'
                continue
            value = data[key]
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
                errors[key] = 'Enter a finite number.'
            elif f['type']=='category' and value not in [o['value'] for o in f['options']]:
                errors[key] = 'Select an accepted category.'
            elif f['type']=='number' and (not f['min'] <= value <= f['max'] or (f['integer'] and value != int(value))):
                errors[key] = f"Enter {'a whole number' if f['integer'] else 'a number'} between {f['min']} and {f['max']}."
        if errors:
            return jsonify(error='Please correct the clinical inputs.',fields=errors),400
        frame = pd.DataFrame([[data[name] for name in NAMES]],columns=NAMES)
        predictions = {key:{'name':MODEL_NAMES[key],'label':int(model.predict(frame)[0])} for key,model in bundle['models'].items()}
        return jsonify(predictions=predictions,preferred=bundle['preferred'],disclaimer='Educational model output only. This is not a medical diagnosis; a negative classification does not establish that a person is healthy.')

    return app

if __name__ == '__main__':
    create_app().run(host='127.0.0.1',port=5001,debug=False)
