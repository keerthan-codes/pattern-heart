import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import joblib
import pandas as pd
from app import create_app, load_artifacts, ROOT
from schema import NAMES, MODEL_NAMES

PROFILE = dict(age=63,sex=1,cp=1,trestbps=145,chol=233,fbs=1,restecg=2,thalach=150,exang=0,oldpeak=2.3,slope=3,ca=0,thal=6)

class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = create_app().test_client()

    def test_health(self):
        response = self.client.get('/api/health')
        self.assertEqual(response.status_code,200)
        self.assertTrue(response.json['models_ready'])

    def test_feature_metadata(self):
        data = self.client.get('/api/features').json
        self.assertEqual(data['feature_order'],NAMES)
        self.assertEqual(len(data['features']),13)
        self.assertEqual([o['value'] for o in data['features'][-1]['options']],[3,6,7])

    def test_valid_prediction_and_feature_order(self):
        response = self.client.post('/api/predict',json=dict(reversed(list(PROFILE.items()))))
        self.assertEqual(response.status_code,200)
        self.assertEqual(set(response.json['predictions']),set(MODEL_NAMES))
        bundle, _ = load_artifacts(ROOT/'models')
        for key,model in bundle['models'].items():
            expected = int(model.predict(pd.DataFrame([PROFILE],columns=NAMES))[0])
            self.assertEqual(response.json['predictions'][key]['label'],expected)

    def test_repeated_predictions_are_identical(self):
        profiles = [
            PROFILE,
            dict(age=40,sex=0,cp=2,trestbps=120,chol=180,fbs=0,restecg=0,thalach=175,exang=0,oldpeak=0,slope=1,ca=0,thal=3),
            dict(age=67,sex=1,cp=4,trestbps=160,chol=286,fbs=0,restecg=2,thalach=108,exang=1,oldpeak=3,slope=2,ca=3,thal=7),
        ]
        outputs = []
        for profile in profiles:
            with self.subTest(profile=profile):
                payload = json.dumps(profile)
                responses = [self.client.post('/api/predict',data=payload,content_type='application/json') for _ in range(10)]
                for response in responses:
                    self.assertEqual(response.status_code,200)
                    self.assertEqual(response.json,responses[0].json)
                    self.assertEqual(response.data,responses[0].data)
                outputs.append(responses[0].json['predictions'])
        # Detect a broken endpoint that always returns a constant prediction.
        for key in MODEL_NAMES:
            self.assertEqual({output[key]['label'] for output in outputs},{0,1})

    def test_device_headers_and_numeric_representation_do_not_change_predictions(self):
        expected = self.client.post('/api/predict',json=PROFILE).json
        devices = [
            {'User-Agent':'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)', 'Sec-CH-Viewport-Width':'1440', 'Sec-CH-UA-Mobile':'?0'},
            {'User-Agent':'Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X)', 'Sec-CH-Viewport-Width':'390', 'Sec-CH-UA-Mobile':'?1'},
        ]
        # JSON numeric values and object key order carry the same feature meaning.
        equivalent = {key:float(value) for key,value in reversed(list(PROFILE.items()))}
        for headers in devices:
            for profile in [PROFILE,equivalent]:
                with self.subTest(headers=headers,profile=profile):
                    response = self.client.post('/api/predict',json=profile,headers=headers)
                    self.assertEqual(response.status_code,200)
                    self.assertEqual(response.json,expected)
        self.assertEqual(expected['preferred'],'naive_bayes')
        self.assertEqual(expected['predictions'][expected['preferred']],expected['predictions']['naive_bayes'])

    def test_artifacts_loaded_once_and_consistent_across_app_instances(self):
        expected = self.client.post('/api/predict',json=PROFILE).json
        with patch('app.joblib.load',wraps=joblib.load) as load:
            client = create_app().test_client()
            for _ in range(10):
                response = client.post('/api/predict',json=PROFILE)
                self.assertEqual(response.status_code,200)
                self.assertEqual(response.json,expected)
            self.assertEqual(load.call_count,1)

    def test_missing_fields(self):
        response = self.client.post('/api/predict',json={})
        self.assertEqual(response.status_code,400)
        self.assertEqual(set(response.json['fields']),set(NAMES))

    def test_invalid_categories(self):
        for field,value in [('thal',0),('cp',0),('ca',4),('sex',2)]:
            with self.subTest(field=field):
                response = self.client.post('/api/predict',json={**PROFILE,field:value})
                self.assertEqual(response.status_code,400)
                self.assertIn(field,response.json['fields'])

    def test_invalid_numbers(self):
        for value in [None,True,'63',-1,121,63.5,float('nan'),float('inf')]:
            with self.subTest(value=value):
                response = self.client.post('/api/predict',json={**PROFILE,'age':value})
                self.assertEqual(response.status_code,400)
                self.assertIn('age',response.json['fields'])

    def test_extra_fields_and_invalid_body(self):
        self.assertEqual(self.client.post('/api/predict',json={**PROFILE,'target':1}).status_code,400)
        for body in [[],None,'text']:
            self.assertEqual(self.client.post('/api/predict',json=body).status_code,400)
        self.assertEqual(self.client.post('/api/predict',data='{',content_type='application/json').status_code,400)

    def test_artifact_loading(self):
        bundle, report = load_artifacts(ROOT/'models')
        self.assertEqual(bundle['features'],NAMES)
        self.assertEqual(report['dataset']['test_rows'],61)
        self.assertEqual(bundle['preferred'],report['preferred'])
        for model in report['models'].values():
            self.assertEqual(sum(map(sum,model['confusion_matrix'])),61)
        winner = max(report['models'],key=lambda k:(report['models'][k]['cv_recall'],report['models'][k]['cv_f1']))
        self.assertEqual(winner,report['preferred'])

    def test_unavailable_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            client = create_app(directory).test_client()
            self.assertEqual(client.get('/api/health').status_code,503)
            self.assertEqual(client.get('/api/metrics').status_code,503)
            self.assertEqual(client.post('/api/predict',json=PROFILE).status_code,503)
            self.assertEqual(client.get('/api/features').status_code,200)

    def test_incompatible_schema_rejected(self):
        bundle, report = load_artifacts(ROOT/'models')
        bundle['features'] = list(reversed(NAMES))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            joblib.dump(bundle,path/'classifiers.joblib')
            (path/'metrics.json').write_text(json.dumps(report))
            with self.assertRaises(ValueError):
                load_artifacts(path)

    def test_models_metrics_and_json_errors(self):
        self.assertEqual(self.client.get('/api/models').status_code,200)
        self.assertEqual(self.client.get('/api/metrics').status_code,200)
        self.assertEqual(self.client.get('/api/predict').status_code,405)
        self.assertIsNotNone(self.client.get('/api/missing').json)

if __name__ == '__main__':
    unittest.main()
