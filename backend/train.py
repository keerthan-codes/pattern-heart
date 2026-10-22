"""Reproducible, leakage-safe Cleveland experiment. Run: python train.py."""
from pathlib import Path
import hashlib
import io
import json
import platform
import urllib.request
import zipfile
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from schema import FEATURES, NAMES, NUMERIC, CATEGORICAL, MODEL_NAMES

ROOT = Path(__file__).resolve().parent
URL = 'https://archive.ics.uci.edu/static/public/45/heart+disease.zip'
CRITERION = 'Highest mean training cross-validation disease-positive recall, then F1; exact ties use fixed model/candidate order.'

def load_data():
    path = ROOT / 'data/processed.cleveland.data'
    if not path.exists():
        try:
            with urllib.request.urlopen(URL, timeout=45) as response:
                archive = zipfile.ZipFile(io.BytesIO(response.read()))
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(archive.read('processed.cleveland.data'))
        except Exception as exc:
            raise RuntimeError(f'Dataset download failed. Download {URL}, extract processed.cleveland.data into backend/data/, then rerun python train.py.') from exc
    frame = pd.read_csv(path, names=NAMES+['target'], na_values='?')
    if len(frame) != 303 or frame['target'].isna().any() or not frame['target'].isin(range(5)).all():
        raise ValueError('Expected the official 303-row Cleveland dataset with targets 0–4.')
    return frame, hashlib.sha256(path.read_bytes()).hexdigest()

def choose(cv):
    return max(range(len(cv['params'])), key=lambda i:(cv['mean_test_recall'][i],cv['mean_test_f1'][i]))

def train():
    frame, checksum = load_data()
    X, y = frame[NAMES], (frame.target > 0).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.2,stratify=y,random_state=42)
    configurations = {
        'knn':(KNeighborsClassifier(), {'classifier__n_neighbors':[3,5,7,9,11,15], 'classifier__weights':['uniform','distance']}),
        'naive_bayes':(GaussianNB(), {'classifier__var_smoothing':[1e-11,1e-9,1e-7,1e-5,.001,.1]}),
        'decision_tree':(DecisionTreeClassifier(random_state=42), {'classifier__max_depth':[2,3,4,5,None], 'classifier__min_samples_leaf':[1,5,10], 'classifier__class_weight':[None,'balanced']}),
    }
    trained, results = {}, {}
    cv = StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
    for key,(classifier,grid) in configurations.items():
        numeric = [('impute',SimpleImputer(strategy='median'))]
        if key == 'knn':
            numeric.append(('scale',StandardScaler()))
        categories = [next(f['options'] for f in FEATURES if f['name']==name) for name in CATEGORICAL]
        preprocessor = ColumnTransformer([
            ('numeric',Pipeline(numeric),NUMERIC),
            ('categorical',Pipeline([('impute',SimpleImputer(strategy='most_frequent')),('encode',OneHotEncoder(categories=[[o['value'] for o in opts] for opts in categories],handle_unknown='error',sparse_output=False))]),CATEGORICAL),
        ])
        search = GridSearchCV(Pipeline([('preprocess',preprocessor),('classifier',classifier)]),grid,scoring={'recall':'recall','f1':'f1'},refit=choose,cv=cv,n_jobs=1,error_score='raise')
        search.fit(X_train,y_train)
        trained[key] = search.best_estimator_
        i = search.best_index_
        results[key] = {'name':MODEL_NAMES[key], 'parameters':search.best_params_, 'cv_recall':float(search.cv_results_['mean_test_recall'][i]), 'cv_f1':float(search.cv_results_['mean_test_f1'][i]), 'cv_recall_std':float(search.cv_results_['std_test_recall'][i])}
    preferred = max(results,key=lambda k:(results[k]['cv_recall'],results[k]['cv_f1']))
    # Selection is complete before held-out predictions are examined.
    for key, model in trained.items():
        predictions = model.predict(X_test)
        results[key].update(accuracy=float(accuracy_score(y_test,predictions)),precision=float(precision_score(y_test,predictions,zero_division=0)),recall=float(recall_score(y_test,predictions,zero_division=0)),f1=float(f1_score(y_test,predictions,zero_division=0)),confusion_matrix=confusion_matrix(y_test,predictions,labels=[0,1]).tolist())
    report = {'preferred':preferred,'selection_criterion':CRITERION,'models':results,'dataset':{'source':URL,'sha256':checksum,'rows':len(frame),'train_rows':len(X_train),'test_rows':len(X_test),'missing_values':int(X.isna().sum().sum()),'train_class_counts':y_train.value_counts().sort_index().tolist(),'test_class_counts':y_test.value_counts().sort_index().tolist()},'methodology':{'random_state':42,'test_fraction':.2,'cv_folds':5,'positive_label':1,'confusion_matrix_axes':'Rows: actual [0, 1]; columns: predicted [0, 1].'},'versions':{'python':platform.python_version(),'sklearn':sklearn.__version__,'numpy':np.__version__,'pandas':pd.__version__}}
    output = ROOT/'models'
    output.mkdir(exist_ok=True)
    joblib.dump({'features':NAMES,'models':trained,'preferred':preferred,'dataset_sha256':checksum},output/'classifiers.joblib')
    (output/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__ == '__main__':
    train()
