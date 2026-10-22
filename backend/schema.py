"""Original Cleveland feature encodings; shared by training and inference."""
def number(name, label, group, unit, low, high, description, integer=True):
    return dict(name=name, label=label, group=group, unit=unit, min=low, max=high, description=description, integer=integer, type='number')

def category(name, label, group, options, description):
    return dict(name=name, label=label, group=group, options=[dict(value=v, label=l) for v,l in options], description=description, type='category')

FEATURES = [
    number('age','Age','Basic measurements','years',18,120,'Age in completed years.'),
    category('sex','Sex recorded in dataset','Basic measurements',[(0,'Female'),(1,'Male')],'Original binary coding in the historical dataset.'),
    category('cp','Chest pain type','Basic measurements',[(1,'Typical angina'),(2,'Atypical angina'),(3,'Non-anginal pain'),(4,'Asymptomatic')],'Use the recorded clinical assessment.'),
    number('trestbps','Resting blood pressure','Basic measurements','mm Hg',50,300,'Blood pressure on hospital admission.'),
    number('chol','Serum cholesterol','Laboratory & resting ECG','mg/dL',50,700,'Serum cholesterol measurement.'),
    category('fbs','Fasting blood sugar','Laboratory & resting ECG',[(0,'120 mg/dL or below'),(1,'Above 120 mg/dL')],'Whether fasting blood sugar exceeds 120 mg/dL.'),
    category('restecg','Resting ECG','Laboratory & resting ECG',[(0,'Normal'),(1,'ST–T wave abnormality'),(2,'Left ventricular hypertrophy')],'Recorded resting electrocardiographic result.'),
    number('thalach','Maximum heart rate','Exercise & imaging','bpm',30,250,'Maximum heart rate achieved during exercise.'),
    category('exang','Exercise-induced angina','Exercise & imaging',[(0,'No'),(1,'Yes')],'Whether exercise induced angina.'),
    number('oldpeak','ST depression','Exercise & imaging','ST units',0,10,'Exercise-induced ST depression relative to rest.',False),
    category('slope','Peak exercise ST slope','Exercise & imaging',[(1,'Upsloping'),(2,'Flat'),(3,'Downsloping')],'Slope of the peak exercise ST segment.'),
    category('ca','Major vessels','Exercise & imaging',[(0,'0'),(1,'1'),(2,'2'),(3,'3')],'Number of major vessels colored by fluoroscopy.'),
    category('thal','Thallium test result','Exercise & imaging',[(3,'Normal'),(6,'Fixed defect'),(7,'Reversible defect')],'Original thallium stress test encoding.'),
]
NAMES = [f['name'] for f in FEATURES]
NUMERIC = [f['name'] for f in FEATURES if f['type'] == 'number']
CATEGORICAL = [f['name'] for f in FEATURES if f['type'] == 'category']
MODEL_NAMES = {'knn':'K-Nearest Neighbours','naive_bayes':'Gaussian Naive Bayes','decision_tree':'Decision Tree'}
