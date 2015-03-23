import os
import os.path as op

import numpy as np

import matplotlib.pyplot as plt
import pandas as pd

# Experiment parameters
open_browser = True
base_path = op.dirname(op.dirname(__file__))
data_path = '/media/Paris/data/'
script_path = '/home/niccolo/Dropbox/DOCUP/scripts/python/'
results_dir = op.join(base_path, 'results')
if not op.exists(results_dir):
    os.mkdir(results_dir)

subjects = ['ak130184']

# Define contrasts
absent = dict(cond='present', values=[0])

contrasts = (
    dict(name='present',
         include=dict(cond='present', values=[0, 1]),
         exclude=[]),
    dict(name='visibility',
         include=dict(cond='response_visibilityCode', values=[1, 2]),
         exclude=[absent]),
)

contrasts = [contrasts[0]]

# Decoding preprocessing steps
preproc = dict(decim=4, crop=dict(tmin=0., tmax=0.700))

# Decoding parameters
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
scaler = StandardScaler()
svc = SVC(C=1, kernel='linear', probability=True)
clf = Pipeline([('scaler', scaler), ('svc', svc)])
decoding_params = dict(n_jobs=-1, clf=clf, predict_type='predict_proba')
