# Author: Jean-Rémi King <jeanremi.king@gmail.com>
#
# BSD License

import os.path as op
import numpy as np
import matplotlib.pyplot as plt

from mne.io.meas_info import create_info
from mne.epochs import EpochsArray
from mne.decoding import GeneralizationAcrossTime


# GENERATE SYNTHETIC DATA:
# 6 orientations observed from 100 sensors across 10 time points and 198 trials.
angles = np.linspace(15, 165, 6)
n_trial = 198
n_chan = 100
n_time = 10

# Template topography for each angle
X0 = np.linspace(0, 2, np.sqrt(n_chan)) - 1
topos = list()
for a, angle in enumerate(angles):
    Xm, Ym = np.meshgrid(X0, X0)
    Xm += np.cos(np.deg2rad(2 * angle))
    Ym += np.sin(np.deg2rad(2 * angle))
    topos.append(np.exp(-((Xm ** 2) + (Ym ** 2))))

# Add noisy topo to each trial, and shuffle topo at each trial to simulate
# a different underlying generator.
snr = 30
data = np.random.randn(n_trial, n_chan, n_time) / snr
y = np.arange(n_trial) % len(angles)
trial_angles = y * 30 + 15
for t in range(n_time / 2, n_time):
    np.random.shuffle(topos)
    for trial in range(n_trial):
        topo = topos[y[trial]].flatten()
        data[trial, :, t] += topo

# Export data into mne structure
time = range(n_time)
chan_names = ['meg' + str(i) for i in range(n_chan)]
chan_types = ['grad'] * n_chan
info = create_info(chan_names, 1, chan_types)
events = np.c_[np.cumsum(np.ones(n_trial)), np.zeros(n_trial), np.zeros(n_trial)]
epochs = EpochsArray(data, info, events)

# Prepare classifier and scorer

# Scorer
def score_angle(truth, prediction):
    angle_error = truth - prediction[:, 0]
    pi = np.pi
    score = np.mean(np.abs((angle_error + pi) % (2 * pi) - pi))
    return score

# Classifier
from sklearn.svm import SVR

class SVR_angle(SVR):

    def __init__(self):
        from sklearn.svm import SVR
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        scaler_cos = StandardScaler()
        scaler_sin = StandardScaler()
        svr_cos = SVR(C=1, kernel='linear')
        svr_sin = SVR(C=1, kernel='linear')
        self.clf_cos = Pipeline([('scaler', scaler_cos), ('svr', svr_cos)])
        self.clf_sin = Pipeline([('scaler', scaler_sin), ('svr', svr_sin)])

    def fit(self, X, y):
        self.clf_cos.fit(X, np.cos(y))
        self.clf_sin.fit(X, np.sin(y))

    def predict(self, X):
        predict_cos = self.clf_cos.predict(X)
        predict_sin = self.clf_sin.predict(X)
        predict_angle = np.arctan2(predict_sin, predict_cos)
        return predict_angle

# Go from orientation space (0-180° degrees) to complex space (0 - 2 pi radians)
angle2circle = lambda angles: np.deg2rad(2 * (angles + 7.5))

# Specify our classifier
svr_angle = SVR_angle()
gat = GeneralizationAcrossTime(n_jobs=-1, clf=svr_angle)
gat.fit(epochs, y=angle2circle(trial_angles))
# Specify ou scorer
gat.score(epochs, y=angle2circle(trial_angles), scorer=score_angle)
gat.plot(vmin=0, vmax=np.pi)
