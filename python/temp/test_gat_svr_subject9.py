import os.path as op

import pickle
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

import mne
from mne.decoding import GeneralizationAcrossTime

from utils import get_data, resample_epochs, decim

from config import (
    open_browser,
    data_path,
    results_path,
    results_dir,
    subjects,
    inputTypes,
    clf_types,
    preproc,
    decoding_params
)

subject = subjects[9]

# PREPROC
preproc = dict(decim=1, crop=dict(tmin=-.100, tmax=0.500))

meg_fname = op.join(data_path, subject, 'preprocessed', subject + '_preprocessed')
bhv_fname = op.join(data_path, subject, 'behavior', subject + '_fixed.mat')
epochs, events = get_data(meg_fname, bhv_fname)

if 'decim' in preproc.keys():
    epochs = decim(epochs, preproc['decim'])
if 'crop' in preproc.keys():
    epochs.crop(preproc['crop']['tmin'],
                preproc['crop']['tmax'])

# DEFINE CONTRAST
cosx = np.cos(2*np.deg2rad([x+7.5 for x in [15, 45, 75, 105, 135, 165]]))
sinx = np.sin(2*np.deg2rad([x+7.5 for x in [15, 45, 75, 105, 135, 165]]))


absent = dict(cond='present', values=[0])
contrasts = (
    dict(name='targetAngle_cos', # values likely to be changed
         include=dict(cond='orientation_target_cos',
                        values=cosx),
         exclude=[absent]),
    dict(name='targetAngle_sin',
         include=dict(cond='orientation_target_sin',
                        values=sinx),
          exclude=[absent])
    )

# RUN GAT
gats = list()
for contrast in contrasts:
    print(contrast['name'])
    # Find excluded trials
    exclude = np.any([events[x['cond']]==ii
                        for x in contrast['exclude']
                            for ii in x['values']],
                    axis=0)

    # Select condition
    include = list()
    cond_name = contrast['include']['cond']
    for value in contrast['include']['values']:
        # Find included trials
        include.append(events[cond_name]==value)
    sel = np.any(include,axis=0) * (exclude==False)
    sel = np.where(sel)[0]

    y = np.array(events[cond_name].tolist())

    # Apply contrast
    decoding_parameters = decoding_params[1]['params']
    gat = GeneralizationAcrossTime(**decoding_parameters)
    gat.fit(epochs[sel], y=y[sel])
    gat.score(epochs[sel], y=y[sel])

    gats.append(gat)


# ALIGN ANGLES
def cart2pol(x, y):
    theta = np.arctan2(y, x)
    radius = np.sqrt(x ** 2 + y ** 2)
    return(theta, radius)

gatx = gats[0]
gaty = gats[1]



pi = np.pi

# get true angle
true_x = gatx.y_train_
true_y = gaty.y_train_
true_angle, _ = cart2pol(true_x, true_y)

# get x and y regressors (cos and sin)
predict_x = np.squeeze(gatx.y_pred_)
predict_y = np.squeeze(gaty.y_pred_)
predict_angles, _ = cart2pol(predict_x, predict_y)

# compute angle error
n_T = len(gatx.train_times['times_'])
n_t = len(gatx.test_times_['times_'][0])
true_angles = np.tile(true_angle, (n_T, n_t, 1))
angle_errors = (predict_angles - true_angles + pi) % (2 * pi) - pi
#angle_errors = predict_angles - true_angles
plt.imshow(np.mean(angle_errors ** 2, axis=2),origin='lower',interpolation='none')

tile = lambda x: np.tile(x, (n_T, n_t, 1))
plt.matshow(np.mean((predict_x - tile(true_x)) ** 2, axis=2))
