"""
This script retrieve classifier results from SVC and SVR on orientation
and plot the tuning curves, ie categories x time along the diagonal
classification.

niccolo.pescetelli@psy.ox.ac.uk
2015
"""

import pickle
import os.path as op

import numpy as np
import matplotlib.pyplot as plt

from config import (
                    subjects,
                    data_path,
                    inputTypes,
                    clf_types,
)
from myfunctions import (
                    realign_angle,
                    recombine_svr_prediction,
                    cart2pol,
                    plot_circ_hist
)

# -----------------SVR----------------------------------------------------------
# input type is ERF (for now)
inputType=inputTypes[0]

# classifier type is SVR
clf_type = clf_types[1]

# input type is target orientation sine and cosine (for now...)
contrast = clf_type['contrasts'][0:2]

# initialize variables
res = 20
trial_proportion = np.zeros([len(subjects),29,29,res])

# loop across subjects
for s, subject in enumerate(subjects):
    print(subject)
    # load individual data
    # XXX contrast[0]['include']['cond'] should be changed into contrast[0]['name']
    path_x = op.join(data_path, subject, 'mvpas',
        '{}-decod_{}_{}.pickle'.format(subject, contrast[0]['include']['cond'], 'SVR'))

    path_y = op.join(data_path, subject, 'mvpas',
        '{}-decod_{}_{}.pickle'.format(subject, contrast[1]['include']['cond'], 'SVR'))

    ###### PREPROC
    # recombine cosine and sine predictions
    predAngle, _, trueX, trial_prop = recombine_svr_prediction(path_x, path_y,res)

    trial_proportion[s,:,:,:] = trial_prop

# plot average tuning curve across subjects on the diagonal
trial_prop_diag = np.array([trial_proportion[:,t,t,:] for t in np.arange(trial_proportion.shape[1])])
trial_prop_diag = trial_prop_diag.transpose([1,2,0])

imshow(trial_prop_diag.mean(axis=0))
