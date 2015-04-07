""" This script retrieve orientation SVR results and create a unified GAT
matrix that has in the key scorer the appropriate scoring system for
orientations, namely angle_errors
"""

import pickle
import numpy as np
import os.path as op
import scipy.io as sio

from postproc_functions import recombine_svr_prediction
from config import (results_path,
                    subjects,
                    inputTypes,
                    clf_types
                    )

# SVR only
clf_type = clf_types[1]

# loop across contrast of interest and subjects
for s, subjects in subjects:
    print(subject,s)
    for c in range(1): # use range(2) to compute also probes
        # define results path
        contrast = clf_type['contrasts'][c*2]
        path_x = op.join(results_path, subject, 'mvpas',
            '{}-decod_{}_{}.pickle'.format(subject, contrast['name'], 'SVR'))

        contrast = clf_type['contrasts'][c*2+1]
        path_y = op.join(results_path, subject, 'mvpas',
            '{}-decod_{}_{}.pickle'.format(subject, contrast['name'], 'SVR'))

        # read GAT data
        with open(path_x) as f:
            gatx, contrast, sel, events = pickle.load(f)

        # read GAT data
        with open(path_y) as f:
            gaty, contrast, sel, events = pickle.load(f)

        n_time, n_test_time, n_trials, n_categories = np.shape(gaty.y_pred_)

        # recombine cos and sin predictions into one predicted angle
        predict_angle, true_angle, angle_error = recombine_svr_prediction(gatx, gaty)

        #### create final gat object
        gat = gatx

        # true y
        gat.y_true_ = None
        gat.y_true_ = true_angle
        gat.y_train_ = gat.y_true_

        # prediction
        gat.y_pred_ = None
        gat.y_pred_ = predict_angle

        # score
        gat.scores_ = None
        gat.scores_ = angle_error

        # Save contrast
        pkl_fname = op.join(results_path, subject, 'mvpas',
            '{}-decod_{}_{}.pickle'.format(subject, contrast['name'], 'SVR'))

        # Save classifier results
        with open(pkl_fname, 'wb') as f:
            pickle.dump([gat, contrast, sel, events], f)
