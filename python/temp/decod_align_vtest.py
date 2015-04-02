"""
This script implements two different versions of how to compute across
subjects stats.
A) It computes across subjects stats by computing stats
within each subject (across trials) and then averaging their results. This
method lacks power.
B) A real and more powerful across subjects implementation  is obtained by
computing the distributions in each subject and then compute the test
across subjects.

niccolo.pescetelli@psy.ox.ac.uk
"""
import pickle
import os.path as op
import numpy as np

import matplotlib.pyplot as plt
from pycircstat import vtest, rayleigh

from config import (subjects, results_path)
from postproc_functions import realign_angle, cart2pol, plot_circ_hist

# This is the stats across trials, but really we only need to apply a similar
# thing across subjects. This is done in orientation_stats_across.py
for s, subject in enumerate(subjects):
    print(subject)

    # define results path
    path_gat = op.join(results_path, subject, 'mvpas',
        '{}-decod_{}_{}.pickle'.format(subject, 'targetAngle', 'SVC'))

    ###### PREPROC
    # Compute angle errors by realigning the predicted categories according to
    # true category

    # read GAT data
    with open(path_gat) as f:
        gat, contrast, _, events = pickle.load(f)

    n_time, n_test_time, n_trials, n_categories = dims = np.shape(gat.y_pred_)
    # initialize variables if first subject
    if s == 0:
        p_val = list()
        M=list()
        dims_=np.array(dims)
        angle_error_across_w = np.zeros(np.append(dims_[[0,1,2]],len(subjects)))

    # realign to 4th angle category
    probas = realign_angle(gat)

    # transform 6 categories into single angle: there are two options here,
    # try it on the pilot subject, and use weighted mean if the two are
    # equivalent
    operator = np.tile(np.arange(n_categories),
                     (n_time, n_time, n_trials, 1))
    weighted_errors = (probas * operator)* np.pi / 3 - np.pi / 6
    x = np.mean(np.cos(weighted_errors),  axis=3)
    y = np.mean(np.sin(weighted_errors), axis=3)
    angle_errors_w, radius = cart2pol(x,y)

    ####### STATS WITHIN SUBJECTS
    # Apply v test and retrieve statistics that is independent of the number of
    # trials.
    p_val_v, V = vtest(angle_errors_w, 0, axis=2) # trials x time

    # apply Rayleigh test (better for of diagonal with distribution shifts, e.g.
    # on the n200, the prediction may reverse)
    p_val_ray, Z = rayleigh(angle_errors_w, axis=2) # trials x time

    # append scores
    M.append([V,Z])
    p_val.append([p_val_v, p_val_ray])

    ####### STATS ACROSS SUBJECTS
    # apply Rayleigh test (better for of diagonal with distribution shifts, e.g.
    # on the n200, the prediction may reverse)
    # angle_error_across_w[:, :, :, s] = angle_errors_w

# plot error, ie class distance
plt.subplot(221)
plt.imshow(-mean(mean(angle_error_across_w,axis=2),axis=2),interpolation='none',origin='lower')
plt.colorbar()
plt.title('weighted negative error')

# plot average Rayleigh p value
plt.subplot(222)
mean_p_vals = np.zeros((n_time, n_test_time, len(subjects)))
for s, subject in enumerate(subjects):
    mean_p_vals[:, :, s] = np.array(p_val[s][0])
plt.imshow(np.mean(mean_p_vals, axis=2), origin='lower',interpolation='none')
plt.colorbar()
plt.title('mean p-val Rayleigh')

## STATS ACROSS SUBJECTS - tentative, dubious method XXX
plt.subplot(223)
p, z = rayleigh(angle_error_across_w, axis=2, d=20)
plt.imshow(np.mean(z, axis=2), origin='lower',interpolation='none')
plt.colorbar()
plt.title('z Rayleigh (dubious method)')
