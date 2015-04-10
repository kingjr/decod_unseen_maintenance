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
from mne.stats import spatio_temporal_cluster_1samp_test

from config import (subjects, results_path)
from toolbox.utils import plot_eb, fill_betweenx_discontinuous
from postproc_functions import (realign_angle,
                                cart2pol,
                                recombine_svr_prediction,
                                plot_circ_hist
)

# ------------------------------------------------------------------------------
# ------------------------------------SVC---------------------------------------
# This is the stats across trials, but really we only need to apply a similar
# thing across subjects. This is done in orientation_stats_across.py
Z, V, p_values_v, p_values_z, angle_errors, angle_errors_vis = list(), list(), list(), list(), list(), list()
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
    dims_=np.array(dims)

    # realign to 0th angle category
    probas = realign_angle(gat)

    # transform 6 categories into single angle: there are two options here,
    # try it on the pilot subject, and use weighted mean if the two are
    # equivalent
    weighted_mean = True
    if weighted_mean:
        # multiply category ind (1, 2, 3, ) by angle, remove pi to be between
        # -pi and pi and remove pi/6 to center on 0
        operator = np.arange(n_categories) * np.pi / 3 + np.pi / 6
        operator = np.tile(operator, (n_time, n_time, n_trials, 1))
        # average angles: # XXX THEORETICALLY INCORRECT
        angle_error = np.mean(probas * operator, axis=3)
        # XXX NB we should average in complex space but somehow it doesnt work
        #xy = np.mean(probas * (np.cos(operator) + 1j *sin(operator)), axis=3)
        #angle_error, _ = cart2pol(real(xy), imag(xy))
    else:
        angle_error = np.argmax(probas, axis=3) * np.pi / 3 - np.pi / 6

    ####### STATS WITHIN SUBJECTS
    # Apply v test and retrieve statistics that is independent of the number of
    # trials.
    p_val_v, v = vtest(angle_error, 0, axis=2) # trials x time

    # apply Rayleigh test (better for of diagonal with distribution shifts, e.g.
    # on the n200, the prediction may reverse)
    p_val_z, z = rayleigh(angle_error, axis=2) # trials x time

    # append scores
    V.append(v)
    Z.append(z)
    p_values_v.append(p_val_v)
    p_values_z.append(p_val_z)
    angle_errors.append(np.mean(angle_error, axis=2))

    # divide by visibility
    for vis in [range(4) + 1]:
        angle_errors_vis.append(np.mean(
                angle_error[:,:,events['response_visibilityCode']==vis],axis=2))


# A)
# plot error, ie class distance
plt.subplot(221)
plt.imshow(mean(angle_errors,axis=0),interpolation='none',origin='lower')
plt.colorbar()
plt.title('weighted negative error')

# B)
# define X
X = np.array(angle_errors) - np.pi / 6.

# stats across subjects
alpha = 0.05
n_permutations = 2 ** 11
threshold = dict(start=1., step=.2)

# ------ Run stats
T_obs_, clusters, p_values, _ = spatio_temporal_cluster_1samp_test(
                                       X,
                                       out_type='mask',
                                       n_permutations=n_permutations,
                                       connectivity=None,
                                       threshold=threshold,
                                       n_jobs=-1)

# ------ combine clusters and retrieve min p_values for each feature
p_values = np.min(np.logical_not(clusters) +
                  [clusters[c] * p for c, p in enumerate(p_values)],
                  axis=0)
x, y = np.meshgrid(gat.train_times['times_'],
                   gat.test_times_['times_'][0],
                   copy=False, indexing='xy')


# PLOT
# ------ Plot GAT
gat.scores_ = np.mean(angle_errors, axis=0)
fig = gat.plot(vmin=np.min(gat.scores_), vmax=np.max(gat.scores_),
               show=False)
ax = fig.axes[0]
ax.contour(x, y, p_values < alpha, colors='black', levels=[0])
plt.show()

# ------ Plot Decoding
times = gat.train_times['times_']
scores_diag = np.transpose([np.array(angle_errors)[:, t, t] for t in range(len(times))])
fig, ax = plt.subplots(1)
plot_eb(times, np.mean(scores_diag, axis=0),
        np.std(scores_diag, axis=0) / np.sqrt(scores_diag.shape[0]),
        color='blue', ax=ax)
ymin, ymax = ax.get_ylim()
sig_times = times[np.where(np.diag(p_values) < alpha)[0]]
sfreq = (times[1] - times[0]) / 1000
fill_betweenx_discontinuous(ax, ymin, ymax, sig_times, freq=sfreq,
                            color='gray', alpha=.3)
ax.axhline(np.pi/6, color='k', linestyle='--', label="Chance level")
ax.set_xlabel('Time (s)')
ax.set_ylabel('Angle error')
plt.show()

# ------------------------------------------------------------------------------
# -------------------------------SVR--------------------------------------------
Z, V, p_values_v, p_values_z, angle_errors, angle_errors_vis = list(), list(), list(), list(), list(), list()
for s, subject in enumerate(subjects):
    print(subject)

    # define results path
    path_x = op.join(results_path, subject, 'mvpas',
        '{}-decod_{}_{}.pickle'.format(subject, 'targetAngle_cos', 'SVR'))

    path_y = op.join(results_path, subject, 'mvpas',
        '{}-decod_{}_{}.pickle'.format(subject, 'targetAngle_sin', 'SVR'))

    # read GAT data
    with open(path_x) as f:
        gatx, contrast, sel, events = pickle.load(f)

    # read GAT data
    with open(path_y) as f:
        gaty, contrast, sel, events = pickle.load(f)

    n_time, n_test_time, n_trials, n_categories = dims = np.shape(gaty.y_pred_)
    # initialize variables if first subject
    dims_=np.array(dims)

    # recombine cos and sin predictions into one predicted angle
    predAngle, trueX, trial_prop = recombine_svr_prediction(gatx,
                                                            gaty, res = 30)

    # squeeze
    predAngle=predAngle.squeeze()

    # compute true angle
    true_angle = np.arccos(trueX) * 2

    # compute prediction error
    angle_error = ((predAngle - true_angle) / 2) % np.pi

    ####### STATS WITHIN SUBJECTS
    # Apply v test and retrieve statistics that is independent of the number of
    # trials.
    p_val_v, v = vtest(angle_error, 0, axis=2) # trials x time

    # apply Rayleigh test (better for of diagonal with distribution shifts, e.g.
    # on the n200, the prediction may reverse)
    p_val_z, z = rayleigh(angle_error, axis=2) # trials x time

    # append scores
    V.append(v)
    Z.append(z)
    p_values_v.append(p_val_v)
    p_values_z.append(p_val_z)
    angle_errors.append(np.mean(angle_error, axis=2))

    # divide by visibility
    if s ==0:
        angle_errors_vis = np.zeros([len(subjects), n_time, n_test_time, 4])
    for vis in arange(4):
        angle_errors_vis[s,:,:,vis] = np.mean(angle_error[:,:,events['response_visibilityCode'][sel]==vis+1],axis=2)


# define X
angle_errors = np.array(angle_errors)
X = angle_errors - np.pi /2.
# use last loaded gat object as a template
gat = gaty

# stats across subjects
alpha = 0.05
n_permutations = 2 ** 11
threshold = dict(start=1., step=.2)

# ------ Run stats
T_obs_, clusters, p_values, _ = spatio_temporal_cluster_1samp_test(
                                       X,
                                       out_type='mask',
                                       n_permutations=n_permutations,
                                       connectivity=None,
                                       threshold=threshold,
                                       n_jobs=-1)

# ------ combine clusters and retrieve min p_values for each feature
p_values = np.min(np.logical_not(clusters) +
                  [clusters[c] * p for c, p in enumerate(p_values)],
                  axis=0)
x, y = np.meshgrid(gat.train_times['times_'],
                   gat.test_times_['times_'][0],
                   copy=False, indexing='xy')


# PLOT
# ------ Plot GAT
gat.scores_ = np.mean(angle_errors, axis=0)
fig = gat.plot(vmin=np.min(gat.scores_), vmax=np.max(gat.scores_),
               show=False)
ax = fig.axes[0]
ax.contour(x, y, p_values < alpha, colors='black', levels=[0])
ax.set_xlabel('Test time')
ax.set_ylabel('Train time')
plt.show()

# ------ Plot Decoding
times = gat.train_times['times_']
scores_diag = np.transpose([np.array(angle_errors)[:, t, t] for t in range(len(times))])
fig, ax = plt.subplots(1)
plot_eb(times, np.mean(scores_diag, axis=0),
        np.std(scores_diag, axis=0) / np.sqrt(scores_diag.shape[0]),
        color='blue', ax=ax)
ymin, ymax = ax.get_ylim()
sig_times = times[np.where(np.diag(p_values) < alpha)[0]]
sfreq = (times[1] - times[0]) / 1000
fill_betweenx_discontinuous(ax, ymin, ymax, sig_times, freq=sfreq,
                            color='gray', alpha=.3)
ax.axhline(np.pi/6, color='k', linestyle='--', label="Chance level")
ax.set_xlabel('Time (s)')
ax.set_ylabel('Angle error (radians)')
plt.show()


""" Divide by visibility"""
# -- Divide by visibility
for vis in arange(4):
    # define X
    X = angle_errors_vis[:,:,:,vis] - np.pi /2.

    # ------ Run stats
    T_obs_, clusters, p_values, _ = spatio_temporal_cluster_1samp_test(
                                           X,
                                           out_type='mask',
                                           n_permutations=n_permutations,
                                           connectivity=None,
                                           threshold=threshold,
                                           n_jobs=-1)

    # ------ combine clusters and retrieve min p_values for each feature
    p_values = np.min(np.logical_not(clusters) +
                      [clusters[c] * p for c, p in enumerate(p_values)],
                      axis=0)
    x, y = np.meshgrid(gat.train_times['times_'],
                       gat.test_times_['times_'][0],
                       copy=False, indexing='xy')


    # PLOT
    # ------ Plot GAT
    gat.scores_ = np.mean(angle_errors_vis[:,:,:,vis], axis=0)
    gat.plot(vmin=np.min(gat.scores_), vmax=np.max(gat.scores_),
                   show=False, title=vis)
    plt.contour(x, y, p_values < alpha, colors='black', levels=[0])
    plt.xlabel('Test time')
    plt.ylabel('Train time')
    plt.show()

    # ------ Plot Decoding
    plt.figure()
    times = gat.train_times['times_']
    scores_diag = np.transpose([angle_errors_vis[:, t, t,vis] for t in range(len(times))])
    ax = plt.gca()
    plot_eb(times, np.mean(scores_diag, axis=0),
            np.std(scores_diag, axis=0) / np.sqrt(scores_diag.shape[0]),
            color='blue', ax=ax)
    ymin, ymax = ax.get_ylim()
    sig_times = times[np.where(np.diag(p_values) < alpha)[0]]
    sfreq = (times[1] - times[0]) / 1000
    fill_betweenx_discontinuous(ax, ymin, ymax, sig_times, freq=sfreq,
                                color='gray', alpha=.3)
    ax.axhline(np.pi/6, color='k', linestyle='--', label="Chance level")
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Angle error (radians)')
    plt.show()
