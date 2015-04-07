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

from config import (subjects, results_path, clf_types)
from toolbox.utils import (plot_eb, fill_betweenx_discontinuous)
from postproc_functions import (
                                realign_angle,
                                cart2pol,
                                recombine_svr_prediction,
                                plot_circ_hist
)


for clf_type in clf_types:
    Z, V, p_values_v, p_values_z, angle_errors = list(), list(), list(), list(), list()
    for s, subject in enumerate(subjects):
        print(subject)

        if clf_type['name']=='SVC':
            # define results path
            path_gat = op.join(results_path, subject, 'mvpas',
                '{}-decod_{}_{}.pickle'.format(subject, 'targetAngle', 'SVC'))

            ###### PREPROC
            # Compute angle errors by realigning the predicted categories according to
            # true category

            # read GAT data
            with open(path_gat) as f:
                gatx, contrast, sel, events = pickle.load(f)

            n_time, n_test_time, n_trials, n_categories = np.shape(gatx.y_pred_)

            # realign to 0th angle category
            probas = realign_angle(gatx)

            # transform 6 categories into single angle: there are two options here,
            # try it on the pilot subject, and use weighted mean if the two are
            # equivalent
            weighted_mean = True
            if weighted_mean:
                # multiply category ind (1, 2, 3, ) by angle, remove pi to be between
                # -pi and pi and remove pi/6 to center on 0
                operator = np.arange(n_categories) * np.pi / 3 + np.pi / 6
                operator = np.tile(operator, (n_time, n_time, n_trials, 1))
                # average angles: XXX THEORETICALLY INCORRECT
                angle_error = np.mean(probas * operator, axis=3)
                # XXX NB we should average in complex space but somehow it doesnt work
                #xy = np.mean(probas * (np.cos(operator) + 1j *sin(operator)), axis=3)
                #angle_error, _ = cart2pol(real(xy), imag(xy))
            else:
                angle_error = np.argmax(probas, axis=3) * np.pi / 3 - np.pi / 6
        elif clf_type['name']=='SVR':
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

            n_time, n_test_time, n_trials, n_categories = np.shape(gaty.y_pred_)

            # recombine cos and sin predictions into one predicted angle
            predict_angle, true_angle, angle_error = recombine_svr_prediction(gatx, gaty)


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
        for v, vis in enumerate(range(1, 5)):
            indx = np.array(events['response_visibilityCode'][sel]==vis)
            angle_errors_vis[s,:,:,v] = np.mean(angle_error[:,:,indx],axis=2)


    # A)
    # plot error, ie class distance
    # plt.subplot(221)
    # plt.imshow(mean(angle_errors,axis=0),interpolation='none',origin='lower')
    # plt.colorbar()
    # plt.title('weighted negative error')

    # B)
    # define X
    baseline = np.pi / 6.

    # perform cluster test on main effect (difference from chance)
    cluster_test_main(gatx, angle_errors, baseline, ylabel='Angle error')

    """ Divide by visibility"""
    # -- Divide by visibility
    for vis in arange(4):
        # define A (non corrected measure)
        A = np.array(angle_errors_vis[0:19,:,:,vis])

        # bound to common scale
        lims = [np.min(A), np.max(A)]

        cluster_test_main(gatx, A, baseline, ylabel='Angle error', title=vis,
                            lims=lims)
