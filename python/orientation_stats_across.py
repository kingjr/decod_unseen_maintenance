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
from postproc_functions import (realign_angle,
                                cart2pol,
                                recombine_svr_prediction,
                                compute_error_svc,
                                compute_error_svr,
                                plot_circ_hist,
                                cluster_test_main)


for clf_type in clf_types:
    #Z, V, p_values_v, p_values_z = list(), list(), list(), list()
    angle_errors, angle_errors_vis = list(), list() * 10
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
                gat, contrast, sel, events = pickle.load(f)

            n_time, n_test_time, n_trials, n_categories = np.shape(gat.y_pred_)

            # get angle error from probas
            angle_error = compute_error_svc(gat)

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
            predict_angle, true_angle = recombine_svr_prediction(gatx, gaty)

            # compute angle error
            angle_error = compute_error_svr(predict_angle, true_angle)

            # Keep gat for later plotting purposes
            gat = gatx


        ####### STATS WITHIN SUBJECTS
        # Apply v test and retrieve statistics that is independent of the number of
        # trials.
        # p_val_v, v = vtest(angle_error, 0, axis=2) # trials x time
        #
        # # apply Rayleigh test (better for of diagonal with distribution shifts, e.g.
        # # on the n200, the prediction may reverse)
        # p_val_z, z = rayleigh(angle_error, axis=2) # trials x time
        #
        # # append scores
        # V.append(v)
        # Z.append(z)
        # p_values_v.append(p_val_v)
        # p_values_z.append(p_val_z)
        angle_errors.append(np.mean(angle_error, axis=2))

        # divide by visibility
        angle_errors_vis_ = list()
        visibilities = range(1, 5)
        for v, vis in enumerate(visibilities):
            indx = np.where(events['response_visibilityCode'][sel]==vis)[0]
            angle_errors_vis_.append(np.mean(angle_error[:,:,indx],axis=2))
        angle_errors_vis.append(angle_errors_vis_)


    # A)
    # plot error, ie class distance
    # plt.subplot(221)
    # plt.imshow(mean(angle_errors,axis=0),interpolation='none',origin='lower')
    # plt.colorbar()
    # plt.title('weighted negative error')

    # B)
    # define X
    chance_level = np.pi / 6.

    # perform cluster test on main effect (difference from chance)
    cluster_test_main(gat, angle_errors, chance_level, ylabel='Angle error')

    # Test wether decoding is significant at eahc visibility level
    for v, vis in enumerate(visibilities):
        # define A (non corrected measure) XXX JRK : to be solved later: last subject doesnt have unseen trials
        A = np.array(angle_errors_vis)[0:19, v, :, :]

        # bound to common scale
        lims = [np.min(A), np.max(A)]

        cluster_test_main(gat, A, chance_level, ylabel='Angle error', title=vis,
                            lims=lims)

    # XXX add saveing
