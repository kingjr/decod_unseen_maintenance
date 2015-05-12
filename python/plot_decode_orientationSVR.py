"""This script is used to loop across subjects to retrieve target
                and probe SVRs and compute the predicted reconstructed angle"""

import os.path as op
import numpy as np
import matplotlib.pyplot as plt
from pycircstat import (rayleigh,
                        omnibus,
                        vtest,
                        corrcc)
from config import (subjects,
                    data_path,
                    inputTypes)

from myfunctions import (recombine_svr_prediction)

# Define subjects and contrasts of interest
# subjects = [subjects[i] for i in range(20) # XXX to be be removed
contrasts = ['orientation_target']

# Loop across conditions
# Input type defines whether we decode ERFs or frequency power
for typ in inputTypes:
    print(typ)
    for contrast in contrasts:
        # define condition string
        cond_name_sin = contrast + '_sin'
        cond_name_cos = contrast + '_cos'

        # Define classifier type
        clf_type = 'SVR'

        # initialize statistics measures
        init_var = np.array(np.zeros([20, 29, 29]))
        error_grand = init_var
        p_rayleigh = init_var
        z_rayleigh = init_var
        p_omni = init_var
        m_omni = init_var
        v_vtest = init_var
        p_vtest = init_var
        r_corr = init_var
        # loop across subjects
        for s, subject in enumerate(subjects):
            print(subject)
            # define meg_path appendix
            fname_appendix = op.join('_Tfoi_mtm_', typ['name'][4:], 'Hz')

            # # load behavioral data
            # meg_fname = op.join(data_path, subject, 'preprocessed',
            #                     subject + '_preprocessed' + fname_appendix)
            # bhv_fname = op.join(data_path, subject, 'behavior',
            #                     subject + '_fixed.mat')
            # epochs, events = get_data(meg_fname, bhv_fname)

            # Define path to retrieve cosine classifier
            path_x = op.join(data_path, subject, 'mvpas',
                             '{}-decod_{}_{}{}.pickle'.format(subject,
                                                              cond_name_cos,
                                                              clf_type,
                                                              fname_appendix))

            # Define path to retrieve sine classifier
            path_y = op.join(data_path, subject, 'mvpas',
                             '{}-decod_{}_{}{}.pickle'.format(subject,
                                                              cond_name_sin,
                                                              clf_type,
                                                              fname_appendix))

            # recombine cos and sin predictions to obtain
            # the predicted angle in radians
            predAngle, radius, trueX = recombine_svr_prediction(path_x,
                                                                path_y)

            # retrieve angle presented in radians
            trueAngle = (np.arccos(trueX) * 2) - np.pi

            # Compute several measures to estimate goodness of fit
            # MEAN SQUARED ERROR ##############################################
            dims = predAngle.shape
            error = (predAngle.squeeze() - np.tile(trueAngle,
                                                   np.append(dims[0:2],
                                                             1))) ** 2

            # compute truth-prediction square error
            error_grand[s, :, :] = error.mean(2)

            # RAYLEIGH TEST CIRCULR  ##########################################
            p_rayleigh[s, :, :], z_rayleigh[s, :, :] = rayleigh(
                predAngle.squeeze(), axis=2)

            #  OMNIBUS TEST  ##################################################
            p_omni[s, :, :], m_omni[s, :, :] = omnibus(
                predAngle.squeeze(), axis=2)

            #  V-TEST #########################################################
            p_vtest[s, :, :], v_vtest[s, :, :] = vtest(
                predAngle.squeeze(), 0, axis=2)

            # CIRCULAR CORRLELATION ###########################################
            r_corr[s, :, :] = corrcc(predAngle.squeeze(),
                                     np.tile(trueAngle,
                                     np.append(dims[0:2], 1)), axis=2)

    break

plt.subplot(231)
plt.imshow(error_grand.mean(axis=0), interpolation='none', origin='lower')
plt.title('squared error')
plt.colorbar()

plt.subplot(232)
plt.imshow(z_rayleigh.mean(axis=0), interpolation='none', origin='lower')
plt.title('rayleigh')
plt.colorbar()

plt.subplot(233)
plt.imshow(m_omni.mean(axis=0), interpolation='none', origin='lower')
plt.title('omnibus test')
plt.colorbar()

plt.subplot(234)
plt.imshow(v_vtest.mean(axis=0), interpolation='none', origin='lower')
plt.title('V- test')
plt.colorbar()

plt.subplot(235)
plt.imshow(r_corr.mean(axis=0), interpolation='none', origin='lower')
plt.title('Circ-correlation')
plt.colorbar()
