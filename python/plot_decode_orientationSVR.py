"""This script is used to loop across subjects to retrieve target
                and probe SVRs and compute the predicted reconstructed angle"""

import os.path as op
import numpy as np
import pickle
import recombine_svr_prediction
import matplotlib.pyplot as plt
from pycircstat import (
        rayleigh,
        omnibus,
        vtest
)
from config import (
        subjects,
        data_path,
        inputTypes)
from utils import get_data
from myfunctions import (
                    cart2pol,
                    recombine_svr_prediction
)

# Define subjects and contrasts of interest
#subjects = [subjects[i] for i in range(20) # XXX to be be removed
contrasts = ['orientation_target']

# Loop across conditions
# Input type defines whether we decode ERFs or frequency power
for typ in inputTypes:
    print(typ)
    # loop only once if ERF and across all frequencies of interest if frequency
    for freq in typ['values']:
        print(freq)
        for contrast in contrasts:
            # define condition string
            cond_name_sin = contrast + '_sin'
            cond_name_cos = contrast + '_cos'

            # Define classifier type
            clf_type = 'SVR'

            # initialize statistics measures
            error_grand = np.array(np.zeros([20,29,29]))
            p_rayleigh = np.array(np.zeros([20,29,29]))
            z_rayleigh = np.array(np.zeros([20,29,29]))
            p_omni = np.array(np.zeros([20,29,29]))
            m_omni = np.array(np.zeros([20,29,29]))
            v_vtest = np.array(np.zeros([20,29,29]))
            p_vtest = np.array(np.zeros([20,29,29]))

            # loop across subjects
            for s, subject in enumerate(subjects):
                print(subject)
                # define meg_path appendix
                if typ['name']=='erf':
                    fname_appendix = ''
                elif typ['name']=='power':
                    fname_appendix = op.join('_Tfoi_mtm_',freq,'Hz')

                # load behavioral data
                meg_fname = op.join(data_path, subject, 'preprocessed',
                                    subject + '_preprocessed' + fname_appendix)
                bhv_fname = op.join(data_path, subject, 'behavior',
                                    subject + '_fixed.mat')
                epochs, events = get_data(meg_fname, bhv_fname)

                # Define path to retrieve cosine classifier
                path_x = op.join(data_path, subject, 'mvpas',
                    '{}-decod_{}_{}{}.pickle'.format(subject, cond_name_cos,
                                                    clf_type,fname_appendix))

                # Define path to retrieve sine classifier
                path_y = op.join(data_path, subject, 'mvpas',
                    '{}-decod_{}_{}{}.pickle'.format(subject, cond_name_sin,
                                                    clf_type,fname_appendix))

                # recombine cos and sin predictions to obtain
                # the predicted angle in radians
                predAngle, radius, trueX = recombine_svr_prediction(path_x,
                                                                    path_y)

                # retrieve angle presented in radians
                trueAngle = np.arccos(trueX)

                # Compute several measures to estimate goodness of fit
                ######################MEAN SQUARED ERROR########################
                dims = predAngle.shape
                error = ((predAngle.squeeze() - np.tile(trueAngle,
                            np.append(dims[0:2],1)) % (2*np.pi)) - np.pi ) ** 2

                # compute truth-prediction square error
                error_grand[s, :, :] = error.mean(2)

                ################ RAYLEIGH TEST CIRCULR  ########################
                p_rayleigh[s, :, :], z_rayleigh[s, :, :] = rayleigh(2 *
                                            predAngle.squeeze() - np.pi, axis=2)

                ###############  OMNIBUS TEST  #################################
                p_omni[s, :, :], m_omni[s, :, :] = omnibus(2 *
                                            predAngle.squeeze() - np.pi, axis=2)

                ################## V-TEST ######################################
                p_vtest[s, :, :], v_vtest[s, :, :] = vtest(2 *
                                         predAngle.squeeze() - np.pi, 0, axis=2)
    break

plt.subplot(2, 2, 1)
plt.imshow(error_grand[0,:,:], interpolation='none', origin='lower')
plt.title('squared error')
plt.colorbar()

plt.subplot(2, 2, 2)
plt.imshow(z_rayleigh[0,:,:], interpolation='none', origin='lower')
plt.title('rayleigh')
plt.colorbar()

plt.subplot(2, 2, 3)
plt.imshow(m_omni[0,:,:], interpolation='none', origin='lower')
plt.title('omnibus test')
plt.colorbar()

plt.subplot(2, 2, 4)
plt.imshow(v_vtest[0,:,:], interpolation='none', origin='lower')
plt.title('V- test')
plt.colorbar()
