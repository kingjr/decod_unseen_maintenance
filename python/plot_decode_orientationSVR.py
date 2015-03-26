"""This script is used to loop across subjects to retrieve target
                and probe SVRs and compute the predicted reconstructed angle"""

import os.path as op
import numpy as np
import pickle
import recombine_svr_prediction
import matplotlib.pyplot as plt
from config import (
        subjects,
        data_path,
        inputTypes)
from utils import get_data


################################################################################
def recombine_svr_prediction(path_x,path_y):

    #load first regressor (cosine)
    with open(path_x) as f:
        gat, contrast = pickle.load(f)
    x = np.array(gat.y_pred_)
    true_x = gat.y_train_

    #load second regressor (sine)
    with open(path_y) as f:
        gat, contrast = pickle.load(f)
    y = np.array(gat.y_pred_)

    # cartesian 2 polar transformation
    angle, radius = cart2pol(x, y)

    return (angle, radius, true_x)


def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

################################################################################

subjects = [subjects[i] for i in range(20) if i not in [1]] # XXX to be be removed
contrasts = ['orientation_target']
for typ in inputTypes:                                                      # Input type defines whether we decode ERFs or frequency power
    print(typ)
    for freq in typ['values']:                                              # loop only once if ERF and across all frequencies of interest if frequency power
        print(freq)
        for contrast in contrasts:
            # define condition string
            cond_name_sin = contrast + '_sin'
            cond_name_cos = contrast + '_cos'

            # Define classifier type
            clf_type = 'SVR'

            error_grand = np.array(np.zeros([20,23,23]))
            for s, subject in enumerate(subjects):
                print(subject)
                # define meg_path appendix
                if typ['name']=='erf':
                    fname_appendix = ''
                elif typ['name']=='power':
                    fname_appendix = op.join('_Tfoi_mtm_',freq,'Hz')

                # load behavioral data
                meg_fname = op.join(data_path, subject, 'preprocessed', subject + '_preprocessed' + fname_appendix)
                bhv_fname = op.join(data_path, subject, 'behavior', subject + '_fixed.mat')
                epochs, events = get_data(meg_fname, bhv_fname)

                # Define path to retrieve cosine classifier
                path_x = op.join(data_path, subject, 'mvpas',
                    '{}-decod_{}_{}{}.pickle'.format(subject, cond_name_cos, clf_type,fname_appendix))

                # Define path to retrieve sine classifier
                path_y = op.join(data_path, subject, 'mvpas',
                    '{}-decod_{}_{}{}.pickle'.format(subject, cond_name_sin, clf_type,fname_appendix))

                # recombine cos and sin predictions to obtain the predicted angle
                predAngle, radius, trueX = recombine_svr_prediction(path_x,path_y)

                # retrieve angle presented in degrees
                trueAngle = np.rad2deg(np.arccos(trueX))

                dims = predAngle.shape
                error = ((predAngle.squeeze() - np.tile(trueAngle,np.append(dims[0:2],1)) % 360) - 180 ) ** 2

                # compute truth-prediction square error
                mn_error = error.mean(2)

                error_grand[s,:,:] = mn_error
    break



fig, ax = plt.subplots(nrows=1, figsize=(10,10))

ax.imshow(error_grand.mean(0), extent=[0,22,0,22], interpolation='none', origin='lower')
ax.set_title('Default')

plt.tight_layout()
plt.show()
