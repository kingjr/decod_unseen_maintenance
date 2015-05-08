import pickle
import os.path as op

import numpy as np
import matplotlib.pyplot as plt

import mne

from config import (
                    subjects,
                    data_path,
                    pyoutput_path,
                    inputTypes,
                    clf_types
)
from utils import get_data
from postproc_functions import (realign_angle)



# ------------------------------------------------------------------------------
# -----------------------------------SVC----------------------------------------
# XXX temporary sub-selection
s=9
subjects = [subjects[s]]

# loop across subjects
for s, subject in enumerate(subjects):
    print(subject)

    # define results path
    path_gat = op.join(pyoutput_path, subject, 'mvpas',
        '{}-decod_{}_{}.pickle'.format(subject, 'targetAngle', 'SVC'))

    ###### PREPROC
    # Compute angle errors by realigning the predicted categories according to
    # true category

    # read GAT data
    with open(path_gat) as f:
        gat, contrast, sel, events = pickle.load(f)

    # define probas
    vis = np.array(events['seen_unseen'][sel])

    # save dimensions
    n_time, n_test_time, n_trials, n_categories = dims = np.shape(gat.y_pred_)

    if s == 0:
        # Initialize variables
        angle_errors = np.zeros(np.append(len(subjects),np.array(dims))

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


    # store variables across subjects
    angle_errors[s,:,:] = np.mean(angle_error[:, :, vis == True], axis=2)


# plot gat
plt.figure()
plt.subplot(211)
ax= plt.imshow(np.mean(angle_error[:, :, vis == True], axis=2),
                                interpolation='none',origin='lower')
plt.subplot(212)
plt.imshow(np.mean(angle_error[:, :, vis == False], axis=2),
                                interpolation='none',origin='lower')

# extract diagonal
probas_diag = np.array([angle_error[t,t,:] for t in range(n_time)])

# plot performance
plt.figure()
plt.plot(gat.train_times['times_'], np.mean(probas_diag[:, vis == True], axis=1))
plt.plot(gat.train_times['times_'], np.mean(probas_diag[:, vis == False], axis=1))
plt.legend(['seen','unseen'])
plt.show()


# ------------------------------------------------------------------------------
# -----------------------------------SVR----------------------------------------
