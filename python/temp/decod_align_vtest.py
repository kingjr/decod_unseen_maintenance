import pickle
import os.path as op
import numpy as np
import matplotlib.pyplot as plt
from pycircstat import vtest, rayleigh
from config import (subjects, data_path)
from myfunctions import realign_angle, cart2pol, plot_circ_hist

# tmp
#subjects=[subjects[i] for i in range(10)]

# This is the stats across trials, but really we only need to apply a similar
# thing across subjects.
p_val = list()
angle_error_across_weighted  = np.zeros([29, 29, 400, 20])
angle_error_across_argmax = np.zeros([29, 29, 400, 20])
for s, subject in enumerate(subjects):
    print(subject)
    path_gat = op.join(data_path, subject, 'mvpas',
        '{}-decod_{}_{}.pickle'.format(subject, 'orientation_target', 'SVC'))

    ###### PREPROC
    # Compute angle errors by realigning the predicted categories according to
    # true category
    # read GAT data
    with open(path_gat) as f:
        gat, contrast = pickle.load(f)

    n_time, n_test_time, n_trials, n_categories = dims = np.shape(gat.y_pred_)
    # realign to 4th angle category
    probas = realign_angle(gat, dims)

    # transform 6 categories into single angle: there are two options here,
    # try it on the pilot subject, and use weighted mean if the two are
    # equivalent
    prob2angle = 'weighted_mean'
    if prob2angle == 'most_likely_angle':
        angle_errors = np.argmax(probas, axis=3) * np.pi / 3 - np.pi / 6
    elif prob2angle == 'weighted_mean':
        operator = np.tile(np.arange(n_categories),
                         (n_time, n_time, n_trials, 1))
        weighted_errors = (probas * operator)* np.pi / 3 - np.pi / 6
        x = np.mean(np.cos(weighted_errors),  axis=3)
        y = np.mean(np.sin(weighted_errors), axis=3)
        angle_errors_weighted, radius = cart2pol(x,y)

    # compare weighted mean with argmax selection. To be removed once decided XXX
    angle_errors_argmax = np.argmax(probas, axis=3) * np.pi / 3 - np.pi / 6

    ####### STATS WITHIN SUBJECTS
    # Apply v test and retrieve statistics that is independent of the number of
    # trials.
    p_val_v, _ = vtest(angle_errors_weighted, 0, axis=2) # trials x time

    # apply Rayleigh test (better for of diagonal with distribution shifts, e.g.
    # on the n200, the prediction may reverse)
    p_val_ray, _ = rayleigh(angle_errors_weighted, axis=2) # trials x time

    # append scores
    p_val.append([p_val_v, p_val_ray])

    ####### STATS ACROSS SUBJECTS
    # apply Rayleigh test (better for of diagonal with distribution shifts, e.g.
    # on the n200, the prediction may reverse)
    angle_error_across_weighted[:, :, :, s] = angle_errors_weighted
    angle_error_across_argmax[:, :, :, s] = angle_errors_argmax


# plot error, ie class distance
plt.subplot(221)
plt.imshow(-mean(mean(angle_error_across_weighted,axis=2),axis=2),interpolation='none',origin='lower')
plt.colorbar()
plt.title('weighted error')

plt.subplot(222)
plt.imshow(-mean(mean(angle_error_across_argmax,axis=2),axis=2),interpolation='none',origin='lower')
plt.colorbar()
plt.title('argmax error')

# plot average Rayleigh p value
plt.subplot(223)
mean_p_vals = np.zeros((n_time, n_test_time, len(subjects)))
for s, subject in enumerate(subjects):
    mean_p_vals[:, :, s] = np.array(p_val[s][0])
plt.imshow(np.mean(mean_p_vals, axis=2), origin='lower',interpolation='none')
plt.colorbar()
plt.title('mean p-val Rayleigh')

## STATS ACROSS SUBJECTS - tentative, dubious method XXX
plt.subplot(224)
p, z = rayleigh(angle_error_across, axis=2, d=20)
plt.imshow(np.mean(z, axis=2), origin='lower',interpolation='none')
plt.colorbar()
plt.title('z Rayleigh (dubious method)')
