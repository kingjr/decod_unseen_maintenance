"""
This script retrieve classifier results from SVC and SVR on orientation
and plot the tuning curves, ie categories x time along the diagonal
classification.

niccolo.pescetelli@psy.ox.ac.uk
2015
"""

import pickle
import os.path as op

import numpy as np
import matplotlib.pyplot as plt

from config import (
                    subjects,
                    data_path,
                    results_path,
                    inputTypes,
                    clf_types,
)
from postproc_functions import (
                    realign_angle,
                    recombine_svr_prediction,
                    cart2pol,
                    plot_circ_hist,
                    hist_tuning_curve
)


# define input type: it is ERF (for now)
inputType=inputTypes[0]

"""
# -----------------SVR----------------------------------------------------------
"""
# classifier type is SVR
clf_type = clf_types[1]
# contrast is target orientation sine and cosine (for now...)
contrasts = clf_type['contrasts'][0:2]

# loop across subjects
for s, subject in enumerate(subjects):
    print(subject)

    # initialize variables if first subject
    if s == 0:
        res = 40
        trial_proportion = np.zeros([len(subjects),29,29,res])

    # define data path
    path_x = op.join(results_path, subject, 'mvpas',
        '{}-decod_{}_{}.pickle'.format(subject, contrasts[0]['name'], 'SVR'))

    path_y = op.join(results_path, subject, 'mvpas',
        '{}-decod_{}_{}.pickle'.format(subject, contrasts[1]['name'], 'SVR'))

    # load individual data
    with open(path_x) as f:
        gatx, contrast, sel, events = pickle.load(f)
    with open(path_y) as f:
        gaty, contrast, sel, events = pickle.load(f)

    ###### PREPROC
    # recombine cosine and sine predictions
    _, _, angle_errors = recombine_svr_prediction(gatx, gaty)

    # compute trial proportion
    trial_prop = hist_tuning_curve(angle_errors, res=res)

    # concatenate individual data
    trial_proportion[s,:,:,:] = trial_prop

# plot average tuning curve across subjects on the diagonal
trial_prop_diag = np.array([trial_proportion[:,t,t,:]
                                for t in np.arange(trial_proportion.shape[1])])
trial_prop_diag = trial_prop_diag.transpose([1,2,0])

plt.figure()
plt.imshow(trial_prop_diag.mean(axis=0), interpolation='none', origin='lower')
plt.colorbar()


"""
#-----------------------SVC-----------------------------------------------------
"""
# classifier type is SVC
clf_type = clf_types[0]
# contrast is target orientation (for now...)
contrast = clf_type['contrasts'][0]

for s, subject in enumerate(subjects):
    print(subject)
    # define individual data path
    path = op.join(results_path,subject,'mvpas',
        '{}-decod_{}_{}.pickle'.format(subject,contrast['name'],'SVC'))

    # load individual data
    with open(path) as f:
        gat, contrast, sel, events = pickle.load(f)

    ##### PREPROC
    # realign angle
    probas = realign_angle(gat)

    # initialize across subjects array if first subject
    if s == 0:
        dims = np.array(shape(probas))
        tuning_diag = np.zeros(np.append(len(subjects), dims[[1,3]]))
        tuning_diag_vis = np.zeros(np.append([len(subjects),4], dims[[1,3]]))

    # store tuning curve along diagonal
    tuning_diag[s,:,:] = np.mean([probas[t,t,:,:] for t in arange(dims[0])],axis=1)

    # divide by visibility
    for v,vis in enumerate(range(1,5)):
        subsel = [probas[t,t,events['response_visibilityCode'][sel]==vis,:] for t in arange(dims[0])]
        tuning_diag_vis[s,v,:,:] = np.mean(subsel,axis=1)

# plot AVERAGE tuning curve across subjects on diagonal
tuning_diag=np.mean(tuning_diag.transpose([0,2,1]),axis=0)
plt.figure()
plt.imshow(np.roll(tuning_diag,2,axis=0),interpolation='none')
plt.colorbar()

# plot each VISIBILITY
tuning_diag_vis_ = np.mean(tuning_diag_vis.transpose([0,1,3,2]),axis=0)
lims = [np.min(tuning_diag_vis_), np.max(tuning_diag_vis_)]
for v, vis in enumerate(range(1,5)):
    plt.subplot(4,1,vis)
    plt.imshow(np.roll(tuning_diag_vis_[v,:,:],2,axis=0),
                interpolation='none', vmin=lims[0], vmax=lims[1])
    plt.title(vis)
    plt.colorbar()
