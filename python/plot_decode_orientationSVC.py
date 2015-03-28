"""This scripts takes the categorical predictions of a orientation SVC
    and realign them to be visually meaningful"""

import matplotlib.pyplot as plt
import numpy as np
import os.path as op
import pickle
from config import (
        subjects,
        data_path,
        clf_types)

def realign_svc_orientations(path_gat):
    # Load classification results
    with open(path_gat) as f:
        gat, contrast = pickle.load(f)

    # realign to 4th angle
    probas = np.zeros(gat.y_pred_.shape)
    angles=[15, 45, 75, 105, 135, 165]
    for a in range(6):
        sel = gat.y_train_ == angles[a]
        prediction = np.array(gat.y_pred_)
        i = np.array(range(a,6)+range(0,(a)))
        p = prediction[:,:,:,i]
        probas[:,:,sel,:] = p[:,:,sel,:]
    mn = probas.mean(2)
    return mn


subjects = [subjects[i] for i in range(20)] # XXX to be be removed
contrasts = ['orientation_target']
probas_grand=np.array(np.zeros([20,29,29]))
for contrast in contrasts:
    for s, subject in enumerate(subjects):
        print(subject)
        subject = subjects[s]
        fname_appendix=''

        cond_name = contrast

        # Define path to retrieve classifier
        path_gat = op.join(data_path, subject, 'mvpas',
            '{}-decod_{}_{}{}.pickle'.format(subject, cond_name,'SVC',fname_appendix))

        probas = realign_svc_orientations(path_gat)
        probas_grand[s,:,:] = probas
        # # Show insividual subjects imagesc plots
        # fig, (ax1) = plt.subplots(nrows=1, figsize=(6,10))
        #
        # ax1.matshow(probas[::-1,:], extent=[0,22,0,22])
        # ax1.set_title('Default')
        #
        # plt.tight_layout()
        # plt.show()

    # Plot grand average across subjects
    gravg_probas = probas_grand.mean(axis=0)
    fig, ax = plt.subplots(nrows=1, figsize=(10,10))

    ax.imshow(gravg_probas, extent=[0,22,0,22], interpolation='none', origin='lower')
    ax.set_title('Default')

    plt.tight_layout()
    plt.show()
