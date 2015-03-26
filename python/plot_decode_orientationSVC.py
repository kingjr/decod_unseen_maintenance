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
    mn_probas = mn[:,:,0]
    return mn_probas



contrasts = ('orientation_target','orientation_probe')
probas_grand=np.array(np.zeros([20,23,23]))
for contrast in contrasts:
    for s in np.append(0,range(2,21)):
        print(s)
        subject = subjects[s]
        # Define path to retrieve classifier
        path_gat = op.join(data_path, subject, 'mvpas',
            '{}-decod_{}.pickle'.format(subject, contrast))

        probas = realign_svc_orientations(path_gat)
        probas_grand[s,:,:] = probas[::-1,:]
        # # Show insividual subjects imagesc plots
        # fig, (ax1) = plt.subplots(nrows=1, figsize=(6,10))
        #
        # ax1.matshow(probas[::-1,:], extent=[0,22,0,22])
        # ax1.set_title('Default')
        #
        # plt.tight_layout()
        # plt.show()

    # Plot grand average across subjects
    gravg_probas = probas_grand.mean(0)
    fig, (ax1) = plt.subplots(nrows=1, figsize=(10,10))

    ax1.matshow(gravg_probas, extent=[0,22,0,22])
    ax1.set_title('Default')

    plt.tight_layout()
    plt.show()
