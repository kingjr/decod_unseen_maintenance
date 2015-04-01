import pickle
import os.path as op

import numpy as np
import matplotlib.pyplot as plt

import mne

from config import subjects, data_path
from utils import get_data

subject = subjects[0]

pkl_fname = op.join(data_path, subject, 'mvpas',
    '{}-decod_{}_{}.pickle'.format(subject, 'presentAbsent', 'SVC'))

# load classifier results
with open(pkl_fname) as f:
    gat, contrast, sel, events = pickle.load(f)


gat_vis = gat
gat_invis = gat

gat.y_pred_ #XXX to be finished with the uploaded new version from MNE

probas = np.array([gat.y_pred_[t, t, :, 0] for t in range(7)])
vis = events['seen_unseen'][sel]

plt.figure()
plt.plot(gat.train_times['times_'], np.mean(probas[:, vis == True], axis=1))
plt.plot(gat.train_times['times_'], np.mean(probas[:, vis == False], axis=1))
plt.show()
