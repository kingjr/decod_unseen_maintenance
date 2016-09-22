"""Make a plot of the selected region of interest typically associated with
visual perception (i.e. ventral and dorsal stream + PFC)"""

import numpy as np
import matplotlib.pyplot as plt
import mne
from surfer import Brain

rois = ['lingual', 'inferiortemporal', 'superiorparietal',
        'supramarginal', 'rostralmiddlefrontal', 'precentral']

brain = Brain('fsaverage', 'split', 'inflated', background='w')
labels = mne.read_labels_from_annot('fsaverage', parc='aparc')
cmap = plt.get_cmap('rainbow')
colors = cmap(np.linspace(0., 1., len(rois)))
for roi, color in zip(rois, colors):
    for hemi in ['lh', 'rh']:
        label = [lbl for lbl in labels if lbl.name == (roi + '-' + hemi)][0]
        brain.add_label(label, color=color, alpha=.9, hemi=hemi)
