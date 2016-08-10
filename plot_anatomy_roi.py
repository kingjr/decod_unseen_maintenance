"""Plot region of interest"""
rois = ['lingual', 'inferiortemporal', 'superiorparietal',
        'supramarginal', 'rostralmiddlefrontal', 'precentral']
import mne
import matplotlib.pyplot as plt
import numpy as np
from surfer import Brain
brain = Brain('fsaverage', 'split', 'inflated', background='w')
labels = mne.read_labels_from_annot('fsaverage', parc='aparc')
cmap = plt.get_cmap('rainbow')
colors = cmap(np.linspace(0., 1., len(rois)))
for roi, color in zip(rois, colors):
    for hemi in ['lh', 'rh']:
        label = [lbl for lbl in labels if lbl.name == (roi + '-' + hemi)][0]
        brain.add_label(label, color=color, alpha=.9, hemi=hemi)
