import sys
sys.path.insert(0, './')
import matplotlib
matplotlib.use('Agg')

import pickle
import numpy as np
import matplotlib.pyplot as plt
from itertools import product

import mne
from meeg_preprocessing.utils import setup_provenance

from base import meg_to_gradmag, nested_analysis, share_clim
from orientations.utils import load_epochs_events

from scripts.config import (
    paths,
    subjects,
    data_types,
    analyses,
    chan_types,
    open_browser,
)

from scripts.transfer_data import upload_report

report, run_id, _, logger = setup_provenance(
    script=__file__, results_dir=paths('report'))

mne.set_log_level('INFO')

for subject, data_type in product(subjects, data_types):
    print('load %s %s' % (subject, data_type))

    epochs, events = load_epochs_events(subject, paths, data_type=data_type)

    # Apply each analysis
    for analysis in analyses:
        print(analysis['name'])
        coef, sub = nested_analysis(
            epochs._data, events, analysis['condition'],
            function=analysis.get('erf_function', None),
            query=analysis.get('query', None),
            single_trial=analysis.get('single_trial', False),
            y=analysis.get('y', None))

        evoked = epochs.average()
        evoked.data = coef

        # Save all_evokeds
        fname = paths('evoked', subject=subject, data_type=data_type,
                      analysis=analysis['name'], log=True)
        with open(fname, 'wb') as f:
            pickle.dump([evoked, sub, analysis], f)

        # Prepare plot delta (subtraction, or regression)
        fig1, ax1 = plt.subplots(1, len(chan_types))
        if type(ax1) not in [list, np.ndarray]:
            ax1 = [ax1]

        # Plot coef
        fig1 = evoked.plot_image(show=False)
        report.add_figs_to_section(fig1, ('%s (%s) %s: COEF' % (
            subject, data_type, analysis['name'])), analysis['name'])

        # Plot subcondition
        fig2, ax2 = plt.subplots(len(meg_to_gradmag(chan_types)),
                                 len(sub['X']), figsize=[19, 10])
        X_mean = np.mean([X for X in sub['X']], axis=0)
        for e, (X, y) in enumerate(zip(sub['X'], sub['y'])):
            evoked.data = X - X_mean
            evoked.plot_image(axes=ax2[:, e], show=False,
                              titles=dict(grad='grad (%.2f)' % y,
                                          mag='mag (%.2s)' % y))
        for chan_type in range(len(meg_to_gradmag(chan_types))):
            share_clim(ax2[chan_type, :])
        report.add_figs_to_section(fig2, ('%s (%s) %s: CONDITIONS' % (
            subject, data_type, analysis['name'])), analysis['name'])


report.save(open_browser=open_browser)
upload_report(report)
