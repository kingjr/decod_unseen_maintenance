import pickle
import numpy as np
import matplotlib.pyplot as plt
from itertools import product

import mne
from meeg_preprocessing.utils import setup_provenance

from base import meg_to_gradmag, build_analysis
from orientations.utils import load_epochs_events

from config import (
    paths,
    subjects,
    data_types,
    evoked_analyses as analyses,  # FIXME
    chan_types,
    open_browser,
)

report, run_id, _, logger = setup_provenance(
    script=__file__, results_dir=paths('report'))

mne.set_log_level('INFO')

for subject, data_type in product(subjects, data_types):
    print(subject, data_type)

    epochs, events = load_epochs_events(subject, paths,
                                        data_type=data_type)
    # Apply each analysis
    for analysis in analyses:
        print(analysis['name'])
        coef, evokeds = build_analysis(analysis['conditions'], epochs, events,
                                       operator=analysis['operator'])

        # Save all_evokeds
        fname = paths('evoked', subject=subject, data_type=data_type,
                      analysis=analysis['name'])
        with open(fname, 'wb') as f:
            pickle.dump([coef, evokeds, analysis, events], f)

        # Prepare plot delta (subtraction, or regression)
        fig1, ax1 = plt.subplots(1, len(chan_types))
        if type(ax1) not in [list, np.ndarray]:
            ax1 = [ax1]

        # Plot coef
        evoked = epochs[0].average()
        evoked.data = coef.data
        fig1 = evoked.plot_image()
        report.add_figs_to_section(fig1, ('%s (%s) %s: COEF' % (
            subject, data_type, analysis['name'])), analysis['name'])

        # Plot subcondition # FIXME #2220
        fig2, ax2 = plt.subplots(len(meg_to_gradmag(chan_types)),
                                 len(evokeds['coef']), figsize=[19, 10])
        for e, evoked in enumerate(evokeds['coef']):
            evoked.data = evoked.data
            evoked.plot_image(axes=ax2[:, e], show=False,
                              titles=dict(grad='grad', mag='mag'))
        report.add_figs_to_section(fig2, ('%s (%s) %s: CONDITIONS' % (
            subject, data_type, analysis['name'])), analysis['name'])


report.save(open_browser=open_browser)
