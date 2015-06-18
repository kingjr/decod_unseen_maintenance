import pickle
import numpy as np
import matplotlib.pyplot as plt
from itertools import product

import mne
from mne.io.pick import _picks_by_type as picks_by_type
from meeg_preprocessing.utils import setup_provenance

from orientations.utils import (meg_to_gradmag, build_analysis,
                                load_epochs_events)

from config import (
    paths,
    subjects,
    data_types,
    evoked_analyses as analyses,  # FIXME
    chan_types,
    open_browser,
)

# report, run_id, _, logger = setup_provenance(
#     script=__file__, results_dir=paths('report'))
#
# mne.set_log_level('INFO')

chan_type = meg_to_gradmag(chan_types)

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
        # Prepare plot all conditions at top level of analysis
        fig2, ax2 = plt.subplots(len(evokeds['coef']), len(chan_types))
        ax2 = np.reshape(ax2, len(evokeds['coef']) * len(chan_types))

        # Plot per channel type
        for ch, chan_type in enumerate(chan_types):
            # Select specific types of sensor
            info = coef.info
            picks = [i for k, p in picks_by_type(info)
                     for i in p if k in chan_type['name']]
            # ---------------------------------------------------------
            # Plot coef (subtraction, or regression)
            # adjust color scale
            mM = np.percentile(np.abs(coef.data[picks, :]), 99.)

            # plot mean sensors x time
            ax1[ch].imshow(coef.data[picks, :], vmin=-mM, vmax=mM,
                           interpolation='none', aspect='auto',
                           cmap='RdBu_r', extent=[min(coef.times),
                           max(coef.times), 0, len(picks)])
            # add t0
            ax1[ch].plot([0, 0], [0, len(picks)], color='black')
            ax1[ch].set_title(chan_type['name'] + ': ' + coef.comment)
            ax1[ch].set_xlabel('Time')
            ax1[ch].set_adjustable('box-forced')

            # ---------------------------------------------------------
            # Plot all conditions at top level of analysis
            # XXX only works for +:- data
            mM = np.median([np.percentile(abs(e.data[picks, :]), 80.)
                            for e in evokeds['coef']])

            for e, evoked in enumerate(evokeds['coef']):
                ax_ind = e * len(chan_types) + ch
                ax2[ax_ind].imshow(evoked.data[picks, :], vmin=-mM,
                                   vmax=mM, interpolation='none',
                                   aspect='auto', cmap='RdBu_r',
                                   extent=[min(coef.times),
                                   max(evoked.times), 0, len(picks)])
                ax2[ax_ind].plot([0, 0], [0, len(picks)], color='k')
                ax2[ax_ind].set_title(chan_type['name'] + ': ' +
                                      evoked.comment)
                ax2[ax_ind].set_xlabel('Time')
                ax2[ax_ind].set_adjustable('box-forced')

        # Save figure
        report.add_figs_to_section(fig1, ('%s (%s) %s: COEF' % (
            subject, data_type, analysis['name'])), analysis['name'])

        report.add_figs_to_section(fig2, ('%s (%s) %s: CONDITIONS' % (
            subject, data_type, analysis['name'])), analysis['name'])


report.save(open_browser=open_browser)
