import sys
sys.path.insert(0, './')

import pickle
import numpy as np
import matplotlib.pyplot as plt
from itertools import product

import mne
from mne.epochs import EpochsArray
from mne.stats import spatio_temporal_cluster_1samp_test as stats

from meeg_preprocessing.utils import setup_provenance

from base import (meg_to_gradmag, share_clim, tile_memory_free)
from orientations.utils import fix_wrong_channel_names

from config import (
    paths,
    subjects,
    data_types,
    analyses,
    chan_types,
    open_browser
)

# XXX uncomment
report, run_id, _, logger = setup_provenance(
    script=__file__, results_dir=paths('report'))

# Apply contrast on each type of epoch
for data_type, analysis in product(data_types, analyses):
    print data_type, analysis['name']

    # Load data across all subjects
    data = list()
    for s, subject in enumerate(subjects):
        pkl_fname = paths('evoked', subject=subject,
                          data_type=data_type,
                          analysis=analysis['name'])
        with open(pkl_fname, 'rb') as f:
            evoked, sub, _ = pickle.load(f)
        # FIXME
        evoked = fix_wrong_channel_names(evoked)
        data.append(evoked.data)

    epochs = EpochsArray(np.array(data), evoked.info,
                         events=np.zeros((len(data), 3)),
                         tmin=evoked.times[0])

    # TODO warning if subjects has missing condition
    p_values_chans = list()
    for chan_type in meg_to_gradmag(chan_types):
        # FIXME: clean this up by cleaning ch_types definition
        if chan_type['name'] == 'grad':
            # XXX JRK: With neuromag, should use virtual sensors.
            # For now, only apply stats to mag and grad.
            continue
        elif chan_type['name'] == 'mag':
            chan_type_ = dict(meg='mag')
        else:
            chan_type_ = dict(meg=chan_type['name'] == 'meg',
                              eeg=chan_type['name'] == 'eeg',
                              seeg=chan_type['name'] == 'seeg')

        pick_type = mne.pick_types(epochs.info, **chan_type_)
        picks = [epochs.ch_names[ii] for ii in pick_type]
        epochs_ = epochs.copy()
        epochs_.pick_channels(picks)

        # Run stats
        X = np.transpose(epochs_._data, [0, 2, 1])

        _, clusters, p_values, _ = stats(
            X, out_type='mask', n_permutations=2 ** 10,
            connectivity=chan_type['connectivity'],
            threshold=dict(start=.1, step=2.), n_jobs=-1)
        p_values = np.sum(clusters *
                          tile_memory_free(p_values, clusters[0].shape),
                          axis=0).T
        alpha = .05
        mask = p_values < alpha

        # Plot
        evoked = epochs_.average()

        # Plot butterfly
        # FIXME should concatenate p value across chan types first
        from matplotlib.path import Path
        from matplotlib.patches import PathPatch
        fig1, ax = plt.subplots(1)
        evoked.plot(axes=ax, show=False)
        sig_times = np.array(np.sum(mask, axis=0) > 0., dtype=int)
        ylim = ax.get_ylim()
        xx = np.hstack((evoked.times[0], evoked.times * 1000))
        yy = [ylim[ii] for ii in sig_times] + [ylim[0]]
        path = Path(np.array([xx, yy]).transpose())
        patch = PathPatch(path, facecolor='none', edgecolor='none')
        ax.add_patch(patch)
        im = ax.imshow(xx.reshape(np.size(yy), 1), cmap=plt.cm.gray,
                       origin='lower', alpha=.2,
                       extent=[np.min(xx), np.max(xx), ylim[0], ylim[1]],
                       aspect='auto', clip_path=patch, clip_on=True,
                       zorder=-1)

        # Plot image
        fig2, ax = plt.subplots(1)
        evoked.plot_image(axes=ax, show=False)
        x, y = np.meshgrid(evoked.times * 1000,
                           np.arange(len(evoked.ch_names)),
                           copy=False, indexing='xy')
        ax.contour(x, y, p_values < alpha, colors='black', levels=[0])

        # Plot topo
        sel_times = np.linspace(min(evoked.times), max(evoked.times), 20)
        fig3 = evoked.plot_topomap(mask=mask, scale=1., sensors=False,
                                   contours=False, times=sel_times,
                                   colorbar=True, show=False)
        share_clim(fig3.get_axes())

        # Add to report
        for fig, fig_name in zip([fig1, fig2, fig3],
                                 ('butterfly', 'image', 'topo')):
            report.add_figs_to_section(
                fig, ('%s (%s) %s: CONDITIONS (%s)' % (
                    subject, data_type, analysis['name'], fig_name)),
                analysis['name'])

    p_values_chans.append(p_values)
    # Save contrast
    pkl_fname = paths('evoked', subject='fsaverage',
                      data_type=data_type,
                      analysis=('stats_' + analysis['name']),
                      log=True)
    with open(pkl_fname, 'wb') as f:
        pickle.dump([p_values_chans, evoked, analysis], f)

report.save(open_browser=open_browser)
