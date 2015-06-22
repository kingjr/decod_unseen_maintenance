import pickle
import numpy as np
import mne
from toolbox.utils import (cluster_stat, Evokeds_to_Epochs, decim)
from meeg_preprocessing.utils import setup_provenance
from base import meg_to_gradmag

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
for data_type in data_types:  # Input type ERFs or frequency power
    for analysis in analyses:
        print(analysis['name'])

        # Load data across all subjects
        evokeds = list()
        for s, subject in enumerate(subjects):
            pkl_fname = paths('evoked', subject=subject,
                              data_type=data_type,
                              analysis=analysis['name'])
            with open(pkl_fname, 'rb') as f:
                evoked, sub, _ = pickle.load(f)
            evokeds.append(evoked.data)

        epochs = Evokeds_to_Epochs(evokeds)

        # XXX to be removed later on
        epochs = decim(epochs, 4)

        # TODO warning if subjects has missing condition
        cluster_chans = list()
        for chan_type in meg_to_gradmag(chan_types):
            if chan_type == 'grad':
                # XXX JRK: With neuromag, should use virtual sensors.
                # For now, only apply stats to mag and grad.
                continue

            # Take first evoked to retrieve all necessary information
            picks = [epochs.ch_names[ii] for ii in mne.pick_types(
                     epochs.info, meg=chan_type['name'])]

            # Stats
            epochs_ = epochs.copy()
            epochs_.pick_channels(picks)

            # XXX wont work for eeg
            # Run stats
            cluster = cluster_stat(epochs_, n_permutations=2 ** 11,
                                   connectivity=chan_type['connectivity'],
                                   threshold=dict(start=1., step=1.),
                                   n_jobs=-1)

            # Plots
            i_clus = np.where(cluster.p_values_ < .01)
            fig = cluster.plot(i_clus=i_clus, show=False)
            report.add_figs_to_section(fig, '{}: {}: Clusters time'.format(
                data_type, analysis['name']),
                data_type + analysis['name'])

            # plot T vales
            fig = cluster.plot_topomap(sensors=False, contours=False,
                                       show=False)

            # Plot contrasted ERF + select sig sensors
            evoked = epochs.average()
            evoked.pick_channels(picks)

            # Create mask of significant clusters
            mask, _, _ = cluster._get_mask(i_clus)
            # Define color limits
            mM = np.percentile(np.abs(evoked.data), 99)
            # XXX JRK: pass plotting function to config
            times = np.linspace(min(evoked.times), max(evoked.times), 20)
            fig = evoked.plot_topomap(mask=mask.T, scale=1., sensors=False,
                                      contours=False,
                                      times=times,
                                      vmin=-mM, vmax=mM, colorbar=True)

            report.add_figs_to_section(fig, '{}: {}: topos'.format(
                data_type, analysis['name']),
                data_type + analysis['name'])

            report.add_figs_to_section(fig, '{}: {}: Clusters'.format(
                data_type, analysis['name']),
                data_type + analysis['name'])

        cluster_chans.append(cluster)
        # Save contrast
        pkl_fname = paths('evoked', subject='fsaverage',
                          data_type=data_type,
                          analysis=('stats_' + analysis['name']),
                          log=True)
        with open(pkl_fname, 'wb') as f:
            pickle.dump([cluster_chans, evokeds, analysis], f)

report.save(open_browser=open_browser)
