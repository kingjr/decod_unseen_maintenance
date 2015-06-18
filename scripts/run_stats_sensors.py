import pickle
import numpy as np
import mne
from toolbox.utils import (cluster_stat, Evokeds_to_Epochs, decim)
from meeg_preprocessing.utils import setup_provenance

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


if 'meg' in [i['name'] for i in chan_types]:
    # find 'meg' ch_type
    i = [i for i, le_dict in enumerate(chan_types)
         if le_dict['name'] == 'meg'][0]
    meg_type = chan_types[i].copy()
    meg_type.pop('name', None)
    chan_types[i] = dict(name='mag', **meg_type)

# Apply contrast on each type of epoch
for data_type in data_types:  # Input type ERFs or frequency power
    print(data_type)
    if data_type == 'erf':
        fname_appendix = ''
        fileformat = '.dat'
    else:
        fname_appendix = '_Tfoi_mtm_' + data_type[4:] + 'Hz'
        fileformat = '.mat'

    for analysis in analyses:
        print(analysis['name'])
        evokeds = list()

        # Load data across all subjects
        for s, subject in enumerate(subjects):
            pkl_fname = paths('evoked', subject=subject,
                              data_type=data_type,
                              analysis=analysis['name'])
            with open(pkl_fname, 'rb') as f:
                coef, evoked, _, _ = pickle.load(f)
            evokeds.append(coef)

        epochs = Evokeds_to_Epochs(evokeds)

        # XXX to be removed later on
        epochs = decim(epochs, 4)

        # TODO warning if subjects has missing condition
        # XXX JRK: With neuromag, should have virtual sensor in the future.
        # For now, only apply stats to mag and grad.
        chan_types = [chan_types[0]]
        for chan_type in chan_types:
            # chan_type = chan_types[0]

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

            # Save contrast
            # TODO CHANGE SAVING TO SAVE MULTIPLE CHAN TYPES

            pkl_fname = paths('evoked', subject='fsaverage',
                              data_type=data_type,
                              analysis=('stats_' + analysis['name']),
                              log=True)
            with open(pkl_fname, 'wb') as f:
                pickle.dump([cluster, evokeds, analysis], f)

report.save(open_browser=open_browser)
