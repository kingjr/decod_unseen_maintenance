"""Performs statistics across subjects at the sensor level"""
import numpy as np
from jr.stats import robust_mean
import mne
from mne.epochs import EpochsArray
from mne.stats import spatio_temporal_cluster_1samp_test as stats
from mne.channels import read_ch_connectivity
from config import save, load, subjects
from conditions import analyses

# Apply contrast on each type of epoch
for analysis in analyses:
    print analysis['name']

    # Load data across all subjects
    data = list()
    for s, subject in enumerate(subjects):
        evoked, sub, _ = load('evoked', subject=subject,
                              analysis=analysis['name'])
        evoked.pick_types(meg=True, eeg=False)
        data.append(evoked.data)

    data = np.array(data)
    epochs = EpochsArray(data, evoked.info,
                         events=np.zeros((len(data), 3), dtype=int),
                         tmin=evoked.times[0])
    # combine grad at subject level
    grad = mne.pick_types(evoked.info, 'grad')
    if analysis['typ'] == 'categorize':
        data[:, grad, :] -= .5  # AUC center
    data[:, grad[::2], :] = np.sqrt(data[:, grad[::2], :] ** 2 +
                                    data[:, grad[1::2], :] ** 2)
    data[:, grad[1::2], :] = 0

    # keep robust averaging for plotting
    epochs = EpochsArray(data, evoked.info,
                         events=np.zeros((len(data), 3), dtype=int),
                         tmin=evoked.times[0])
    evoked = epochs.average()
    evoked.data = robust_mean(data, axis=0)

    # Run stats
    p_values_chans = list()
    epochs.pick_types('mag')
    connectivity, _ = read_ch_connectivity('neuromag306mag')
    X = np.transpose(epochs._data, [0, 2, 1])
    _, clusters, cl_p_val, _ = stats(
        X, out_type='mask', n_permutations=2 ** 11,
        connectivity=connectivity, n_jobs=-1)
    p_values = np.ones_like(X[0]).T
    for cluster, pval in zip(clusters, cl_p_val):
        p_values[cluster.T] = pval
    sig = p_values < .05

    # Save contrast
    save([evoked, data, p_values, sig, analysis],
         'evoked', analysis=('stats_' + analysis['name']))
