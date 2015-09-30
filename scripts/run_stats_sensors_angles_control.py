import pickle
import numpy as np
import mne
from mne.stats import spatio_temporal_cluster_1samp_test as stats
from mne.epochs import EpochsArray
from mne.channels import read_ch_connectivity
from jr.stats import corr_linear_circular, robust_mean
from orientations.utils import load_epochs_events, fix_wrong_channel_names
from scripts.config import paths, subjects

# Stats within subjects -----------------------------------------------
evoked_list = list()
for subject in subjects:
    # load
    print('load %s' % subject)
    epochs, events = load_epochs_events(subject, paths)
    data = epochs._data
    n_chans, n_times = data.shape[1:]
    data = np.reshape(data, [len(data), -1])
    angles = np.array(events['probe_circAngle'])
    tilts = np.array(events['probe_tilt'])
    present = np.array(events['target_present'])
    sel = np.where(present)[0]

    # circular linear correlation across all trials
    _, R2_all, _ = corr_linear_circular(data[sel, ...], angles[sel])

    # circular linear correlation within each tilt
    R2_tilts = list()
    for tilt in [-1, 1]:
        sel = np.where(tilts == tilt)[0]
        _, R2, _ = corr_linear_circular(data[sel, ...], angles[sel])
        R2_tilts.append(R2)

    # If pulling trials within tilt category improves the correlation
    # coefficient, it means that the correlation is biased towards or away
    # from the tilt
    bias = np.reshape(np.mean(R2_tilts, axis=0) - R2_all, [n_chans, n_times])

    # keep meg info
    evoked = epochs.average()
    evoked.data = bias
    evoked = fix_wrong_channel_names(evoked)  # FIXME ?
    evoked_list.append(evoked)

# Stats across subjects -----------------------------------------------
data = np.array([evo.data for evo in evoked_list])
epochs = EpochsArray(data, evoked_list[0].info, tmin=evoked.times[0],
                     events=np.zeros((len(data), 3), dtype=int))

# Combine grad at subject level
data = np.array(data)
grad = mne.pick_types(evoked.info, 'grad')
data[:, grad[::2], :] = np.sqrt(data[:, grad[::2], :] ** 2 +
                                data[:, grad[1::2], :] ** 2)
data[:, grad[1::2], :] = 0

# Keep robust averaging for plotting
epochs = EpochsArray(data, evoked.info, tmin=evoked.times[0],
                     events=np.zeros((len(data), 3), dtype=int))
evoked = epochs.average()
evoked.data = robust_mean(data, axis=0)

# Compute cluster corrected stats
p_values_chans = list()
epochs.pick_types('mag')
connectivity, _ = read_ch_connectivity('neuromag306mag')
X = np.transpose(epochs._data, [0, 2, 1])
_, clusters, cl_p_val, _ = stats(X, out_type='mask', n_permutations=2 ** 11,
                                 connectivity=connectivity, n_jobs=-1)
p_values = np.ones_like(X[0]).T
for cluster, pval in zip(clusters, cl_p_val):
    p_values[cluster.T] = pval
sig = p_values < .05

# Save contrast
pkl_fname = paths('evoked', analysis=('stats_angle_bias'), log=True)
with open(pkl_fname, 'wb') as f:
    pickle.dump([evoked, data, p_values, sig,
                 dict(name='angle_bias', title='Angle Bias')], f)
