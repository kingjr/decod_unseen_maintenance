# Author: Jean-Remi King <jeanremi.king@gmail.com>
#
# Licence: BSD 3-clause

"""Preprocess continuous (raw) data and epoch it into epochs."""
import os.path as op
import numpy as np
from scipy.io import loadmat
from nose.tools import assert_true
from mne.io import RawArray
from mne import Epochs, find_events, create_info, concatenate_epochs
from mne.filter import low_pass_filter
from config import paths, load, save


def _epoch_raw(subject, block, overwrite=False):
    """high pass filter raw data, make consistent channels and epoch."""

    # Checks if preprocessing has already been done
    epo_fname = paths('epo_block', subject=subject, block=block)
    if op.exists(epo_fname) and not overwrite:
        return
    print(subject, block)

    # Load raw data
    raw = load('sss', subject=subject, block=block, preload=True)

    # Explicit picking of channel to ensure same channels across subjects
    picks = ['STI101', 'EEG060', 'EOG061', 'EOG062', 'ECG063', 'EEG064',
             'MISC004']

    # Potentially add forgotten channels
    ch_type = dict(STI='stim', EEG='eeg', EOG='eog', ECG='ecg',
                   MIS='misc')
    missing_chans = list()
    for channel in picks:
        if channel not in raw.ch_names:
            missing_chans.append(channel)
    if missing_chans:
        info = create_info(missing_chans, raw.info['sfreq'],
                           [ch_type[ch[:3]] for ch in missing_chans])
        raw.add_channels([RawArray(
            np.zeros((len(missing_chans), raw.n_times)), info,
            raw.first_samp)], force_update_info=True)

    # Select same channels order across subjects
    picks = [np.where(np.array(raw.ch_names) == ch)[0][0] for ch in picks]
    picks = np.r_[np.arange(306), picks]

    # high pass filtering
    raw.filter(.1, None, l_trans_bandwidth=.05, filter_length='30s',
               n_jobs=1)

    # Ensure same sampling rate
    if raw.info['sfreq'] != 1000.0:
        raw.resample(1000.0)

    # Select events
    events = find_events(raw, stim_channel='STI101', shortest_event=1)
    sel = np.where(events[:, 2] <= 255)[0]
    events = events[sel, :]

    # Compensate for delay (as measured manually with photodiod
    events[1, :] += .050 * raw.info['sfreq']

    # Epoch continuous data
    epochs = Epochs(raw, events, reject=None, tmin=-.600, tmax=1.8,
                    picks=picks, baseline=None)
    save(epochs, 'epo_block', subject=subject, block=block, overwrite=True,
         upload=False)


def _concatenate_epochs(subject, overwrite=False):
    """Concatenate epoched blocks and check that matches with behavior file."""
    epo_fname = paths('epochs', subject=subject)
    if op.exists(epo_fname) and not overwrite:
        return
    print(subject)
    epochs = list()
    for block in range(1, 6):
        this_epochs = load('epo_block', subject=subject, block=block,
                           preload=False)
        epochs.append(this_epochs)
    epochs = concatenate_epochs(epochs)
    save(epochs, 'epochs', subject=subject, overwrite=True, upload=False)


def _check_epochs(subject):
    """Some triggers values are misread, but it should not concern more than
    5 trials"""
    epochs = load('epochs', subject=subject, preload=False)
    bhv_fname = paths('behavior', subject=subject)
    mat = loadmat(bhv_fname, squeeze_me=True, struct_as_record=False)
    trials = mat["trials"]
    diff = epochs.events[:, 2] - [trial.ttl_value for trial in trials]
    assert_true(len(np.unique(diff)) <= 5)


def _check_photodiod():
    """Use photodiod channel to detect stim latency with regard to trigger."""
    import matplotlib.pyplot as plt
    import numpy as np
    # load data
    epochs = load('epochs', subject=10)
    # pick photo diod channel
    epochs.pick_types(meg=False, eeg=False, stim=False, misc=True)
    # crop for better visibility
    epochs.crop(-.050, .150)
    # plot
    epochs.plot_image(picks=[0])
    # get precise delay
    data = np.mean(epochs.get_data(), axis=0)
    data -= np.median(data)
    data /= np.std(data)
    plt.plot(data.T)
    plt.show()
    print(epochs.times[np.where(data > 1.)[0][1]])


def _decimate_epochs(subject, overwrite=True):
    # check overwrite
    epo_fname = paths('epochs_decim', subject=subject)
    if op.exists(epo_fname) and not overwrite:
        return
    # load non decimated data
    epochs = load('epochs', subject=subject, preload=True)
    epochs._data = low_pass_filter(epochs._data, epochs.info['sfreq'], 30.,
                                   n_jobs=-1)
    epochs.crop(-.200, 1.600)
    epochs.decimate(10)
    save(epochs, 'epochs_decim', subject=subject, overwrite=True, upload=True)

for subject in range(1, 21):
    # high pass filter and epoch
    for block in range(1, 6):
        _epoch_raw(subject, block, overwrite=False)
    # concatenate epochs
    _concatenate_epochs(subject, overwrite=False)
    # check that match behavioral file
    _check_epochs(subject)
    # save decimated copy for ERF analyses
    _decimate_epochs(subject, overwrite=True)
