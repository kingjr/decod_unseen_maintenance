import os
import os.path as op

import numpy as np

import matplotlib.pyplot as plt
import pandas as pd
import scipy.io as sio

def get_data(meg_fname, bhv_fname):
    from mne.io.meas_info import create_info
    from mne.epochs import EpochsArray
    """XXX Here explain what this does"""
    # import information from fieldtrip data to get data shape
    ft_data = sio.loadmat(meg_fname + '.mat', squeeze_me=True, struct_as_record=True)['data']
    # import binary MEG data
    bin_data = np.fromfile(meg_fname + '.dat', dtype=np.float32)
    Xdim = ft_data['Xdim'].item()
    bin_data = np.reshape(bin_data, Xdim[[2, 1, 0]]).transpose([2, 1, 0])

    # Create an MNE Epoch
    n_trial, n_chans, n_time = bin_data.shape
    sfreq = ft_data['fsample'].item().item()
    time = ft_data['time'].item()[0]
    tmin = min(time)
    chan_names = [str(label) for label in ft_data['label'].item()]
    chan_types = np.squeeze(np.concatenate(
                    (np.tile(['grad', 'grad', 'mag'], (1, 102)),
                     np.tile('misc', (1, n_chans - 306))), axis=1))
    info = create_info(chan_names, sfreq, chan_types)
    events = np.c_[np.cumsum(np.ones(n_trial)) * 5 * sfreq,
                   np.zeros(n_trial),
                   ft_data['trialinfo'].item()]
    epochs = EpochsArray(bin_data, info, events=events, tmin=tmin)

    # Load behavioral file
    trials = sio.loadmat(bhv_fname, squeeze_me=True,
                             struct_as_record=True)["trials"]

    keys = ['ISI', 'response_keyPressed', 'time_jitter',
            'block', 'response_responsed', 'time_maskIn',
            'break', 'response_tilt', 'time_preparation',
            'contrast', 'response_time', 'time_probe',
            'correct', 'response_vis_RT', 'time_prompt',
            'feedback', 'response_vis_keyPressed', 'time_response',
            'gabors', 'response_vis_responsed', 'time_targetIn',
            'lambda', 'response_vis_time', 'time_targetOut',
            'localizer', 'response_visibilityCode', 'trialid',
            'orientation', 'tilt', 'ttl_value',
            'present', 'time_delay', 'response_RT', 'time_feedback_on']

    events = list()
    for trial in trials:
        event = dict()
        for key in keys:
                event[key] = trial[key]
        # manual new keys
        event['orientation_target']        = event['orientation']*30-15
        event['orientation_probe']         = (event['orientation']*30-15 + event['tilt'] * 30) % 180
        event['orientation_target_cos']    = np.cos(event['orientation_target'])
        event['orientation_target_sin']    = np.sin(event['orientation_target'])
        event['orientation_probe_cos']     = np.cos(event['orientation_probe'])
        event['orientation_probe_sin']     = np.sin(event['orientation_probe'])
        event['seen_unseen']               = event['response_visibilityCode'] > 1

        # append to all events
        events.append(event)
    events = pd.DataFrame(events)

    # Determine unique values within each panda column
#    tmp_list=['orientation_target',
#    'orientation_probe',
#    'orientation_target_cos',
#    'orientation_target_sin',
#    'orientation_probe_cos',
#    'orientation_probe_sin',
#    ]
#    unique_values=dict()
#    for x in tmp_list:
#        unique_values[x] = np.unique(events[x])

    return epochs, events


def resample_epochs(epochs, sfreq):
    """faster resampling"""
    # from librosa import resample
    # librosa.resample(channel, o_sfreq, sfreq, res_type=res_type)
    from scipy.signal import resample

    # resample
    epochs._data = resample(epochs._data,
                            epochs._data.shape[2] / epochs.info['sfreq'] * sfreq,
                            axis=2)
    # update metadata
    epochs.info['sfreq'] = sfreq
    epochs.times = (np.arange(epochs._data.shape[2],
                              dtype=np.float) / sfreq + epochs.times[0])
    return epochs

def decim(inst, decim):
    """faster resampling"""
    from mne.io.base import _BaseRaw
    from mne.epochs import _BaseEpochs
    if isinstance(inst, _BaseRaw):
         inst._data =  inst._data[:,::decim]
         inst.info['sfreq'] /= decim
         inst._first_samps /= decim
         inst.first_samp /= decim
         inst._last_samps /= decim
         inst.last_samp /= decim
         inst._raw_lengths /= decim
         inst._times =  inst._times[::decim]
    elif isinstance(inst, _BaseEpochs):
        inst._data = inst._data[:,:,::decim]
        inst.info['sfreq'] /= decim
        inst.times = inst.times[::decim]
    return inst
