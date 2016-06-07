"""Preprocess continuous (raw) data with very high pass filtering to
demonstrate that late generalization is indeed stable at the single trials
level"""

import os.path as op
import numpy as np
import matplotlib.pyplot as plt
from mne.io import RawArray
from mne import Epochs, find_events, create_info, concatenate_epochs
from jr.plot import pretty_gat
from config import load, save, paths, report
from conditions import analyses
from base import stats


def _get_epochs(subject):
    # if already computed, lets load it from disk
    epo_fname = paths('epochs_vhp', subject=subject)
    if op.exists(epo_fname):
        return load('epochs_vhp', subject=subject, preload=True)

    # high pass filter and epoch
    for block in range(1, 6):

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

        # Filtered
        raw.filter(2, 30, l_trans_bandwidth=.5, filter_length='30s',
                   n_jobs=1)

        # Ensure same sampling rate
        if raw.info['sfreq'] != 1000.0:
            raw.resample(1000.0)

        # Select events
        events = find_events(raw, stim_channel='STI101', shortest_event=1)
        sel = np.where(events[:, 2] <= 255)[0]
        events = events[sel, :]

        # Compensate for delay (as measured manually with photodiod
        events[1, :] += int(.050 * raw.info['sfreq'])

        # Epoch continuous data
        this_epochs = Epochs(raw, events, reject=None, tmin=-.200, tmax=1.6,
                             picks=picks, baseline=None, decim=10)
        save(this_epochs, 'epo_block', subject=subject, block=block)
        this_epochs._data = None
        raw.data = None
        del this_epochs, raw

    epochs = list()
    for block in range(1, 6):
        this_epochs = load('epo_block', subject=subject, block=block)
        epochs.append(this_epochs)
    epochs = concatenate_epochs(epochs)

    # save for faster retest
    save(epochs, 'epochs_vhp', subject=subject, overwrite=True, upload=False)

    return epochs


def _decod(subject, analysis):
    from mne.decoding import GeneralizationAcrossTime

    # if already computed let's just load it from disk
    fname_kwargs = dict(subject=subject, analysis=analysis['name'] + '_vhp')
    score_fname = paths('score', **fname_kwargs)
    if op.exists(score_fname):
        return load('score', **fname_kwargs)

    epochs = _get_epochs(subject)
    events = load('behavior', subject=subject)

    # Let's not recompute everything, this is just a control analysis
    print(subject, analysis['name'])
    epochs._data = epochs.get_data()
    epochs.preload = True
    epochs.crop(0., .900)
    epochs.decimate(2)

    query, condition = analysis['query'], analysis['condition']
    sel = range(len(events)) if query is None else events.query(query).index
    sel = [ii for ii in sel if ~np.isnan(events[condition][sel][ii])]
    y = np.array(events[condition], dtype=np.float32)

    print analysis['name'], np.unique(y[sel]), len(sel)

    if len(sel) == 0:
        return

    # Apply analysis
    gat = GeneralizationAcrossTime(clf=analysis['clf'],
                                   cv=analysis['cv'],
                                   scorer=analysis['scorer'],
                                   n_jobs=-1)
    print(subject, analysis['name'], 'fit')
    gat.fit(epochs[sel], y=y[sel])
    print(subject, analysis['name'], 'score')
    score = gat.score(epochs[sel], y=y[sel])
    print(subject, analysis['name'], 'save')

    # save space
    gat.estimators_ = None
    gat.y_pred_ = None

    # Save analysis
    save([score, epochs.times], 'score', overwrite=True, upload=True,
         **fname_kwargs)
    return score, epochs.times


def _stats(analysis):
    """2nd order stats across subjects"""

    # if already computed lets just load it
    ana_name = 'stats_' + analysis['name'] + '_vhp'
    if op.exists(paths('score', analysis=ana_name)):
        return load('score', analysis=ana_name)

    # gather scores across subjects
    scores = list()
    for subject in range(1, 21):
        kwargs = dict(subject=subject, analysis=analysis['name'] + '_vhp')
        fname = paths('score', **kwargs)
        if op.exists(fname):
            score, times = load(**kwargs)
        else:
            score, times = _decod(subject, analysis)
        scores.append(score)
    scores = np.array(scores)

    # compute stats across subjects
    p_values = stats(scores - analysis['chance'])
    diag_offdiag = scores - np.tile([np.diag(sc) for sc in scores],
                                    [len(times), 1, 1]).transpose(1, 0, 2)
    p_values_off = stats(diag_offdiag)

    # Save stats results
    out = dict(scores=scores, p_values=p_values, p_values_off=p_values_off,
               times=times, analysis=analysis)
    save(out, 'score',  analysis=ana_name)
    return out


# only recompute on the relevant analyses
analyses = [ana for ana in analyses if ana['name'] in
            ['target_present', 'target_circAngle']]
for analysis in analyses:
    out = _stats(analysis)
    scores = out['scores']
    times = out['times']
    plt.matshow(np.mean(scores, axis=0))

    alpha = .05
    chance = analysis['chance']
    p_values = stats(scores - chance)
    clim = np.percentile(np.diag(np.mean(scores, axis=0)), 97)
    clim = [chance-(clim-chance), clim]
    fig, ax_gat = plt.subplots(1, figsize=[7, 5.5])
    pretty_gat(np.mean(scores, axis=0), times=times, sig=p_values < alpha,
               chance=chance, ax=ax_gat, clim=clim)
    report.add_figs_to_section([fig], [analysis['name']], analysis['name'])

report.save()
