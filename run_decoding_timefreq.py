""""Main decoding pipeline, consisting in fitting for each subject separately,
a linear multivariate regresser (catgorical, ordinal or circular) that
optimally predicts the trials' value from a single time slice.
"""
import numpy as np
from jr.gat import TimeFrequencyDecoding
from mne.decoding import TimeDecoding
from config import subjects, load, save, client, paths
from conditions import analyses


def _run(epochs, events, analysis):

    # Set time frequency parameters
    start = np.where(epochs.times >= -.200)[0][0]
    stop = np.where(epochs.times >= 1.400)[0][0]
    frequencies = np.logspace(np.log10(4), np.log10(80), 25)
    decim = slice(start, stop, 8)  # ~62 Hz after TFR

    # Get relevant trials
    query, condition = analysis['query'], analysis['condition']
    sel = range(len(events)) if query is None \
        else events.query(query).index
    sel = [ii for ii in sel if ~np.isnan(events[condition][sel][ii])]
    y = np.array(events[condition], dtype=np.float32)

    print analysis['name'], np.unique(y[sel]), len(sel)

    if len(sel) == 0:
        return

    # Apply analysis
    td = TimeDecoding(clf=analysis['clf'], cv=analysis['cv'],
                      scorer=analysis['scorer'], n_jobs=-1)
    tfd = TimeFrequencyDecoding(
        frequencies, td=td, n_jobs=-1,
        tfr_kwargs=dict(n_cycles=5, decim=decim))
    print(subject, analysis['name'], 'fit')
    tfd.fit(epochs[sel], y=y[sel])
    print(subject, analysis['name'], 'score')
    score = tfd.score()

    # Save analysis
    print(subject, analysis['name'], 'save')
    if analysis['name'] not in ['target_present', 'target_circAngle']:
        save([tfd.td.y_pred_, sel, events, epochs.times[decim], frequencies],
             'decod_tfr',
             subject=subject, analysis=analysis['name'], overwrite=True)
    save([score, epochs.times[decim], frequencies], 'score_tfr',
         subject=subject, analysis=analysis['name'], overwrite=True)

for s, subject in enumerate(subjects):  # Loop across each subject
    print(subject)

    epochs = load('epochs', subject=subject, preload=True)
    epochs.decimate(2)  # 500 Hz is enough
    epochs.pick_types(meg=True, eeg=False, stim=False, eog=False, ecg=False)
    events = load('behavior', subject=subject)

    # Apply to each analysis
    for analysis in analyses:
        fname = paths('score_tfr', subject=subject, analysis=analysis['name'])
        if client.metadata(fname)['exist']:
            continue
        _run(epochs, events, analysis)
