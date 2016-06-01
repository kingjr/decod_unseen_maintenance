""""Main decoding pipeline, consisting in fitting for each subject separately,
a linear multivariate regresser (catgorical, ordinal or circular) that
optimally predicts the trials' value from a single time slice.
"""
import numpy as np
from jr.gat import TimeFrequencyDecoding
from mne.decoding import TimeDecoding
from config import subjects, load, save
from conditions import analyses


for s, subject in enumerate(subjects):  # Loop across each subject
    print(subject)

    epochs = load('epochs', subject=subject, preload=True)
    events = load('behavior', subject=subject)
    start = np.where(epochs.times >= -.200)[0][0]
    stop = np.where(epochs.times >= 1.400)[0][0]

    # Apply to each analysis
    for analysis in analyses:
        query, condition = analysis['query'], analysis['condition']
        sel = range(len(events)) if query is None \
            else events.query(query).index
        sel = [ii for ii in sel if ~np.isnan(events[condition][sel][ii])]
        y = np.array(events[condition], dtype=np.float32)

        print analysis['name'], np.unique(y[sel]), len(sel)

        if len(sel) == 0:
            continue

        # Apply analysis
        td = TimeDecoding(clf=analysis['clf'], cv=analysis['cv'],
                          scorer=analysis['scorer'], n_jobs=-1)
        tfd = TimeFrequencyDecoding(frequencies=np.arange(8, 70, .5), td=td,
                                    tfr_kwargs=dict(n_cycles=5),
                                    decim=slice(start, stop, 10))
        tfd.fit(epochs[sel], y=y[sel])
        tfd.score(epochs[sel], y=y[sel])

        # Save analysis
        save([tfd, analysis, sel, events], 'decod-tfr',
             subject=subject, analysis=analysis['name'],
             upload=True, overwrite=True)
