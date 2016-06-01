""""Main decoding pipeline, consisting in fitting for each subject separately,
a linear multivariate regresser (catgorical, ordinal or circular) that
optimally predicts the trials' value from a single time slice.
"""
import numpy as np
from mne.decoding import GeneralizationAcrossTime
from config import subjects, load, save
from conditions import analyses


for s, subject in enumerate(subjects):  # Loop across each subject
    print(subject)

    epochs = load('epochs', subject=subject, preload=True)
    events = load('behavior', subject=subject)

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
        gat = GeneralizationAcrossTime(clf=analysis['clf'],
                                       cv=analysis['cv'],
                                       scorer=analysis['scorer'],
                                       n_jobs=-1)
        gat.fit(epochs[sel], y=y[sel])
        gat.score(epochs[sel], y=y[sel])

        # Save analysis
        save([gat, analysis, sel, events], 'decod',
             subject=subject, analysis=analysis['name'])
