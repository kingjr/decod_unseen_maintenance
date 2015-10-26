import numpy as np
import pickle
from mne.decoding import GeneralizationAcrossTime
from orientations.utils import load_epochs_events
from scripts.base import resample_epochs, decim
from scripts.config import paths, subjects, preproc, analyses

for s, subject in enumerate(subjects):  # Loop across each subject
    print(subject)
    epochs, events = load_epochs_events(subject, paths)

    # preprocess data for memory issue
    if 'resample' in preproc.keys():
        epochs = resample_epochs(epochs, preproc['resample'])
    if 'decim' in preproc.keys():
        epochs = decim(epochs, preproc['decim'])
    if 'crop' in preproc.keys():
        epochs.crop(preproc['crop']['tmin'],
                    preproc['crop']['tmax'])

    # select trials
    analysis = [a for a in analyses if a['name'] == 'target_circAngle'][0]
    query, condition = analysis['query'], analysis['condition']
    sel = range(len(events)) if query is None \
        else events.query(query).index
    sel = [ii for ii in sel if ~np.isnan(events[condition][sel][ii])]
    y = np.array(events[condition], dtype=np.float32)
    print analysis['name'], np.unique(y[sel]), len(sel)

    if len(sel) == 0:
        continue

    # Apply analysis
    train_times = dict(length=.200, step=.100)
    gat = GeneralizationAcrossTime(clf=analysis['clf'], cv=analysis['cv'],
                                   scorer=analysis['scorer'], n_jobs=-1,
                                   train_times=train_times,
                                   test_times=train_times)
    gat.fit(epochs[sel], y=y[sel])
    gat.score(epochs[sel], y=y[sel])

    # Save analysis
    pkl_fname = paths('decod', subject=subject, log=True,
                      analysis=analysis['name'] + '_toi')

    # Save classifier results
    with open(pkl_fname, 'wb') as f:
        pickle.dump([gat, analysis, sel, events], f)
