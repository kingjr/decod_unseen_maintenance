import numpy as np
import pickle
from orientations.utils import load_epochs_events
from scripts.config import paths, subjects, analyses
analysis = [ana for ana in analyses if ana['name'] == 'target_circAngle'][0]

for s, subject in enumerate(subjects):  # Loop across each subject
    print(subject)
    # load MEG data
    epochs, events = load_epochs_events(subject, paths)
    epochs.crop(-.1, 1.100)  # XXX
    # Load classifier
    pkl_fname = paths('decod', subject=subject, analysis=analysis['name'])
    with open(pkl_fname, 'rb') as f:
        gat, analysis, ana_sel, events = pickle.load(f)

    sel = np.where(np.array(events.target_present) == False)[0]
    y = np.array(events.probe_circAngle, dtype=np.float32)
    y[::2] = (y[::2] - np.pi/3.) % (2 * np.pi)
    y[1::2] = (y[1::2] + np.pi/3.) % (2 * np.pi)

    # Apply analysis
    gat.predict_mode = 'mean-prediction'
    gat.predict(epochs[sel])
    gat.score(epochs[sel], y=y[sel])

    # Save classifier results
    pkl_fname = paths('decod', subject=subject,
                      analysis=analysis['name'] + '_absent', log=True)
    with open(pkl_fname, 'wb') as f:
        pickle.dump([gat, analysis, sel, events], f)
