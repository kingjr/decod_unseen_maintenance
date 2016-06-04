"""One of the control analyses (Decoding Bias towards target for virtual
trials) requires making angle predictions for each absent trials. The main
run_decoding.py script does not considered these trials, so we recompute them
here"""

import numpy as np
from config import subjects, load, save
from conditions import analyses
analysis = [ana for ana in analyses if ana['name'] == 'target_circAngle'][0]

for s, subject in enumerate(subjects):  # Loop across each subject
    print(subject)
    # load MEG data
    epochs = load('epochs_decim', subject=subject, preload=True)
    events = load('behavior', subject=subject)

    # Load classifier
    gat, analysis, ana_sel, events = load('decod', subject=subject,
                                          analysis=analysis['name'])

    sel = np.where(np.array(events.target_present) == False)[0]
    y = np.array(events.probe_circAngle, dtype=np.float32)
    y[::2] = (y[::2] - np.pi/3.) % (2 * np.pi)
    y[1::2] = (y[1::2] + np.pi/3.) % (2 * np.pi)

    # Apply analysis
    gat.predict_mode = 'mean-prediction'
    gat.predict(epochs[sel])
    gat.score(epochs[sel], y=y[sel])

    # Save classifier results
    save([gat, analysis, sel, events], 'decod', subject=subject,
         analysis=analysis['name'] + '_absent')
