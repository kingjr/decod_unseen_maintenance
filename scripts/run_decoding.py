import numpy as np
import pickle
from mne.decoding import GeneralizationAcrossTime

from meeg_preprocessing.utils import setup_provenance

from orientations.utils import load_epochs_events
from base import resample_epochs, decim

from config import (
    open_browser,
    paths,
    subjects,
    data_types,
    preproc,
    analyses
)

report, run_id, _, logger = setup_provenance(
    script=__file__, results_dir=paths('report'))

for s, subject in enumerate(subjects):  # Loop across each subject
    print(subject)
    for data_type in data_types:  # Input type ERFs or frequency power
        epochs, events = load_epochs_events(subject, paths,
                                            data_type=data_type)

        # preprocess data for memory issue
        if 'resample' in preproc.keys():
            epochs = resample_epochs(epochs, preproc['resample'])
        if 'decim' in preproc.keys():
            epochs = decim(epochs, preproc['decim'])
        if 'crop' in preproc.keys():
            epochs.crop(preproc['crop']['tmin'],
                        preproc['crop']['tmax'])

        # Apply to each analysis
        for analysis in analyses:
            query, condition = analysis['query'], analysis['condition']
            sel = range(len(events)) if query is None \
                else events.query(query).index
            sel = [ii for ii in sel if ~np.isnan(events[condition][sel][ii])]
            y = np.array(events[condition], dtype=np.float32)

            print analysis['name'], np.unique(y[sel]), len(sel)

            if len(sel) == 0:
                logger.warning('%s: no epoch in %s for %s.' % (
                    subject, data_type, analysis['name']))
                continue

            # Apply analysis
            gat = GeneralizationAcrossTime(clf=analysis['clf'],
                                           cv=analysis['cv'],
                                           scorer=analysis['scorer'],
                                           n_jobs=-1)
            gat.fit(epochs[sel], y=y[sel])
            gat.score(epochs[sel], y=y[sel])

            # Save analysis
            pkl_fname = paths('decod', subject=subject, data_type=data_type,
                              analysis=analysis['name'], log=True)

            # Save classifier results
            with open(pkl_fname, 'wb') as f:
                pickle.dump([gat, analysis, sel, events], f)

            # Plot
            fig = gat.plot_diagonal(show=False)
            report.add_figs_to_section(fig, ('%s %s %s: (diagonal)' %
                                       (subject, data_type, analysis['name'])),
                                       analysis['name'])

            fig = gat.plot(vmin=np.min(gat.scores_),
                           vmax=np.max(gat.scores_), show=False)
            report.add_figs_to_section(fig, ('%s %s %s: GAT' % (
                                       subject, data_type, analysis['name'])),
                                       analysis['name'])

report.save(open_browser=open_browser)
