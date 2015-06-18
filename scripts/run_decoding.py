import numpy as np
import pickle
from mne.decoding import GeneralizationAcrossTime

from orientations.utils import load_epochs_events, resample_epochs, decim
from meeg_preprocessing.utils import setup_provenance

from config import (
    open_browser,
    paths,
    subjects,
    data_types,
    preproc,
    contrasts
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

        # Apply to each contrast
        for contrast in contrasts:
            print(contrast)
            # Find excluded trials
            exclude = np.any([
                events[x['cond']] == ii for x in contrast['exclude']
                for ii in x['values']],
                axis=0)

            # Select condition
            include = list()
            cond_name = contrast['include']['cond']
            for value in contrast['include']['values']:
                # Find included trials
                include.append(events[cond_name] == value)
            sel = np.any(include, axis=0) * (exclude == False)
            sel = np.where(sel)[0]

            # XXX reduce number or trials if too many XXX just for speed
            # if len(sel) > 400:
            #     import random
            #     random.shuffle(sel)
            #     sel = sel[0:400]

            y = np.array(events[cond_name].tolist())

            # Apply contrast
            gat = GeneralizationAcrossTime(clf=contrast['clf'], n_jobs=-1)
            gat.fit(epochs[sel], y=y[sel])
            gat.score(epochs[sel], y=y[sel], scorer=contrast['scorer'])

            # Plot
            fig = gat.plot_diagonal(show=False)
            report.add_figs_to_section(
                fig, ('%s %s %s: (diagonal)' %
                      (subject, cond_name, data_type)), subject)

            fig = gat.plot(vmin=np.min(gat.scores_),
                           vmax=np.max(gat.scores_), show=False)
            report.add_figs_to_section(
                fig, ('%s %s %s: GAT' % (subject, cond_name, data_type)),
                subject)

            # Save contrast
            pkl_fname = paths('decod', subject=subject, data_type=data_type,
                              analysis=contrast['name'])

            # Save classifier results
            with open(pkl_fname, 'wb') as f:
                pickle.dump([gat, contrast, sel, events], f)


report.save(open_browser=open_browser)
