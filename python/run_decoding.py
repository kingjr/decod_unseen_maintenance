import os.path as op
import numpy as np
import pickle
from mne.decoding import GeneralizationAcrossTime

from utils import get_data, resample_epochs, decim
from meeg_preprocessing import setup_provenance

from config import (
    open_browser,
    data_path,
    pyoutput_path,
    results_dir,
    subjects,
    inputTypes,
    preproc,
    contrasts
)

report, run_id, results_dir, logger = setup_provenance(
    script=__file__, results_dir=results_dir)

for s, subject in enumerate(subjects):  # Loop across each subject
    print(subject)
    for typ in inputTypes:  # Input type ERFs or frequency power
        print(typ)
        if typ['name'] == 'erf':
            fname_appendix = ''
        else:
            fname_appendix = '_Tfoi_mtm_' + typ['name'][4:] + 'Hz'

        # define paths
        meg_fname = op.join(data_path, subject, 'preprocessed',
                            subject + '_preprocessed' + fname_appendix)
        bhv_fname = op.join(data_path, subject, 'behavior',
                            subject + '_fixed.mat')
        epochs, events = get_data(meg_fname, bhv_fname)

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
                fig, ('%s %s: (decoding)' % (subject, cond_name)), subject)

            fig = gat.plot(vmin=np.min(gat.scores_),
                           vmax=np.max(gat.scores_), show=False)
            report.add_figs_to_section(
                fig, ('%s %s: GAT' % (subject, cond_name)), subject)

            # Save contrast
            pkl_fname = op.join(
                pyoutput_path, subject, 'mvpas',
                '{}-decod_{}{}.pickle'.format(subject, contrast['name'],
                                              fname_appendix))

            # Save classifier results
            with open(pkl_fname, 'wb') as f:
                pickle.dump([gat, contrast, sel, events], f)


report.save(open_browser=open_browser)
