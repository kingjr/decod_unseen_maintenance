import os.path as op
import numpy as np
# import matplotlib.pyplot as plt

from toolbox.utils import fill_betweenx_discontinuous, plot_eb
from meeg_preprocessing import setup_provenance

from mne.stats import spatio_temporal_cluster_1samp_test

import pickle

###############################################################################

from config import (
    subjects,
    pyoutput_path,
    results_dir,
    open_browser,
    inputTypes,
    contrasts
)


report, run_id, results_dir, logger = setup_provenance(
    script=__file__, results_dir=results_dir)

# subjects = [subjects[i] for i in range(20)] # XXX to be be removed

# Apply contrast to ERFs or frequency power
for typ in inputTypes:
    # TODO: erf and all freqs in inputTypes, remove freq loop
    print(typ)

    # loop only once if ERF and across all frequencies if frequency power
    for freq in typ['values']:
        print(freq)

        # Loop across contrasts
        for contrast in contrasts:

            # Uncomment to look at individual contrasts. contrast=contrasts[0]
            # DATA
            for s, subject in enumerate(subjects):
                print('load GAT %s %s %s %s' % (subject, contrast['name'],
                                                typ['name'], freq))

                # define meg_path appendix
                if typ['name'] == 'erf':
                    fname_appendix = '_'  # XXX REMOVE NEXT TIME
                elif typ['name'] == 'power':
                    fname_appendix = op.join('_Tfoi_mtm_', freq, 'Hz')

                # define path to file to be loaded
                pkl_fname = op.join(
                    pyoutput_path, subject, 'mvpas',
                    '{}-decod_{}{}.pickle'.format(
                        subject, contrast['name'], fname_appendix))

                with open(pkl_fname) as f:
                    gat, contrast, _, _ = pickle.load(f)
                if s == 0:
                    scores = np.array(gat.scores_)[:, :, None]
                else:
                    scores = np.concatenate((
                        scores, np.array(gat.scores_)[:, :, None]), axis=2)

            # STATS
            # ------ Parameters XXX to be transfered to config?
            # XXX JRK set stats level for each type of analysis
            alpha = 0.05
            n_permutations = 2 ** 11
            threshold = dict(start=.2, step=.2)

            X = scores.transpose((2, 0, 1)) - contrast['chance']

            # ------ Run stats
            T_obs_, clusters, p_values, _ = spatio_temporal_cluster_1samp_test(
                X,
                out_type='mask',
                n_permutations=n_permutations,
                connectivity=None,
                threshold=threshold,
                n_jobs=-1)

            # ------ combine clusters & retrieve min p_values for each feature
            cluster_p = [clusters[c] * p for c, p in enumerate(p_values)]
            p_values = np.min(np.logical_not(clusters) + cluster_p, axis=0)
            x, y = np.meshgrid(gat.train_times['times_'],
                               gat.test_times_['times_'][0],
                               copy=False, indexing='xy')

            # PLOT
            # ------ Plot GAT
            gat.scores_ = np.mean(scores, axis=2)
            fig = gat.plot(vmin=np.min(gat.scores_), vmax=np.max(gat.scores_),
                           show=False)
            ax = fig.axes[0]
            ax.contour(x, y, p_values < alpha, colors='black', levels=[0])
            # plt.show()
            report.add_figs_to_section(
                fig, '%s (%s): Decoding ' % (typ['name'], contrast['name']),
                typ['name'])

            # ------ Plot Decoding
            fig = gat.plot_diagonal(show=False)
            ax = fig.axes[0]
            ymin, ymax = ax.get_ylim()

            scores_diag = np.array([np.diag(s) for s in
                                    scores.transpose((2, 0, 1))])
            times = gat.train_times['times_']

            sig_times = times[np.where(np.diag(p_values) < alpha)[0]]
            sfreq = (times[1] - times[0]) / 1000
            fill_betweenx_discontinuous(ax, ymin, ymax, sig_times, freq=sfreq,
                                        color='orange')

            plot_eb(times, np.mean(scores_diag, axis=0),
                    np.std(scores_diag, axis=0) / np.sqrt(scores.shape[2]),
                    ax=ax, color='blue')
            # plt.show()
            report.add_figs_to_section(
                fig, '%s (%s): Decoding ' % (typ['name'], contrast['name']),
                typ['name'])

            # SAVE
            pkl_fname = op.join(
                pyoutput_path, 'fsaverage', 'decoding',
                'decod_stats_{}{}.pickle'.format(contrast['name'],
                                                 fname_appendix))
            with open(pkl_fname, 'wb') as f:
                pickle.dump([scores, p_values], f)

report.save(open_browser=open_browser)
