import os.path as op
import numpy as np
import copy
# import matplotlib.pyplot as plt

from toolbox.utils import fill_betweenx_discontinuous, plot_eb
from meeg_preprocessing.utils import setup_provenance

from mne.stats import spatio_temporal_cluster_1samp_test

import pickle

###############################################################################

from config import (
    paths,
    subjects,
    open_browser,
    data_types,
    contrasts)


report, run_id, _, logger = setup_provenance(
    script=__file__, results_dir=paths('report'))

# subjects = [subjects[i] for i in range(20)] # XXX to be be removed

# XXX might go in config.py
# define subselections of interest
subselections = (dict(name='allTrials'),
                 dict(name='seenOnly'),
                 dict(name='unseenOnly'))

# Apply contrast to ERFs or frequency power
for data_type in data_types:
    print(data_type)

    # Loop across contrasts
    for contrast in contrasts:
        # Loop across different subselections of trials (subscoring)
        for subselection in subselections:
            # Uncomment to look at individual contrasts. contrast=contrasts[0]
            # DATA
            for s, subject in enumerate(subjects):
                print('load GAT %s %s %s' % (subject, contrast['name'],
                                             data_type))

                # define path to file to be loaded
                pkl_fname = paths('decod', subject=subject,
                                  data_type=data_type,
                                  analysis=contrast['name'])

                # retrieve classifier data
                with open(pkl_fname) as f:
                    gat, contrast, sel, events = pickle.load(f)

                gat_ = copy.deepcopy(gat)

                # define seen vs unseen
                vis = np.array(events['seen_unseen'][sel])

                # define subselection of trials of interest
                if subselection['name'] == 'allTrials':
                    # select all trials used for classification
                    subsel = [(vis[t] == True) | (vis[t] == False)
                              for t in np.arange(len(vis))]
                elif subselection['name'] == 'seenOnly':
                    # subscore only seen trials
                    subsel = [(vis[t] == True)
                              for t in np.arange(len(vis))]
                elif subselection['name'] == 'unseenOnly':
                    subsel = [(vis[t] == False)
                              for t in np.arange(len(vis))]

                # rescore subselection
                gat_.y_pred_ = gat.y_pred_[:, :, subsel]
                y = np.array(events[contrast['include']['cond']].tolist())
                gat_.score(y=y[subsel],
                           scorer=contrast['scorer'])

                # concatenate scores in a gat * subject array
                if s == 0:
                    scores = np.array(gat_.scores_)[:, :, None]
                else:
                    scores = np.concatenate((
                        scores, np.array(gat_.scores_)[:, :, None]),
                        axis=2)

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
                fig, '%s (%s) - %s: Decoding GAT' %
                (data_type, contrast['name'], subselection['name']),
                data_type)

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
                fig, '%s (%s) - %s: Decoding diag' %
                (data_type, contrast['name'], subselection['name']),
                data_type)

            # SAVE
            pkl_fname = paths('decod', subject='fsaverage',
                              data_type=data_type,
                              analysis=('stats_' + contrast['name'] +
                                        '-' + subselection['name']),
                              log=True)
            with open(pkl_fname, 'wb') as f:
                pickle.dump([scores, p_values], f)

report.save(open_browser=open_browser)
