import os.path as op
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from toolbox.utils import cluster_stat, fill_betweenx_discontinuous, plot_eb
from meeg_preprocessing import setup_provenance

from mne.stats import spatio_temporal_cluster_1samp_test

import mne
import pickle

###############################################################################

from config import (
    data_path,
    subjects,
    results_dir,
    open_browser,
    inputTypes,
    open_browser
)


report, run_id, results_dir, logger = setup_provenance(
    script=__file__, results_dir=results_dir)

subjects = [subjects[i] for i in range(20)] # XXX to be be removed

# Apply contrast to ERFs or frequency power
for typ in inputTypes:
    print(typ)

    # loop only once if ERF and across all frequencies of interest if frequency power
    for freq in typ['values']:
        print(freq)

        # Define classification types available to loop across (SVR and SVC)
        clf_types=typ['clf']

        # Loop across classifier type (SVC or SVR)
        for clf_type in clf_types:
            # Loop across contrasts
            for contrast in clf_type['contrasts']:

                # Uncomment to look at individual contrasts. contrast=contrasts[0]
                # DATA
                for s, subject in enumerate(subjects):
                    print('load GAT %s %s %s %s %s' %
                        (subject, clf_type['name'], contrast['name'],typ['name'], freq))

                    # define meg_path appendix
                    if typ['name']=='erf':
                        fname_appendix = ''
                    elif typ['name']=='power':
                        fname_appendix = op.join('_Tfoi_mtm_',freq,'Hz')

                    # define path to file to be loaded
                    cond_name = contrast['include']['cond']
                    if False:
                        pkl_fname = op.join(data_path, subject, 'mvpas',
                            '{}-decod_{}.pickle'.format(subject, cond_name))
                    else:
                        pkl_fname = op.join(data_path, subject, 'mvpas',
                            '{}-decod_{}_{}{}.pickle'.format(subject, cond_name,clf_type['name'],fname_appendix))

                    with open(pkl_fname) as f:
                        gat, contrast = pickle.load(f)
                    if s == 0:
                        scores = np.array(gat.scores_)[:, :, None]
                    else:
                        scores = np.concatenate((scores,
                                                 np.array(gat.scores_)[:, :, None]),
                                                axis=2)

                # STATS
                # ------ Parameters XXX to be transfered to config?
                alpha = 0.05
                n_permutations = 2 ** 11
                threshold = dict(start=.2, step=.2)

                X = scores.transpose((2, 0, 1)) - .5

                # ------ Run stats
                T_obs_, clusters, p_values, _ = spatio_temporal_cluster_1samp_test(
                                                       X,
                                                       out_type='mask',
                                                       n_permutations=n_permutations,
                                                       connectivity=None,
                                                       threshold=threshold,
                                                       n_jobs=-1)

                # ------ combine clusters and retrieve min p_values for each feature
                p_values = np.min(np.logical_not(clusters) +
                                  [clusters[c] * p for c, p in enumerate(p_values)],
                                  axis=0)
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
                #plt.show()
                report.add_figs_to_section(fig, '%s %s (%s): Decoding ' % (typ['name'],
                                           clf_type['name'],cond_name), typ['name'])

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
                        np.std(scores_diag, axis=0) / np.sqrt(scores.shape[0]),
                        ax=ax, color='blue')
                #plt.show()
                report.add_figs_to_section(fig, '%s %s (%s): Decoding ' % (typ['name'],
                                           clf_type['name'],cond_name), typ['name'])
        break
    break

report.save(open_browser=open_browser)
