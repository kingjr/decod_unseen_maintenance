"""
This script is the second-order analogue of pipeline_subscore. Here we load the
GAT matrices obtained by subscoring and compare their differences against an
expected difference of zero.
"""

import os.path as op
import pickle
import numpy as np

# import mne
from mne.stats import spatio_temporal_cluster_1samp_test
from meeg_preprocessing.utils import setup_provenance

# from meeg_preprocessing.utils import setup_provenance
from toolbox.utils import fill_betweenx_discontinuous, plot_eb

from config import (
    paths('report'),
    pyoutput_path,
    subjects,
    data_types,
    open_browser,
    subscores,
    subscores2
)
from temp.pipeline_subscore import (sel_events,
                                    pkl_fname,
                                    gat_subscore,
                                    gat_order_y,
                                    mean_pred)

# Setup logs:
# mne.set_log_level('INFO')
report, run_id, _, logger = setup_provenance(
    script=__file__, results_dir=paths('report'))

for data_type in data_types:
    # logger.info(data_type)
    for subscore in subscores2:
        # logger.info(subscore['name'])
        # Gather subjects data
        scores_list = list()
        y_pred_list = list()
        for subject in subjects:
            # Load 1st GAT
            file = pkl_fname(typ, subject, subscore['contrast1'])
            with open(file) as f:
                gat1, _, sel, events = pickle.load(f)

            file = pkl_fname(typ, subject, subscore['contrast2'])
            with open(file) as f:
                gat2, _, sel, events = pickle.load(f)

            # Put back trials predictions in order according to original
            # selection
            gat = gat_order_y(gat, order=sel, n_pred=len(events))
            # TODO FIXME XXX
            gat.y_pred_ = np.array(gat.y_pred_)

            # Subscore overall from new selection
            sel = sel_events(events, subscore)
            key = subscore['include']['cond']
            y = np.array(events[key][sel].tolist())
            # HACK to skip few subjects that do not have enough trials in some
            # conditions of interest
            try:
                score = gat_subscore(gat, sel, y=y, scorer=subscore['scorer'])
            except:
                continue
            scores_list.append(score)

            # Keep mean prediction
            y_pred_list.append(mean_pred(gat, y))

        # STATS
        scores = np.array(scores_list)
        # ------ Parameters XXX to be transfered to config?
        # XXX JRK set stats level for each type of subscore
        alpha = 0.05
        n_permutations = 2 ** 11
        threshold = dict(start=.2, step=.2)

        X = scores - subscore['chance']

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
        # keep mean score for plotting
        for train in range(scores.shape[1]):
            for test in range(scores.shape[2]):
                gat.scores_[train][test] = np.mean(scores[:, train, test],
                                                   axis=0)
        # ------ Plot GAT
        fig = gat.plot(vmin=np.min(gat.scores_), vmax=np.max(gat.scores_),
                       show=False)
        ax = fig.axes[0]
        ax.contour(x, y, p_values < alpha, colors='black', levels=[0])
        # plt.show()
        report.add_figs_to_section(
            fig, '%s - %s (trained on %s): Decoding GAT' %
            (data_type, subscore['name'], subscore['contrast']), data_type)

        # ------ Plot Decoding
        fig = gat.plot_diagonal(show=False)
        ax = fig.axes[0]
        ymin, ymax = ax.get_ylim()

        scores_diag = np.array([np.diag(s) for s in scores])
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
            fig, '%s - %s (trained on %s): Decoding diag' %
            (data_type, subscore['name'], subscore['contrast']), data_type)

        # SAVE
        fname = op.join(
            pyoutput_path, 'fsaverage', 'decoding',
            'decod_stats_{}_{}.pickle'.format(data_type,
                                              subscore['name']))
        with open(fname, 'wb') as f:
            pickle.dump([scores, p_values], f)

report.save(open_browser=open_browser)
