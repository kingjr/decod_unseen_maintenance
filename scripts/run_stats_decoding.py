import sys
sys.path.insert(0, './')
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pickle
from itertools import product
import numpy as np

from mne.stats import spatio_temporal_cluster_1samp_test
from meeg_preprocessing.utils import setup_provenance
from base import plot_sem, plot_widths

from scripts.config import (
    paths,
    subjects,
    open_browser,
    data_types,
    analyses
)


report, run_id, _, logger = setup_provenance(
    script='scripts/run_stats_decoding.py', results_dir=paths('report'))


for data_type, analysis in product(data_types, analyses):
    print analysis['name']
    # DATA
    scores = list()
    y_pred = list()
    for subject in subjects:
        print subject
        # define path to file to be loaded
        score_fname = paths('score', subject=subject, data_type=data_type,
                            analysis=analysis['name'])
        with open(score_fname, 'rb') as f:
            out = pickle.load(f)
            gat = out[0]

        scores.append(gat.scores_)
        y_pred.append(gat.y_pred_)

    # STATS
    def stat_fun(x, sigma=0, method='relative'):
        from mne.stats import ttest_1samp_no_p
        t_values = ttest_1samp_no_p(x, sigma=sigma, method=method)
        t_values[np.isnan(t_values)] = 0
        return t_values

    def stats(X):
        T_obs_, clusters, p_values, _ = spatio_temporal_cluster_1samp_test(
            X,
            out_type='mask',
            stat_fun=stat_fun,
            n_permutations=2**10,
            threshold=dict(start=.1, step=.2),
            n_jobs=2)
        return p_values.reshape(X.shape[1:])

    scores = [score for score in scores if not np.isnan(score[0][0])]
    if len(scores) < 7:
        continue
    chance = analysis['chance']
    # FIXME should be in scorer
    if abs(chance - np.pi / 2) < 1e-4:
        scores = np.pi / 2 - np.array(scores)
        chance = 0.
    alpha = 0.05
    times = gat.train_times_['times'] * 1000
    # STATS
    p_values = stats(np.array(scores) - chance)
    diag_offdiag = scores - np.tile([np.diag(score) for score in scores],
                                    [len(times), 1, 1]).transpose(1, 0, 2)
    p_values_offdiag = stats(diag_offdiag)

    # PLOT
    sem = np.std(scores, axis=0) / np.sqrt(len(scores))
    ymin, ymax = chance + np.array([-1, 1]) * np.round(np.max(np.abs(
        np.diag(np.mean(scores, axis=0)) - chance) + sem) * 100) / 100
    widths = 3. * (p_values < .05)

    def pretty_plot(ax):
        ax.tick_params(colors='dimgray')
        ax.xaxis.label.set_color('dimgray')
        ax.yaxis.label.set_color('dimgray')
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        ax.spines['left'].set_color('dimgray')
        ax.spines['bottom'].set_color('dimgray')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    # ------ Plot GAT
    fig_gat, ax_gat = plt.subplots(1, figsize=[7, 5.5])
    im = ax_gat.imshow(np.mean(scores, axis=0),
                       extent=[min(times), max(times)] * 2,
                       cmap='RdBu_r', origin='lower', vmin=ymin, vmax=ymax)
    ax_gat.axhline(0, color='k')
    ax_gat.axvline(0, color='k')
    ax_gat.set_xlabel('Test times (ms.)')
    ax_gat.set_ylabel('Train times (ms.)')
    cb = plt.colorbar(im, ax=ax_gat, ticks=[ymin, chance, ymax])
    cb.ax.set_yticklabels(['%.2f' % ymin, 'Chance', '%.2f' % ymax],
                          color='dimgray')
    cb.ax.xaxis.label.set_color('dimgray')
    cb.ax.yaxis.label.set_color('dimgray')
    cb.ax.spines['left'].set_color('dimgray')
    cb.ax.spines['right'].set_color('dimgray')
    xx, yy = np.meshgrid(times, times, copy=False, indexing='xy')
    ax_gat.contour(xx, yy, p_values < alpha, colors='black', levels=[0],
                   linestyles='dotted')

    # ------ Plot Decoding
    fig_diag, ax_diag = plt.subplots(1, figsize=[5, 2.5])
    scores_diag = np.array([np.diag(score) for score in scores])
    widths_diag = np.diag(widths)
    plot_sem(times, scores_diag, color='k', ax=ax_diag)
    ax_diag.set_ylim(ymin, ymax)
    plot_widths(times, scores_diag.mean(0), widths_diag, ax=ax_diag, color='k')
    ax_diag.axhline(chance, linestyle='dotted', color='k')
    ax_diag.set_xlabel('Times (ms.)')
    pretty_plot(ax_diag)

    # ------ Plot times slices score
    plot_times = np.round(np.linspace(times[0], times[-1], 8)[1:-1] / 50) * 50
    fig_offdiag, axs = plt.subplots(len(plot_times), 1, figsize=[5, 6])
    for sel_time, ax in zip(plot_times, reversed(axs)):

        # plot shading between two lines if diag is sig. != from off diag
        idx = np.argmin(abs(times-sel_time))
        scores_off = np.array(scores)[:, idx, :]
        scores_sig = (scores_diag.mean(0) * (p_values_offdiag[idx] > .05) +
                      scores_off.mean(0) * (p_values_offdiag[idx] < .05))
        ax.fill_between(times, scores_diag.mean(0), scores_sig, color='yellow',
                        alpha=.5, linewidth=0)

        # change line width where sig != from change
        plot_sem(times, scores_diag, color='k', ax=ax)
        plot_sem(times, scores_off, color='b', ax=ax)
        plot_widths(times, scores_diag.mean(0), widths_diag, ax=ax, color='k')
        plot_widths(times, scores_off.mean(0), widths[idx], ax=ax, color='b')
        ax.axhline(chance, linestyle='dotted', color='k')
        ax.axvline(0, color='k')
        ax.plot([sel_time] * 2, [ymin, scores_off.mean(0)[idx]], color='b')
        ax.text(sel_time, ymin, '%i ms.' % sel_time,
                color='b', backgroundcolor='w', ha='center')

        ax.set_xlim(np.min(times), np.max(times))
        ax.set_ylim(ymin, ymax)
        ax.set_yticks([ymin, ymax])
        pretty_plot(ax)
        if ax != axs[-1]:
            ax.set_xticks([])
            ax.set_xlabel('')
            ax.spines['bottom'].set_visible(False)

    #  ------ Plot mean prediction # XXX

    # Report
    report.add_figs_to_section([fig_diag, fig_gat, fig_offdiag],
                               ['%s %s' % (data_type, analysis['name'])] * 3,
                               analysis['name'])
    # SAVE
    stats_fname = paths('score', subject='fsaverage', data_type=data_type,
                        analysis=('stats_' + analysis['name']),
                        log=True)
    with open(stats_fname, 'wb') as f:
        pickle.dump([scores, p_values], f)


report.save(open_browser=open_browser)
# upload_report(report)
