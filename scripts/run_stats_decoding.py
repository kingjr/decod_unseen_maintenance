import sys
sys.path.insert(0, './')
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pickle
from itertools import product
import os.path as op
import numpy as np

from mne.stats import spatio_temporal_cluster_1samp_test
from meeg_preprocessing.utils import setup_provenance
from gat.utils import mean_ypred, subscore
from base import plot_sem, plot_widths

from scripts.config import (
    paths,
    subjects,
    open_browser,
    data_types,
    subscores as analyses
)
# from scripts.transfer_data import upload_report


report, run_id, _, logger = setup_provenance(
    script='scripts/run_stats_decoding.py', results_dir=paths('report'))

# Apply contrast to ERFs or frequency power
for data_type, analysis in product(data_types, analyses):
    # DATA
    scores = list()
    y_pred = list()
    for subject in subjects:
        # define path to file to be loaded
        score_fname = paths('score', subject=subject, data_type=data_type,
                            analysis=analysis['name'])
        if not op.exists(score_fname):
            # load
            gat_fname = paths('decod', subject=subject, data_type=data_type,
                              analysis=analysis['train_analysis'])
            # FIXME
            gat_fname = '/media/jrking/My Passport/Niccolo/' + gat_fname
            with open(gat_fname, 'rb') as f:
                gat, _, sel, events = pickle.load(f)

            # subsel
            query, condition = analysis['query'], analysis['condition']
            sel = range(len(events)) if query is None \
                else events.query(query).index
            sel = [ii for ii in sel if ~np.isnan(events[condition][sel][ii])]
            y = np.array(events[condition], dtype=np.float32)

            # subscore
            if len(np.unique(y[sel])) > 1:
                gat.scores_ = subscore(gat, sel, y[sel])
            else:
                gat.scores_ = np.nan * np.array(gat.scores_)
            gat.y_pred_ = mean_ypred(gat, classes=np.unique(y))

            # optimize memory
            gat.estimators_ = list()

            # save
            with open(score_fname, 'wb') as f:
                pickle.dump([gat, analysis, sel, events], f)
        else:
            with open(score_fname, 'rb') as f:
                gat, _, sel, events = pickle.load(f)

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
            n_permutations=128,
            threshold=dict(start=2, step=2.),
            n_jobs=1)

        # ------ combine clusters & retrieve min p_values for each feature
        return p_values.reshape(X.shape[1:])

    chance = analysis['chance']
    alpha = 0.05
    times = gat.train_times_['times'] * 1000

    p_values = stats(np.array(scores) - chance)
    diag_offdiag = scores - np.tile([np.diag(score) for score in scores],
                                    [len(times), 1, 1]).transpose(1, 0, 2)
    p_values_offdiag = stats(diag_offdiag)

    # PLOT
    sem = np.std(scores, axis=0) / np.sqrt(len(scores))
    ymin = round(np.min(np.mean(scores, axis=0) - sem) * 5) / 5
    ymax = round(np.max(np.mean(scores, axis=0) + sem) * 5) / 5

    widths = 3. * (p_values < .05)

    def pretty_plot(ax):
        ax.tick_params(colors='dimgray')
        ax.xaxis.label.set_color('dimgray')
        ax.yaxis.label.set_color('dimgray')
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
    cb.ax.set_yticklabels(['%.2f' % ymin, 'Chance', '%.2f' % ymax])
    pretty_plot(ax_gat)
    xx, yy = np.meshgrid(times, times, copy=False, indexing='xy')
    ax_gat.contour(xx, yy, p_values < alpha, colors='black', levels=[0])

    # ------ Plot Decoding
    fig_diag, ax_diag = plt.subplots(1, figsize=[5, 2.5])
    scores_diag = np.array([np.diag(score) for score in scores])
    widths_diag = np.diag(widths)
    plot_sem(times, scores_diag, color='k', ax=ax_diag)
    ax_diag.set_ylim(ymin, ymax)
    plot_widths(times, scores_diag.mean(0), widths_diag, ax=ax_diag, color='k')
    ax_diag.axhline(chance, linestyle='--', color='k')
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
        ax.axhline(chance, linestyle='--', color='k')
        ax.axvline(0, color='k')
        ax.plot([sel_time] * 2, [ymin, scores_off.mean(0)[idx]], color='b')
        ax.text(sel_time, ymin, '%i ms.' % sel_time,
                color='b', backgroundcolor='w', ha='center')

        ax.set_xlim(np.min(times), np.max(times))
        ax.set_ylim(ymin, ymax)
        if ax != axs[-1]:
            ax.set_xticks([])
            ax.set_xlabel('')
            ax.spines['bottom'].set_visible(False)

        ax.tick_params(colors='dimgray')
        ax.xaxis.label.set_color('dimgray')
        ax.yaxis.label.set_color('dimgray')
        ax.spines['left'].set_color('dimgray')
        ax.spines['bottom'].set_color('dimgray')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

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
