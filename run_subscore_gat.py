"""GAT subscore and regress as a function of visibility"""
import os
import numpy as np
import matplotlib.pyplot as plt
from jr.gat import subscore
from jr.stats import repeated_spearman
from jr.plot import (pretty_plot, pretty_gat, share_clim, pretty_axes,
                     pretty_decod, plot_sem)
from jr.utils import align_on_diag
from config import subjects, load, save, paths, report, tois
from base import stats
from conditions import analyses
analyses = [analysis for analysis in analyses if analysis['name'] in
            ['target_present', 'target_circAngle']]


def _subscore(analysis):
    """Subscore each analysis as a function of the reported visibility"""
    ana_name = analysis['name'] + '-vis'

    # don't recompute if not necessary
    fname = paths('score', analysis=ana_name)
    if os.path.exists(fname):
        return load('score', analysis=ana_name)

    # gather data
    all_scores = list()
    for subject in subjects:
        gat, _, events_sel, events = load('decod', subject=subject,
                                          analysis=analysis['name'])
        times = gat.train_times_['times']
        # remove irrelevant trials
        events = events.iloc[events_sel].reset_index()
        scores = list()
        gat.score_mode = 'mean-sample-wise'
        for vis in range(4):
            sel = np.where(events['detect_button'] == vis)[0]
            # If target present, we use the AUC against all absent trials
            if len(sel) < 5:
                scores.append(np.nan * np.empty(gat.y_pred_.shape[:2]))
                continue
            if analysis['name'] == 'target_present':
                sel = np.r_[sel,
                            np.where(events['target_present'] == False)[0]]  # noqa
            score = subscore(gat, sel)
            scores.append(score)
        all_scores.append(scores)
    all_scores = np.array(all_scores)

    # stats
    pval = list()
    for vis in range(4):
        pval.append(stats(all_scores[:, vis, :, :] - analysis['chance']))

    save([all_scores, pval, times],
         'score', analysis=ana_name, overwrite=True, upload=True)
    return all_scores, pval, times


def _correlate(analysis):
    """Correlate estimator prediction with a visibility reports"""
    ana_name = analysis['name'] + '-Rvis'

    # don't recompute if not necessary
    fname = paths('score', analysis=ana_name)
    if os.path.exists(fname):
        return load('score', analysis=ana_name)

    # gather data
    all_R = list()
    for subject in subjects:
        gat, _, events_sel, events = load('decod', subject=subject,
                                          analysis=analysis['name'])
        times = gat.train_times_['times']
        # remove irrelevant trials
        events = events.iloc[events_sel].reset_index()
        y_vis = np.array(events['detect_button'])

        # only analyse present trials
        sel = np.where(events['target_present'])[0]
        y_vis = y_vis[sel]
        gat.y_pred_ = gat.y_pred_[:, :, sel, :]

        # make 2D y_pred
        y_pred = gat.y_pred_.transpose(2, 0, 1, 3)[..., 0]
        y_pred = y_pred.reshape(len(y_pred), -1)
        # regress
        R = repeated_spearman(y_pred, y_vis)
        # reshape and store
        R = R.reshape(*gat.y_pred_.shape[:2])
        all_R.append(R)
    all_R = np.array(all_R)

    # stats
    pval = stats(all_R)

    save([all_R, pval, times], 'score', analysis=ana_name,
         overwrite=True, upload=True)
    return all_R, pval, times


def _duration_toi(analysis):
    """Estimate temporal generalization
    Re-align on diagonal, average per toi and compute stats."""
    ana_name = analysis['name'] + '-duration-toi'
    if os.path.exists(paths('score', analysis=ana_name)):
        return load('score', analysis=ana_name)
    all_scores, _, times = load('score', analysis=analysis['name'] + '-vis')
    # Add average duration
    n_subject = len(all_scores)
    all_score_tois = np.zeros((n_subject, 4, len(tois), len(times)))
    all_pval_tois = np.zeros((4, len(tois), len(times)))
    for vis in range(4):
        scores = all_scores[:, vis, ...]
        # align score on training time
        scores = [align_on_diag(score) for score in scores]
        # center effect
        scores = np.roll(scores, len(times) // 2, axis=2)
        for t, toi in enumerate(tois):
            toi = np.where((times >= toi[0]) & (times <= toi[1]))[0]
            score_toi = np.mean(scores[:, toi, :], axis=1)
            all_score_tois[:, vis, t, :] = score_toi
            all_pval_tois[vis, t, :] = stats(score_toi - analysis['chance'])
    save([all_score_tois, all_pval_tois, times], 'score', analysis=ana_name)
    return [all_score_tois, all_pval_tois, times]


# Main plotting
cmap = plt.get_cmap('bwr')
colors = cmap(np.linspace(0, 1, 4.))
for analysis in analyses:
    all_scores, score_pvals, times = _subscore(analysis)
    if 'circAngle' in analysis['name']:
        all_scores /= 2.
    # plot subscore GAT
    figs, axes = list(), list()
    for vis in range(4):
        fig, ax = plt.subplots(1, figsize=[7, 5.5])
        scores = all_scores[:, vis, ...]
        p_val = score_pvals[vis]
        pretty_gat(np.nanmean(scores, axis=0), times=times,
                   chance=analysis['chance'],
                   sig=p_val < .05, ax=ax, colorbar=False)
        ax.axvline(.800, color='k')
        ax.axhline(.800, color='k')
        axes.append(ax)
        figs.append(fig)
    share_clim(axes)
    fig_names = [analysis['name'] + str(vis) for vis in range(4)]
    report.add_figs_to_section(figs, fig_names, 'subscore')

    # plot GAT slices
    slices = np.arange(.100, .901, .200)
    fig, axes = plt.subplots(len(slices), 1, figsize=[5, 6],
                             sharex=True, sharey=True)
    for this_slice, ax in zip(slices, axes[::-1]):
        toi = np.where(times >= this_slice)[0][0]
        for vis in range(4)[::-1]:
            if vis not in [0, 3]:
                continue
            score = all_scores[:, vis, toi, :]
            sig = np.array(score_pvals)[vis, toi, :] < .05
            pretty_decod(score, times, color=colors[vis], ax=ax, sig=sig,
                         fill=True, chance=analysis['chance'])
        if ax != axes[-1]:
            ax.set_xlabel('')
        ax.axvline(.800, color='k')
        ax.axvline(this_slice, color='b')
    lim = np.nanmax(all_scores.mean(0))
    ticks = np.array([2 * analysis['chance'] - lim, analysis['chance'], lim])
    ticks = np.round(ticks * 100) / 100.
    ax.set_ylim(ticks[0], ticks[-1])
    ax.set_yticks(ticks)
    ax.set_yticklabels([ticks[0], 'chance', ticks[-1]])
    ax.set_xlim(-.100, 1.201)
    for ax in axes:
        ax.axvline(.800, color='k')
        if analysis['typ'] == 'regress':
            ax.set_ylabel('R', labelpad=-15)
        elif analysis['typ'] == 'categorize':
            ax.set_ylabel('AUC', labelpad=-15)
        else:
            ax.set_ylabel('rad.', labelpad=-15)
        ax.set_yticklabels(['', '', '%.2f' % ax.get_yticks()[2]])
    ax.set_xlabel('Times', labelpad=-10)
    report.add_figs_to_section([fig], [analysis['name']], 'slice_duration')

    # plot average slices toi to show duration
    all_durations, toi_pvals, times = _duration_toi(analysis)
    roll_times = times-times[len(times)//2]
    if 'circAngle' in analysis['name']:
        all_durations /= 2.
    fig, axes = plt.subplots(2, 1, sharex=True, sharey=True, figsize=[3, 6])
    for t, (toi, ax) in enumerate(zip(tois[1:-1], axes[::-1])):
        for vis in range(4)[::-1]:
            score = all_durations[:, vis, t+1, :]
            sig = toi_pvals[vis, t+1, :] < .05
            plot_sem(roll_times, score, color=colors[vis], alpha=.05, ax=ax)
            pretty_decod(np.nanmean(score, 0), roll_times,
                         color=colors[vis],
                         chance=analysis['chance'], sig=sig, ax=ax)
        if ax != axes[-1]:
            ax.set_xlabel('')
    mean_score = np.nanmean(all_durations[1:-1], axis=0)
    ticks = np.array([mean_score.min(), analysis['chance'], mean_score.max()])
    ticks = np.round(ticks * 100) / 100.
    ax.set_ylim(ticks[0], ticks[-1])
    ax.set_yticks(ticks)
    ax.set_yticklabels([ticks[0], 'chance', ticks[-1]])
    ax.set_xlim(-.700, .700)
    pretty_plot(ax)
    report.add_figs_to_section([fig], [analysis['name']], 'toi_duration')

    # plot sig scores and correlation with visibility
    _, R_pval, _ = _correlate(analysis)
    fig, ax = plt.subplots(1, figsize=[5, 6])
    for vis in range(4)[::-1]:
        if vis not in [0, 3]:  # for clarity only plot min max visibility
            continue
        pval = score_pvals[vis]
        sig = pval > .05
        xx, yy = np.meshgrid(times, times, copy=False, indexing='xy')
        ax.contourf(xx, yy, sig, levels=[-1, 0], colors=[colors[vis]],
                    aspect='equal')
    ax.contour(xx, yy, R_pval > .05, levels=[-1, 0], colors='k',
               aspect='equal')
    ax.axvline(.800, color='k')
    ax.axhline(.800, color='k')
    ticks = np.arange(-.100, 1.101, .100)
    ticklabels = [int(1e3 * ii) if ii in [0, .800] else '' for ii in ticks]
    ax.set_xlabel('Test Time')
    ax.set_ylabel('Train Time')
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(ticklabels)
    ax.set_yticklabels(ticklabels)
    ax.set_xlim(-.100, 1.100)
    ax.set_ylim(-.100, 1.100)
    pretty_plot(ax)
    ax.set_aspect('equal')
    report.add_figs_to_section([fig], [analysis['name']], 'R')


report.save()
