"""GAT subscore and regress as a function of visibility"""
import os
import numpy as np
import matplotlib.pyplot as plt
from jr.gat import subscore
from jr.stats import repeated_spearman
from jr.plot import pretty_plot, pretty_gat, share_clim, pretty_axes
from config import subjects, load, save, paths, report
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


cmap = plt.get_cmap('bwr')
colors = cmap(np.linspace(0, 1, 4.))
for analysis in analyses:
    all_scores, score_pvals, times = _subscore(analysis)
    # plot subscore GAT
    fig, axes = plt.subplots(1, 4)
    for vis, ax in enumerate(axes):
        scores = all_scores[:, vis, ...]
        p_val = score_pvals[vis]
        if 'circAngle' in analysis['name']:
            scores /= 2.
        pretty_gat(np.nanmean(scores, axis=0), times=times,
                   chance=analysis['chance'],
                   sig=p_val < .05, ax=ax, colorbar=False)
        ax.axvline(.800, color='k')
        ax.axhline(.800, color='k')
    share_clim(axes)
    ticks = np.arange(-.100, 1.101, .100)
    ticklabels = [int(1e3 * ii) if ii in [0, .800] else '' for ii in ticks]
    pretty_axes(axes, xlabel='Test Times', ylabel='Train Times',
                xticks=ticks, yticks=ticks,
                xticklabels=ticklabels, yticklabels=ticklabels)
    report.add_figs_to_section([fig], [analysis['name']], 'subscore')

    # plot correlation with visibility
    _, R_pval, _ = _correlate(analysis)
    fig, ax = plt.subplots(1, figsize=[4, 4])
    for vis in range(4)[::-1]:
        pval = score_pvals[vis]
        sig = pval > .05
        xx, yy = np.meshgrid(times, times, copy=False, indexing='xy')
        ax.contourf(xx, yy, sig, levels=[-1, 0], colors=[colors[vis]],
                    aspect='image')
    # ax.contour(xx, yy, R_pval > .05, levels=[-1, 0], colors='k',
    #            aspect='image')
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
    report.add_figs_to_section([fig], [analysis['name']], 'R')

report.save()
