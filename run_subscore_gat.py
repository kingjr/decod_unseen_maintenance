"""GAT subscore and regress as a function of visibility"""
import os
import numpy as np
import matplotlib.pyplot as plt
from jr.gat import subscore
from jr.stats import repeated_spearman
from jr.plot import (pretty_plot, pretty_gat, share_clim, pretty_axes,
                     pretty_decod, plot_sem, bar_sem)
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


def _average_ypred_toi(gat, toi, analysis):
    """Average single trial predictions of each time point in a given TOI"""
    from jr.gat import get_diagonal_ypred
    y_pred = np.transpose(get_diagonal_ypred(gat), [1, 0, 2])
    times = gat.train_times_['times']
    # select time sample
    toi = np.where((times >= toi[0]) & (times <= toi[1]))[0]
    if 'circAngle' in analysis['name']:
        # weight average by regressor radius
        cos = np.cos(y_pred[:, toi, 0])
        sin = np.sin(y_pred[:, toi, 0])
        radius = y_pred[:, toi, 1]
        y_pred = np.angle(np.median((cos + 1j * sin) * radius, axis=1))
    else:
        y_pred = np.median(y_pred[:, toi], axis=1)
    return y_pred


def _subscore_toi(y_pred, events, analysis, factor):
    """Subscore each visibility
    y_pred : shape(n_trials,)
    events : dataframe, shape(n_trials,)
    analysis : dict(name='target_present' | 'target_circAngle', scorer)
    key: 'detect_button' | 'target_contrast'
    values: range(4) | [.50, .75, 1.]
    """
    factors = dict(visibility=['detect_button', range(4)],
                   contrast=['target_contrast', [.50, .75, 1.]])
    key, values = factors[factor]
    y_true = events[analysis['name']]

    scores = np.nan * np.zeros(len(values))

    for ii, value in enumerate(values):
        # select trials e.g. according to visibility or contrast
        sel = np.where(events[key] == value)[0]

        # for clarity, add all absent trials in target_present analysis
        if analysis['name'] == 'target_present':
            sel = np.r_[sel,
                        np.where(events['target_present'] == False)[0]]  # noqa

        # skip if not enough trials
        if len(sel) < 5 or len(np.unique(y_true[sel])) < 2:
            continue

        # score
        scores[ii] = analysis['scorer'](y_true=y_true[sel], y_pred=y_pred[sel])
    return scores


def _subregress_toi(y_pred, events, analysis, factor):
    """Correlate single trial error with factor"""
    factors = dict(visibility=['detect_button', range(4)],
                   contrast=['target_contrast', [.50, .75, 1.]])
    key, values = factors[factor]
    y_true = np.array(events[analysis['name']])

    # Check dimensiality
    y_pred = np.squeeze(y_pred)
    if (y_pred.ndim != 1) or len(y_pred) != len(y_true):
        raise ValueError

    # Compute single trial error
    if 'circAngle' in analysis['name']:
        y_error = (y_pred - y_true) % (2 * np.pi)
        y_error = np.abs(np.pi - y_error)
    else:
        y_error = np.abs(y_pred - y_true)

    # Do the prediction vary across visibilities/contrasts?
    sel = np.where(events[key] >= values[0])[0]
    R = repeated_spearman(y_error[sel], events[key][sel])
    return R


def _analyze_toi(analysis):
    """Subscore each analysis as a function of the reported visibility"""
    ana_name = analysis['name'] + '-toi'

    # don't recompute if not necessary
    fname = paths('score', analysis=ana_name)
    if os.path.exists(fname):
        return load('score', analysis=ana_name)

    # gather data
    n_subject = 20
    scores = dict(visibility=np.zeros((n_subject, len(tois), 4)),
                  contrast=np.zeros((n_subject, len(tois), 3)))
    R = dict(visibility=np.zeros((n_subject, len(tois))),
             contrast=np.zeros((n_subject, len(tois))),)
    for s, subject in enumerate(subjects):
        gat, _, events_sel, events = load('decod', subject=subject,
                                          analysis=analysis['name'])
        events = events.iloc[events_sel].reset_index()
        for t, toi in enumerate(tois):
            # Average predictions on single trials across time points
            y_pred = _average_ypred_toi(gat, toi, analysis)
            # visibility
            for factor in ['visibility', 'contrast']:
                # subscore per condition (e.g. each visibility rating)
                scores[factor][s, t, :] = _subscore_toi(y_pred, events,
                                                        analysis, factor)
                # correlate residuals with factor
                R[factor][s, t] = _subregress_toi(y_pred, events,
                                                  analysis, factor)

    save([scores, R], 'score', analysis=ana_name, overwrite=True, upload=True)
    return [scores, R]


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
colors = dict(visibility=plt.get_cmap('bwr')(np.linspace(0, 1, 4.)),
              contrast=plt.get_cmap('hot_r')([.5, .75, 1.]))

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
            pretty_decod(score, times, color=colors['visibility'][vis],
                         ax=ax, sig=sig, fill=True, chance=analysis['chance'])
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
            plot_sem(roll_times, score, color=colors['visibility'][vis],
                     alpha=.05, ax=ax)
            pretty_decod(np.nanmean(score, 0), roll_times,
                         color=colors['visibility'][vis],
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
        ax.contourf(xx, yy, sig, levels=[-1, 0],
                    colors=[colors['visibility'][vis]], aspect='equal')
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

    # plot and report subscore per visibility and contrast for each toi
    from scipy.stats import wilcoxon
    from jr.utils import table2html
    toi_scores, toi_R = _analyze_toi(analysis)

    # report angle error because orientation
    if 'circAngle' in analysis['name']:
        toi_scores['visibility'] /= 2.
        toi_scores['visibility'] /= 2.

    def quick_stats(x, chance):
        x = x[np.where(~np.isnan(x))[0]]
        text = '[%.3f+/-%.3f, p=%.4f]'
        m = np.nanmean(x)
        sem = np.nanstd(x) / np.sqrt(len(x))
        pval = wilcoxon(x - chance)[1]
        return text % (m, sem, pval)

    # subscore visibility then subscore contrast
    for factor in ('visibility', 'contrast'):
        score = toi_scores[factor]
        R = toi_R[factor]
        color = colors[factor]
        n_subscore = score.shape[2]

        fig, axes = plt.subplots(1, len(tois), sharey=True, figsize=[6, 2])
        table = np.empty((len(tois), n_subscore + 3), dtype=object)

        # effect in each toi
        for toi, ax in enumerate(axes):
            # plot subscores
            bar_sem(range(n_subscore), score[:, toi, :] - analysis['chance'],
                    bottom=analysis['chance'], ax=ax, color=color)

            # compute stats for each subscore
            for ii in range(n_subscore):
                score_ = score[:, toi, ii]
                table[toi, ii] = quick_stats(score[:, toi, ii],
                                             chance=analysis['chance'])

            # difference min max (e.g. seen - unseen)
            table[toi, n_subscore] = quick_stats(score[:, toi, -1] -
                                                 score[:, toi, 0], chance=0.)
            # regression across scores: single trials
            table[toi, n_subscore + 1] = quick_stats(R[:, toi], chance=0.)

            # regression across scores: not single trials
            adhoc_R = [repeated_spearman(range(n_subscore), subject)[0]
                       for subject in score[:, toi, :]]
            table[toi, n_subscore + 2] = quick_stats(np.array(adhoc_R), 0.)

        pretty_axes(axes, xticks=[])
        report.add_figs_to_section([fig], [factor], analysis['name'])

        table = np.c_[[str(t) for t in tois], table]

        headers = [factor + str(ii) for ii in range(n_subscore)]
        table = np.vstack((np.r_[[''], headers, ['max-min'],
                                 ['R', 'R (not single trials)']], table))
        report.add_htmls_to_section(table2html(table), 'subscore_' + factor,
                                    analysis['name'])

        # Does the effect vary over time
        # e.g. seen-unseen stronger in early vs late
        table = np.empty((len(tois), len(tois)), dtype=object)
        for t1 in range(len(tois)):
            for t2 in range(len(tois)):
                table[t1, t2] = quick_stats(R[:, t1] - R[:, t2], 0.)
        report.add_htmls_to_section(table2html(table), 'toi_toi_' + factor,
                                    analysis['name'])

    # Do contrast and visibility affect different time points? FIXME:
    # Not the right test?
    table = np.empty(len(tois), dtype=object)
    for toi in range(len(tois)):
        table[toi] = quick_stats(toi_R['visibility'][:, toi] -
                                 toi_R['contrast'][:, toi], 0.)
    table = np.vstack(([str(toi) for toi in tois], table))
    report.add_htmls_to_section(table2html(table),
                                'toi_subscore_', analysis['name'])

report.save()
