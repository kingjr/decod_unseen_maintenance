"""GAT subscore as a function of visibility"""
import numpy as np
import matplotlib.pyplot as plt
from jr.gat import subscore
from jr.plot import pretty_gat, pretty_axes, share_clim
from config import subjects, load, save, paths, report
from base import stats
from conditions import analyses
analyses = [analysis for analysis in analyses if analysis['name'] in
            ['target_present', 'target_circAngle']]


def _subscore(analysis):
    """Subscore each analysis as a function of the reported visibility"""
    import os
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
    p_val = list()
    for vis in range(4):
        p_val.append(stats(all_scores[:, vis, :, :] - analysis['chance']))

    save([all_scores, p_val, times],
         'score', analysis=ana_name, overwrite=True, upload=True)
    return all_scores, p_val, times

for analysis in analyses:
    all_scores, p_vals, times = _subscore(analysis)
    # plot gat
    fig, axes = plt.subplots(1, 4)
    for vis, ax in enumerate(axes):
        scores = all_scores[:, vis, ...]
        p_val = p_vals[vis]
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
    report.add_figs_to_section([fig], [analysis['name']], analysis['name'])

report.save()
